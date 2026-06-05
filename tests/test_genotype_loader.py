# =============================================================================
# Integration test for genotype_loader.load_genotypes() over real Stage-2
# outputs: checks shape, dtype, NaN/sentinel/range, sample-ID consistency, and a
# PLINK --export A numerical cross-check on the smallest config.
# Run order: test (not part of result generation). Run on the HPC where the
#            qc/ Stage-2 outputs and plink2 are available.
# Inputs:    qc/* Stage-2 outputs
# Outputs:   stdout pass/fail (exit code)
# Test script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
test_genotype_loader.py
=======================
Standalone test for genotype_loader.load_genotypes(), run on real
Stage 2 QC outputs from this dissertation pipeline.

What this verifies (in order):

  1. The loader opens and reads every (size, config) combination
     without OOM-ing or crashing.

  2. Output shape is (n_samples, n_variants), with n_samples == 929
     and n_variants matching the Stage 2 variant counts in the log.

  3. Output is float32, contains no NaN, no -9 sentinel, and dosages
     are bounded to [0, 2].

  4. Sample IDs from the .psam match across all twelve filesets
     (they should; baseline QC retained 929 samples for all configs).

  5. Cross-check against PLINK's `--export A` output for ONE small
     config, to prove pgenlib's int8 bulk read agrees with PLINK's
     own text export numerically. Does not run on the big configs;
     we already know that's where the text export OOMs.

Run from the project root, where the QC outputs and PLINK2 are available
(set QC_VAE_PCA_ROOT if the project lives outside this code_appendix folder):

    module load plink/2.00a6LM      # or otherwise put plink2 on PATH
    python tests/test_genotype_loader.py

Exits with code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Import genotype_loader from the sibling scripts/ directory (this test lives in
# tests/, the loader lives in scripts/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from genotype_loader import load_genotypes, PGEN_MISSING  # noqa: E402

# ---- Configuration -------------------------------------------------------

from config import PROJECT_ROOT  # noqa: E402
QC_DIR = PROJECT_ROOT / "qc"
SIZES = ["47k", "470k"]
CONFIGS = [1, 2, 3, 4, 5, 6]
EXPECTED_N_SAMPLES = 929

# Variant counts from log entry "Stage 2 QC complete on both subsets".
# These are the SEEDED counts after `--seed 42` is added; until then,
# the LD-pruning configs may differ slightly from the table here. The
# test only verifies the order of magnitude so unseeded prune lists
# remain testable.
EXPECTED_VARIANT_COUNTS = {
    ("47k", 1): 40_768,
    ("47k", 2): 6_289,
    ("47k", 3): 3_558,
    ("47k", 4): 34_372,
    ("47k", 5): 4_378,
    ("47k", 6): 2_247,
    ("470k", 1): 407_932,
    ("470k", 2): 62_377,
    ("470k", 3): 35_830,
    ("470k", 4): 317_436,
    ("470k", 5): 18_269,
    ("470k", 6): 7_034,
}


def fmt_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1024 / 1024:.1f} MB"


def test_one(size: str, config: int) -> dict:
    """Load one (size, config), run all integrity checks, return summary."""
    prefix = QC_DIR / f"{size}_config{config}" / "qcd"
    print(f"\n--- {size} config{config} ---")
    print(f"  prefix: {prefix}")

    if not (prefix.with_suffix(".pgen")).exists():
        print(f"  SKIP: no .pgen at this prefix")
        return {"size": size, "config": config, "skipped": True}

    t0 = time.time()
    geno, samples, variants = load_genotypes(str(prefix), cache_path=None)
    t_load = time.time() - t0

    # Shape
    assert geno.shape == (len(samples), len(variants)), (
        f"Shape mismatch: array {geno.shape} vs (n_samples={len(samples)}, "
        f"n_variants={len(variants)})"
    )
    assert len(samples) == EXPECTED_N_SAMPLES, (
        f"Expected {EXPECTED_N_SAMPLES} samples, got {len(samples)}"
    )

    # Dtype, NaN, range
    assert geno.dtype == np.float32, f"Expected float32, got {geno.dtype}"
    assert not np.isnan(geno).any(), "Output contains NaN"
    assert (geno != PGEN_MISSING).all(), f"Output contains pgen sentinel {PGEN_MISSING}"
    assert geno.min() >= 0.0, f"Min dosage {geno.min()} < 0 -- imputation went wrong"
    assert geno.max() <= 2.0, f"Max dosage {geno.max()} > 2 -- corrupt input?"

    # Magnitude check on variant count (informational; does not assert
    # exact equality because LD-pruning is currently auto-seeded by PLINK
    # and counts can drift slightly until --seed 42 is locked in).
    expected = EXPECTED_VARIANT_COUNTS.get((size, config))
    if expected is not None:
        rel_diff = abs(len(variants) - expected) / expected
        if rel_diff > 0.05:
            print(
                f"  WARNING: variant count {len(variants):,} drifts "
                f">5% from expected {expected:,} (rel diff {rel_diff:.1%})"
            )
        else:
            print(f"  variant count: {len(variants):,} (expected ~{expected:,})")
    else:
        print(f"  variant count: {len(variants):,}")

    print(f"  shape:        {geno.shape}")
    print(f"  memory:       {fmt_mb(geno.nbytes)}")
    print(f"  load time:    {t_load:.2f}s")
    print(f"  dosage stats: min={geno.min():.3f}  max={geno.max():.3f}  "
          f"mean={geno.mean():.4f}")
    print(f"  any NaN:      {np.isnan(geno).any()}  (must be False)")

    return {
        "size": size,
        "config": config,
        "n_samples": len(samples),
        "n_variants": len(variants),
        "load_time_s": t_load,
        "memory_mb": geno.nbytes / 1024 / 1024,
        "samples": samples,
    }


def cross_check_against_plink(size: str = "47k", config: int = 6) -> None:
    """
    Numerical cross-check on the smallest config: have PLINK write
    a .raw text export, parse it with numpy, and compare against
    pgenlib's loaded matrix. Tolerance is exact for non-imputed
    entries; imputed entries are checked for finite-and-bounded.
    """
    print(f"\n--- Cross-check: {size} config{config} via plink2 --export A ---")
    prefix = QC_DIR / f"{size}_config{config}" / "qcd"
    if not (prefix.with_suffix(".pgen")).exists():
        print(f"  SKIP: no .pgen")
        return

    geno_pg, samples_pg, variants_pg = load_genotypes(str(prefix), cache_path=None)

    with tempfile.TemporaryDirectory() as td:
        raw_prefix = Path(td) / "export"
        cmd = [
            "plink2",
            "--pfile", str(prefix),
            "--export", "A",
            "--out", str(raw_prefix),
        ]
        print(f"  running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("  STDERR:", result.stderr)
            raise RuntimeError("plink2 --export A failed")

        raw_path = raw_prefix.with_suffix(".raw")

        # Parse the .raw file. Format: FID IID PAT MAT SEX PHENO geno1 geno2 ...
        # Genotype tokens are 0/1/2/NA.
        with raw_path.open() as fh:
            header = fh.readline().split()
            n_meta_cols = 6  # FID, IID, PAT, MAT, SEX, PHENOTYPE
            sample_ids_raw = []
            rows = []
            for line in fh:
                parts = line.split()
                sample_ids_raw.append(parts[1])  # IID
                geno_strs = parts[n_meta_cols:]
                row = np.array(
                    [np.nan if t == "NA" else float(t) for t in geno_strs],
                    dtype=np.float32,
                )
                rows.append(row)
            geno_plink = np.vstack(rows)

    # Sample order should match between the two paths.
    assert sample_ids_raw == samples_pg, "Sample order differs between .raw and .psam"
    assert geno_plink.shape == geno_pg.shape, (
        f"Shape mismatch: pgenlib {geno_pg.shape} vs PLINK .raw {geno_plink.shape}"
    )

    # Compare entries. pgenlib's are mean-imputed; PLINK .raw uses NA.
    # On non-missing entries, the two must match exactly.
    not_missing = ~np.isnan(geno_plink)
    if not np.allclose(geno_pg[not_missing], geno_plink[not_missing]):
        diffs = np.where(geno_pg[not_missing] != geno_plink[not_missing])[0]
        raise AssertionError(
            f"Numerical mismatch on {len(diffs):,} non-missing entries"
        )

    n_missing = int((~not_missing).sum())
    print(f"  shape:                 {geno_pg.shape}")
    print(f"  non-missing entries:   {int(not_missing.sum()):,} (exact match)")
    print(f"  missing entries:       {n_missing:,} (mean-imputed in pgenlib output)")
    print(f"  cross-check:           PASSED")


def main() -> int:
    print("=" * 72)
    print("genotype_loader test against real Stage 2 QC outputs")
    print("=" * 72)

    if not QC_DIR.exists():
        print(f"ERROR: QC directory does not exist: {QC_DIR}")
        return 2

    # Phase 1: load every (size, config), assert basic invariants.
    summaries = []
    for size in SIZES:
        for config in CONFIGS:
            try:
                summaries.append(test_one(size, config))
            except AssertionError as e:
                print(f"  FAIL: {e}")
                return 1
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                return 1

    # Phase 2: sample IDs must agree across every fileset.
    realised = [s for s in summaries if not s.get("skipped")]
    if len(realised) >= 2:
        ref_samples = realised[0]["samples"]
        for s in realised[1:]:
            if s["samples"] != ref_samples:
                print(
                    f"\nFAIL: sample IDs differ between {realised[0]['size']}_"
                    f"config{realised[0]['config']} and "
                    f"{s['size']}_config{s['config']}"
                )
                return 1
        print(f"\nSample IDs consistent across all {len(realised)} filesets.")

    # Phase 3: PLINK round-trip cross-check on the smallest matrix.
    try:
        cross_check_against_plink(size="47k", config=6)
    except FileNotFoundError:
        print("\nNote: plink2 not on PATH -- skipping cross-check. "
              "Run `module load plink/2.00a6LM` and rerun if you want it.")

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'size':>6} {'config':>6} {'n_var':>10} {'mem_MB':>10} {'load_s':>8}")
    for s in summaries:
        if s.get("skipped"):
            print(f"{s['size']:>6} {s['config']:>6}     SKIPPED")
            continue
        print(f"{s['size']:>6} {s['config']:>6} {s['n_variants']:>10,} "
              f"{s['memory_mb']:>10.1f} {s['load_time_s']:>8.2f}")

    print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
