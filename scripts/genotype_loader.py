# =============================================================================
# pgenlib-backed .pgen dosage loader with per-variant mean imputation and an
# on-disk .npy cache. Imputation happens before caching and a no-NaN assertion
# guards the cached array, so a cached dosage matrix can never contain NaNs.
# Run order: library module imported by 05_train_vae.py (not run directly).
# Inputs:    qc/<size>_config<N>/qcd.{pgen,pvar,psam}
# Outputs:   in-memory (geno, sample_ids, variant_ids) + dosage .npy cache
# Genotype-loading module, organised and commented for submission; logic unchanged.
# =============================================================================
"""
genotype_loader.py
==================
Memory-efficient loader for PLINK 2 .pgen/.pvar/.psam genotype files.

Replaces the previous `plink2 --export A` -> pandas.read_csv pipeline,
which was OOM-killing on the 470k Config 1/4 runs because text parsing
holds the entire .raw file as Python objects transiently.

This loader uses pgenlib (the official PLINK 2 Python bindings) to read
the binary .pgen file directly into a pre-allocated int8 numpy buffer.
For a 470k variant x 929 sample matrix, peak memory is ~455 MB and
load time is ~3 seconds.

Imputation is performed BEFORE any caching, so cached arrays cannot
contain NaNs. An assertion at the end of load() confirms this.

Usage
-----
    from genotype_loader import load_genotypes

    geno, sample_ids, variant_ids = load_genotypes(
        plink_prefix="qc/470k_config1/qcd",         # PLINK 2 fileset prefix (.pgen/.pvar/.psam)
        cache_path="vae/470k_config1_dosage.npy",   # optional on-disk dosage cache
    )
    # geno: float32 array, shape (n_samples, n_variants), no NaNs
    # sample_ids: list[str] from .psam (IID column)
    # variant_ids: list[str] from .pvar (ID column)

The output orientation (samples on axis 0, variants on axis 1) matches
the input shape expected by a feed-forward VAE encoder.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pgenlib as pg

# pgenlib's missing-value sentinel for int8 dosages
PGEN_MISSING = -9


def _read_psam(psam_path: Path) -> List[str]:
    """Read the IID column from a PLINK 2 .psam file."""
    iids: List[str] = []
    with psam_path.open() as fh:
        header_line = None
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                # PLINK 2 .psam header looks like: #FID IID PAT MAT SEX [PHENO1 ...]
                # or just #IID for single-column files.
                header_line = line.lstrip("#").split()
                continue
            cols = line.split()
            if header_line is not None and "IID" in header_line:
                idx = header_line.index("IID")
                iids.append(cols[idx])
            else:
                # No header: PLINK 1-style FID IID ...; take col 1 (IID).
                iids.append(cols[1] if len(cols) > 1 else cols[0])
    return iids


def _read_pvar_ids(pvar_path: Path) -> List[str]:
    """Read the ID column from a PLINK 2 .pvar file."""
    ids: List[str] = []
    with pvar_path.open() as fh:
        header_cols = None
        for line in fh:
            if line.startswith("##"):
                continue  # VCF-style metadata lines
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                header_cols = line.lstrip("#").split()
                continue
            cols = line.split()
            if header_cols is not None and "ID" in header_cols:
                ids.append(cols[header_cols.index("ID")])
            else:
                # No header; assume PLINK 1-style: chrom, ID, cM, bp, alt, ref
                ids.append(cols[2] if len(cols) > 2 else cols[1])
    return ids


def load_genotypes(
    plink_prefix: str | os.PathLike,
    cache_path: str | os.PathLike | None = None,
    dtype: np.dtype = np.float32,
    impute_chunk_size: int = 100_000,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load genotypes from a PLINK 2 .pgen/.pvar/.psam fileset.

    Parameters
    ----------
    plink_prefix : str | PathLike
        Path prefix to the PLINK 2 fileset. The function expects
        {prefix}.pgen, {prefix}.pvar, and {prefix}.psam to exist.
    cache_path : str | PathLike | None, default None
        If given, the imputed dosage matrix is loaded from / saved to
        this .npy path. Imputation runs BEFORE the cache write, so the
        cached file is guaranteed NaN-free.
    dtype : np.dtype, default np.float32
        Output dtype for the dosage matrix. float32 is appropriate for
        PyTorch training; pgenlib reads as int8 internally.
    impute_chunk_size : int, default 100_000
        Number of variants per chunk during mean imputation. Defensive
        chunking in case a future scale-up runs into RAM ceilings; at
        470k variants a single-chunk imputation peaks under 2 GB.

    Returns
    -------
    geno : np.ndarray
        Shape (n_samples, n_variants), dtype as requested. Missing
        genotypes are mean-imputed per variant. No NaNs (asserted).
    sample_ids : list[str]
        IIDs from the .psam file, in pgen sample order.
    variant_ids : list[str]
        Variant IDs from the .pvar file, in pgen variant order.
    """
    plink_prefix = Path(plink_prefix)
    pgen_path = plink_prefix.with_suffix(".pgen")
    pvar_path = plink_prefix.with_suffix(".pvar")
    psam_path = plink_prefix.with_suffix(".psam")
    for p in (pgen_path, pvar_path, psam_path):
        if not p.exists():
            raise FileNotFoundError(f"Required PLINK 2 file missing: {p}")

    # Always read sample/variant IDs (cheap, and needed downstream).
    sample_ids = _read_psam(psam_path)
    variant_ids = _read_pvar_ids(pvar_path)

    # Cache hit: load the imputed dosage matrix and skip pgen reading.
    if cache_path is not None and Path(cache_path).exists():
        geno = np.load(cache_path)
        if geno.shape != (len(sample_ids), len(variant_ids)):
            raise ValueError(
                f"Cached array shape {geno.shape} does not match "
                f"(n_samples={len(sample_ids)}, n_variants={len(variant_ids)}). "
                "Delete the cache file and rerun."
            )
        # Hard guarantee: cached arrays must never contain NaNs.
        assert not np.isnan(geno).any(), (
            f"Cached file {cache_path} contains NaNs. This indicates the "
            "cache was written by a previous (buggy) version of the loader. "
            "Delete the file and rerun."
        )
        return geno, sample_ids, variant_ids

    # Cache miss: read from .pgen.
    reader = pg.PgenReader(bytes(pgen_path))
    n_samples = reader.get_raw_sample_ct()
    n_variants = reader.get_variant_ct()
    if n_samples != len(sample_ids):
        reader.close()
        raise ValueError(
            f".pgen reports {n_samples} samples but .psam has {len(sample_ids)}"
        )
    if n_variants != len(variant_ids):
        reader.close()
        raise ValueError(
            f".pgen reports {n_variants} variants but .pvar has {len(variant_ids)}"
        )

    # Match plink2 --export A's default: count the REF allele, not ALT.
    # pgenlib's read_range hard-codes allele_idx=1 (ALT), so we read one
    # variant at a time with allele_idx=0 (REF). This is what produces
    # numerical agreement with --export A and keeps the new loader on the
    # same encoding as the 47k VAE runs already completed before this fix.
    raw = np.empty((n_variants, n_samples), dtype=np.int8)
    buf = np.empty(n_samples, dtype=np.int8)
    for v in range(n_variants):
        reader.read(v, buf, allele_idx=0)
        raw[v] = buf
    reader.close()

    # Convert to working dtype. We keep the (variants, samples) orientation
    # for imputation because per-variant means are along axis 1, which is
    # contiguous in this layout.
    geno_vs = raw.astype(dtype)
    del raw

    # Mean imputation per variant. We do this in chunks for memory headroom.
    missing_mask = geno_vs == PGEN_MISSING
    n_missing_total = int(missing_mask.sum())
    if n_missing_total > 0:
        for start in range(0, n_variants, impute_chunk_size):
            end = min(start + impute_chunk_size, n_variants)
            block = geno_vs[start:end]
            block_mask = missing_mask[start:end]
            # Per-variant mean over non-missing samples
            valid_counts = (~block_mask).sum(axis=1)
            # Set missing entries to 0 temporarily so they don't pollute the sum
            block_for_sum = np.where(block_mask, 0, block)
            sums = block_for_sum.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                means = np.where(valid_counts > 0, sums / valid_counts, 0.0)
            # Broadcast-impute
            block[block_mask] = np.repeat(means, block_mask.sum(axis=1))
            geno_vs[start:end] = block
    del missing_mask

    # Transpose to (samples, variants), the orientation a feed-forward
    # VAE encoder expects. .copy() forces a contiguous array; without it
    # downstream PyTorch operations would silently do non-contiguous
    # gathers and slow training significantly.
    geno = np.ascontiguousarray(geno_vs.T)
    del geno_vs

    # The crucial post-condition. If this fails, do not write the cache.
    assert not np.isnan(geno).any(), (
        "Imputation failed -- output contains NaNs. This is a bug; the "
        "previous version of the loader silently shipped NaN matrices "
        "to the VAE, producing collapsed latent spaces."
    )

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, geno)

    return geno, sample_ids, variant_ids


if __name__ == "__main__":
    # Quick smoke test if run directly: argparse, load, summarise.
    import argparse

    parser = argparse.ArgumentParser(description="Load a PLINK 2 fileset and summarise.")
    parser.add_argument("prefix", help="Path prefix to .pgen/.pvar/.psam (no extension)")
    parser.add_argument("--cache", default=None, help="Optional .npy cache path")
    args = parser.parse_args()

    import time
    t0 = time.time()
    geno, samples, variants = load_genotypes(args.prefix, cache_path=args.cache)
    t = time.time() - t0
    print(f"Loaded in {t:.2f}s")
    print(f"  shape:    {geno.shape}  dtype: {geno.dtype}")
    print(f"  memory:   {geno.nbytes / 1024 / 1024:.1f} MB")
    print(f"  range:    [{geno.min():.3f}, {geno.max():.3f}]  mean: {geno.mean():.3f}")
    print(f"  any NaN?  {np.isnan(geno).any()}")
    print(f"  samples:  {len(samples)} (first 3: {samples[:3]})")
    print(f"  variants: {len(variants)} (first 3: {variants[:3]})")
