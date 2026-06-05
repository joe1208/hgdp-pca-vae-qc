#!/usr/bin/env python3
# =============================================================================
# Build nested random 47k/470k chr5 SNP subsets (seed 42). The 47k list is a
# strict prefix of the 470k list, so dataset-size effects are isolated from
# sampling variance when the two sizes are compared.
# Run order: step 01 of the main pipeline; run once per project.
# Inputs:    data/subsets/all_variants.snplist (full chr5 variant IDs)
# Outputs:   data/subsets/variants_47k.snplist, variants_470k.snplist
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
"""
Build nested random subsets of chr5 SNPs from the HGDP variant pool.

Reads the full variant ID list from PLINK's --write-snplist output,
shuffles it with a fixed seed for reproducibility, and writes two
nested subsets:
  - 470k subset: first 470,000 variants from the shuffled list
  - 47k subset:  first 47,000 variants from the shuffled list
                 (strict prefix of the 470k, so 47k is nested in 470k)

The nested design isolates dataset-size effects from sampling variance
when comparing 47k vs 470k results across QC configurations.
"""

import random

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
from config import PROJECT_ROOT
SEED = 42
SUBSET_DIR = PROJECT_ROOT / "data" / "subsets"
INPUT_FILE = SUBSET_DIR / "all_variants.snplist"
OUTPUT_470K = SUBSET_DIR / "variants_470k.snplist"
OUTPUT_47K  = SUBSET_DIR / "variants_47k.snplist"
SIZE_470K = 470_000
SIZE_47K  = 47_000

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Reading variant list from {INPUT_FILE}")
    with INPUT_FILE.open() as f:
        variants = [line.strip() for line in f if line.strip()]
    n_total = len(variants)
    print(f"  Loaded {n_total:,} variants")

    if n_total < SIZE_470K:
        raise RuntimeError(
            f"Only {n_total} variants available, cannot draw {SIZE_470K}"
        )

    print(f"Shuffling with seed={SEED}")
    rng = random.Random(SEED)
    rng.shuffle(variants)

    subset_470k = variants[:SIZE_470K]
    subset_47k  = subset_470k[:SIZE_47K]

    assert subset_47k == subset_470k[:SIZE_47K], "Nesting invariant broken"

    def sort_key(v: str) -> tuple[int, int]:
        chrom, pos, _ref, _alt = v.split(":")
        return (int(chrom), int(pos))

    for path, subset in [(OUTPUT_470K, subset_470k), (OUTPUT_47K, subset_47k)]:
        subset_sorted = sorted(subset, key=sort_key)
        with path.open("w") as f:
            f.write("\n".join(subset_sorted) + "\n")
        print(f"  Wrote {len(subset_sorted):,} variants to {path}")

    set_470k = set(subset_470k)
    set_47k  = set(subset_47k)
    assert set_47k.issubset(set_470k), "47k is not a subset of 470k on disk"
    print("\nNesting verified: 47k is a subset of 470k")
    print(f"Seed: {SEED}")
    print(f"Done.")


if __name__ == "__main__":
    main()
