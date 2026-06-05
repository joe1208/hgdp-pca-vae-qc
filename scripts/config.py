"""Shared path configuration for the analysis code.

This resolves the project root and the raw input files without hardcoding any
absolute path, so the code runs both on the original HPC and from a local copy.

Project root resolution order:
  1. The QC_VAE_PCA_ROOT environment variable, if set. On Apocrita the project
     lived at /gpfs/scratch/bt24080/qc_vae_pca, so the original runs are
     reproduced exactly by `export QC_VAE_PCA_ROOT=/gpfs/scratch/bt24080/qc_vae_pca`.
  2. Otherwise, the code_appendix directory that contains this scripts/ folder.
     This lets the pipeline run from wherever the appendix is unpacked.

Raw inputs (the HGDP VCF and the sample metadata) are discovered by pattern in
<root>/data/raw, so the only setup step is to drop those two files into that
folder. No analysis logic, threshold, hyperparameter, or output filename
depends on this module; it only locates files.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "QC_VAE_PCA_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _find_one(pattern, description):
    """Return the single file in data/raw matching pattern, with clear errors."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches = sorted(RAW_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            "No %s found in %s (looked for '%s').\n"
            "Place the HGDP %s there, or set QC_VAE_PCA_ROOT to the project "
            "root that contains data/raw." % (description, RAW_DIR, pattern, description))
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise FileNotFoundError(
            "Expected exactly one %s in %s but found several: %s.\n"
            "Leave only the intended file in data/raw." % (description, RAW_DIR, names))
    return matches[0]


def find_metadata():
    """Locate the HGDP sample metadata (a *metadata*.txt file in data/raw)."""
    return _find_one("*metadata*.txt", "metadata file")


def find_vcf():
    """Locate the HGDP VCF (a *.vcf.gz file in data/raw)."""
    return _find_one("*.vcf.gz", "VCF (.vcf.gz)")
