#!/usr/bin/env bash
# =============================================================================
# Thin convenience driver for the DETERMINISTIC, CPU-only part of the pipeline:
# Stage-1 QC -> Stage-2 QC -> PCA, for a single (size, config).
#
# This driver DOES NOT cover the VAE branch (GPU; submitted via Slurm with
# scripts/vae_slurm.sh or scripts/vae_slurm_bigmem.sh) or the scoring, stats,
# and figure steps. Those are run separately; see RUNNING.md for the full run
# order and the "what runs where" table.
#
# Prerequisite (run once, not performed here): scripts/01_make_subsets.py, plus
# the raw VCF / subset lists it and Stage-1 depend on.
#
# Paths are resolved by scripts/env.sh: the project root defaults to this
# code_appendix directory, or is taken from $QC_VAE_PCA_ROOT if set (the HPC
# used /gpfs/scratch/bt24080/qc_vae_pca). Raw inputs are read from data/raw.
#
# Usage: bash run_qc_pca.sh <size: 47k|470k> <config: 1-6>
# =============================================================================
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: bash run_qc_pca.sh <size: 47k|470k> <config: 1-6>" >&2
    exit 2
fi

SIZE="$1"
CONFIG="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"

echo "[$(date -Is)] Stage-1 QC (size=${SIZE})"
bash "${SCRIPT_DIR}/02_stage1_qc.sh" "${SIZE}"

echo "[$(date -Is)] Stage-2 QC (size=${SIZE}, config=${CONFIG})"
bash "${SCRIPT_DIR}/03_stage2_qc.sh" "${SIZE}" "${CONFIG}"

echo "[$(date -Is)] PCA (size=${SIZE}, config=${CONFIG})"
bash "${SCRIPT_DIR}/04_run_pca.sh" "${SIZE}" "${CONFIG}"

echo "[$(date -Is)] done: Stage-1 + Stage-2 + PCA for ${SIZE} config ${CONFIG}"
echo "Next, separately: VAE via scripts/vae_slurm.sh (Slurm/GPU), then scoring"
echo "(scripts/06_score_metrics.py). See RUNNING.md."
