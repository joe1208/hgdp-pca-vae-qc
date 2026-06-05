#!/usr/bin/env bash
# =============================================================================
# PCA via PLINK2 --pca 10 (deterministic; one run per config). This is the PCA
# branch (04); the VAE branch is 05_train_vae.py and runs from the same Stage-2
# QC inputs.
# Run order: step 04 of the main pipeline; run once per (size, config).
# Inputs:    qc/<size>_config<N>/qcd.*
# Outputs:   pca/<size>_config<N>/pcs.{eigenvec,eigenval,log}
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
# Run the reproducible PLINK PCA command for one QC output.
# This records the previously manual PCA step as a script.
# Inputs must already exist in qc/<size>_config<config>/qcd.

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <size: 47k|470k|full> <config: 1-6>" >&2
    exit 2
fi

SIZE="$1"
CONFIG="$2"

case "$SIZE" in
    47k|470k|full) ;;
    *) echo "ERROR: size must be 47k, 470k, or full, got '$SIZE'" >&2; exit 2 ;;
esac

case "$CONFIG" in
    1|2|3|4|5|6) ;;
    *) echo "ERROR: config must be 1-6, got '$CONFIG'" >&2; exit 2 ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
INPUT_PREFIX="${PROJECT_ROOT}/qc/${SIZE}_config${CONFIG}/qcd"
OUT_DIR="${PROJECT_ROOT}/pca/${SIZE}_config${CONFIG}"
OUT_PREFIX="${OUT_DIR}/pcs"

command -v plink2 >/dev/null 2>&1 || {
    echo "ERROR: plink2 not on PATH. Load plink/2.00a6LM or activate the CPU env." >&2
    exit 1
}

for ext in pgen pvar psam; do
    if [ ! -f "${INPUT_PREFIX}.${ext}" ]; then
        echo "ERROR: missing input ${INPUT_PREFIX}.${ext}" >&2
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

echo "==> PCA: size=${SIZE}, config=${CONFIG}"
echo "    Input:  ${INPUT_PREFIX}"
echo "    Output: ${OUT_PREFIX}.{eigenvec,eigenval,log}"

plink2 \
    --pfile "${INPUT_PREFIX}" \
    --pca 10 \
    --out "${OUT_PREFIX}"

echo "==> DONE: ${OUT_PREFIX}.{eigenvec,eigenval,log}"
