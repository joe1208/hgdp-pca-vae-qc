#!/usr/bin/env bash
# =============================================================================
# Stage-2 QC: apply the per-config MAF filter; configs 4-6 additionally apply
# LD pruning (--indep-pairwise 50 5 0.2 --seed 42).
# Run order: step 03 of the main pipeline; run once per (size, config).
# Inputs:    qc/<size>_base/staged.{pgen,pvar,psam}
# Outputs:   qc/<size>_config<N>/qcd.{pgen,pvar,psam,log} (+ qcd.prune.* cfg4-6)
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
#
# 03_stage2_qc.sh - Stage 2 QC: apply config-specific MAF and LD pruning
# to a Stage 1 staged subset.
#
# Usage: bash scripts/03_stage2_qc.sh <size> <config>
#   <size>:   47k | 470k | full
#   <config>: 1 | 2 | 3 | 4 | 5 | 6
#
# Reads:  qc/<size>_base/staged.{pgen,pvar,psam}
# Writes: qc/<size>_config<N>/qcd.{pgen,pvar,psam,log}
#         (configs 4-6 also write qcd.prune.in / qcd.prune.out / qcd.prune.log)

set -euo pipefail

# --- args ---
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <size: 47k|470k|full> <config: 1-6>" >&2
  exit 1
fi

SIZE="$1"
CONFIG="$2"

case "$SIZE" in
  47k|470k|full) ;;
  *) echo "ERROR: size must be '47k', '470k', or 'full', got '$SIZE'" >&2; exit 1 ;;
esac

case "$CONFIG" in
  1|2|3|4|5|6) ;;
  *) echo "ERROR: config must be 1-6, got '$CONFIG'" >&2; exit 1 ;;
esac

# --- env check ---
command -v plink2 >/dev/null 2>&1 || {
  echo "ERROR: plink2 not on PATH. Activate the conda env first:" >&2
  echo "  e.g. 'module load plink/2.00a6LM' or activate an env that provides plink2" >&2
  exit 1
}

# --- paths ---
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
INPUT_PREFIX="${PROJECT_ROOT}/qc/${SIZE}_base/staged"
OUTPUT_DIR="${PROJECT_ROOT}/qc/${SIZE}_config${CONFIG}"
OUTPUT_PREFIX="${OUTPUT_DIR}/qcd"

mkdir -p "${OUTPUT_DIR}"

# --- config table ---
case "$CONFIG" in
  1) MAF_FLAG=""           ; LD="off" ;;
  2) MAF_FLAG="--maf 0.01" ; LD="off" ;;
  3) MAF_FLAG="--maf 0.05" ; LD="off" ;;
  4) MAF_FLAG=""           ; LD="on"  ;;
  5) MAF_FLAG="--maf 0.01" ; LD="on"  ;;
  6) MAF_FLAG="--maf 0.05" ; LD="on"  ;;
esac

echo "==> Stage 2 QC: size=${SIZE}, config=${CONFIG}"
echo "    MAF:        ${MAF_FLAG:-none}"
echo "    LD pruning: ${LD}"
echo "    Input:      ${INPUT_PREFIX}"
echo "    Output:     ${OUTPUT_PREFIX}"
echo

# --- input check ---
for ext in pgen pvar psam; do
  [[ -f "${INPUT_PREFIX}.${ext}" ]] || {
    echo "ERROR: missing input ${INPUT_PREFIX}.${ext}" >&2; exit 1;
  }
done

# --- run ---
if [[ "$LD" == "off" ]]; then
  plink2 \
    --pfile "${INPUT_PREFIX}" \
    ${MAF_FLAG} \
    --make-pgen \
    --out "${OUTPUT_PREFIX}"
else
  plink2 \
    --pfile "${INPUT_PREFIX}" \
    ${MAF_FLAG} \
    --indep-pairwise 50 5 0.2 \
    --seed 42 \
    --out "${OUTPUT_PREFIX}"

  mv "${OUTPUT_PREFIX}.log" "${OUTPUT_PREFIX}.prune.log"

  plink2 \
    --pfile "${INPUT_PREFIX}" \
    ${MAF_FLAG} \
    --extract "${OUTPUT_PREFIX}.prune.in" \
    --make-pgen \
    --out "${OUTPUT_PREFIX}"
fi

# --- summary ---
N_VARIANTS=$(grep -vc '^#' "${OUTPUT_PREFIX}.pvar")
N_SAMPLES=$(grep -vc '^#' "${OUTPUT_PREFIX}.psam")

echo
echo "==> DONE: ${N_VARIANTS} variants, ${N_SAMPLES} samples"
echo "    Files:  ${OUTPUT_PREFIX}.{pgen,pvar,psam,log}"
[[ "$LD" == "on" ]] && echo "    Pruning: ${OUTPUT_PREFIX}.{prune.in,prune.out,prune.log}"
