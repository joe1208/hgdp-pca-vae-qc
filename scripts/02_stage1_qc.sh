#!/usr/bin/env bash
# =============================================================================
# Stage-1 QC: restrict to chr5 biallelic ACGT SNPs, rebuild variant IDs as
# chr:pos:ref:alt, extract the subset, then drop variants with >2% missingness
# (--geno 0.02) and HWE p<1e-6 (--hwe 1e-6). No per-sample --mind filter is
# applied (the samples were already QC'd upstream; see README).
# Run order: step 02 of the main pipeline; run once per dataset size.
# Inputs:    data/raw/...full.chr5.vcf.gz, data/subsets/variants_<size>.snplist
# Outputs:   qc/<size>_base/staged.{pgen,pvar,psam,log}
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
# Usage: bash scripts/02_stage1_qc.sh {47k|470k}
# Produces a staged .pgen for the config-specific Stage-2 QC.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 {47k|470k}" >&2
    exit 1
fi

DATASET_SIZE=$1
case $DATASET_SIZE in
    47k|470k) ;;
    *) echo "Invalid size: must be 47k or 470k" >&2; exit 1 ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
VCF=$(find_vcf) || exit 1
SUBSET=${PROJECT_ROOT}/data/subsets/variants_${DATASET_SIZE}.snplist
OUT_DIR=${PROJECT_ROOT}/qc/${DATASET_SIZE}_base
mkdir -p ${OUT_DIR}

echo "=== Stage 1: Subset + baseline QC ==="
echo "Size:   ${DATASET_SIZE}"
echo "VCF:    ${VCF}"
echo "Subset: ${SUBSET}"
echo "Output: ${OUT_DIR}/staged"
echo "======================================"

plink2 \
    --vcf ${VCF} \
    --chr 5 \
    --max-alleles 2 --snps-only just-acgt \
    --set-all-var-ids '@:#:$r:$a' --new-id-max-allele-len 50 --rm-dup exclude-all \
    --extract ${SUBSET} \
    --geno 0.02 \
    --hwe 1e-6 \
    --make-pgen \
    --out ${OUT_DIR}/staged

echo "=== Done ==="
