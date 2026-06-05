# Shared path configuration for the shell steps. This file is sourced, not run.
#
# PROJECT_ROOT resolution order:
#   1. The QC_VAE_PCA_ROOT environment variable, if set. On Apocrita the project
#      lived at /gpfs/scratch/bt24080/qc_vae_pca, so the original runs are
#      reproduced by `export QC_VAE_PCA_ROOT=/gpfs/scratch/bt24080/qc_vae_pca`.
#   2. Otherwise, the code_appendix directory that contains this scripts/ folder.
#
# Raw inputs live in $PROJECT_ROOT/data/raw; find_vcf locates the HGDP VCF there
# by pattern, so the only setup step is to drop the VCF and metadata into it.

if [ -n "${QC_VAE_PCA_ROOT:-}" ]; then
  PROJECT_ROOT="${QC_VAE_PCA_ROOT}"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PROJECT_ROOT
RAW_DIR="${PROJECT_ROOT}/data/raw"
export RAW_DIR

# Echo the single HGDP VCF in data/raw, or fail with a clear message.
find_vcf() {
  mkdir -p "${RAW_DIR}"
  local matches
  matches=$(ls -1 "${RAW_DIR}"/*.vcf.gz 2>/dev/null)
  local n
  n=$(printf '%s\n' "${matches}" | grep -c . )
  if [ "${n}" -eq 0 ]; then
    echo "ERROR: no .vcf.gz found in ${RAW_DIR}." >&2
    echo "       Place the HGDP chr5 VCF there, or set QC_VAE_PCA_ROOT." >&2
    return 1
  fi
  if [ "${n}" -gt 1 ]; then
    echo "ERROR: multiple .vcf.gz files in ${RAW_DIR}; leave only the intended one." >&2
    return 1
  fi
  printf '%s\n' "${matches}"
}
