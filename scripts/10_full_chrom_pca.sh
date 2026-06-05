#!/usr/bin/env bash
# =============================================================================
# Feasibility-gated full-chromosome (4.7M) PCA-only sensitivity driver. Builds
# the full configs only if qc/full_base/staged exists (it never rebuilds from
# the raw VCF), runs PCA, then scores. VAE is out of scope at 4.7M, so this
# path is PCA-only (see README and Results 3.1).
# Run order: sensitivity path (step 10). Calls scripts/03_stage2_qc.sh,
#            scripts/04_run_pca.sh, and scripts/11_score_full_pca.py.
# Inputs:    qc/full_base/staged.* (intended), qc/full_config*/qcd.*
# Outputs:   qc/full_config*/qcd.*, pca/full_config*/pcs.*, results/metrics_full_pca.tsv
# Analysis script, organised and commented for submission; logic unchanged.
# =============================================================================
# Attempt the PCA-only full-chromosome sensitivity analysis.
# Runs only if qc/full_config* already exists or qc/full_base/staged
# exists, so it does not silently rebuild the full project from raw VCF.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "${PROJECT_ROOT}"

echo "==> Full-chromosome PCA feasibility check"
echo "    Project: ${PROJECT_ROOT}"
df -h "${PROJECT_ROOT}" || true

module load plink/2.00a6LM 2>/dev/null || true
command -v plink2 >/dev/null 2>&1 || {
    echo "ERROR: plink2 not found after module load." >&2
    exit 1
}

needs_full_build=false
for cfg in 1 2 3 4 5 6; do
    for ext in pgen pvar psam; do
        if [ ! -f "qc/full_config${cfg}/qcd.${ext}" ]; then
            needs_full_build=true
        fi
    done
done

for cfg in 4 5 6; do
    log="qc/full_config${cfg}/qcd.prune.log"
    if [ ! -f "${log}" ]; then
        echo "    full config ${cfg} is missing qcd.prune.log"
        needs_full_build=true
    elif ! grep -Eq -- '(^|[[:space:]])--seed 42($|[[:space:]])|Random number seed: 42' "${log}"; then
        echo "    full config ${cfg} exists but does not record seed 42"
        needs_full_build=true
    fi
done

if [ "${needs_full_build}" = true ]; then
    have_full_base=true
    for ext in pgen pvar psam; do
        if [ ! -f "qc/full_base/staged.${ext}" ]; then
            have_full_base=false
        fi
    done

    if [ "${have_full_base}" = false ]; then
        echo "NOT FEASIBLE: no qc/full_config*/qcd files and no qc/full_base/staged files found."
        echo "Remove the full-chromosome PCA claim unless those files can be recovered quickly."
        exit 2
    fi

    echo
    echo "==> Found qc/full_base/staged. Building full configs 1-6."
    du -sh qc/full_base || true
    for cfg in 1 2 3 4 5 6; do
        cfg_needs_build=false
        for ext in pgen pvar psam; do
            if [ ! -f "qc/full_config${cfg}/qcd.${ext}" ]; then
                cfg_needs_build=true
            fi
        done
        if [ "${cfg}" = "4" ] || [ "${cfg}" = "5" ] || [ "${cfg}" = "6" ]; then
            log="qc/full_config${cfg}/qcd.prune.log"
            if [ ! -f "${log}" ]; then
                cfg_needs_build=true
            elif ! grep -Eq -- '(^|[[:space:]])--seed 42($|[[:space:]])|Random number seed: 42' "${log}"; then
                cfg_needs_build=true
            fi
        fi

        if [ "${cfg_needs_build}" = false ]; then
            echo "    skip existing qc/full_config${cfg}"
        else
            if [ -e "qc/full_config${cfg}" ]; then
                qdir="repair_quarantine/full_pca_$(date +%F_%H%M%S)"
                mkdir -p "${qdir}/qc"
                echo "    quarantine existing qc/full_config${cfg} to ${qdir}"
                mv "qc/full_config${cfg}" "${qdir}/qc/"
            fi
            bash scripts/03_stage2_qc.sh full "${cfg}"
        fi
    done
else
    echo "==> Found existing seeded qc/full_config1-6 inputs."
fi

echo
echo "==> Running missing full PCA outputs"
for cfg in 1 2 3 4 5 6; do
    if [ -f "pca/full_config${cfg}/pcs.eigenvec" ] &&
       [ -f "pca/full_config${cfg}/pcs.eigenval" ]; then
        echo "    skip existing pca/full_config${cfg}"
    else
        bash scripts/04_run_pca.sh full "${cfg}"
    fi
done

echo
echo "==> Scoring full PCA sensitivity outputs"
python scripts/11_score_full_pca.py

echo
echo "==> Full-chromosome PCA attempt complete"
echo "    Metrics: results/metrics_full_pca.tsv"
