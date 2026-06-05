#!/usr/bin/env bash
# =============================================================================
# Slurm wrapper for 05_train_vae.py: gpushort, 1 GPU, mem=32G, 55 min wall
# (standard-memory configs).
# Run order: launcher for step 05 (VAE branch).
# Inputs:    args SIZE CONFIG SEED -> runs scripts/05_train_vae.py
# Outputs:   one VAE seed-run (see 05_train_vae.py) + logs/vae_*.{out,err}
# Slurm submission wrapper, organised and commented for submission; logic unchanged.
# =============================================================================
# vae_slurm.sh - submit one VAE run on gpushort
#
# Usage:   sbatch scripts/vae_slurm.sh SIZE CONFIG SEED
# Example: sbatch scripts/vae_slurm.sh 47k 1 42
#
# SIZE in {47k, 470k}; CONFIG in {1..6}; SEED an integer (e.g. 42, 43, 44).
#
# Notes
# -----
# * gpushort is the only GPU partition open to the `pilot` account
#   (the longer `gpu` partition needs `pilot_gpu`). 1 hr wall-clock cap.
# * Cache redirects below are required: home quota is full and conda /
#   matplotlib will silently fail to write caches without these.
# * PLINK module load is harmless even though the loader uses pgenlib;
#   it's there in case the dosage cache is missing and the loader has to
#   regenerate from .pgen.

#SBATCH --job-name=vae
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=logs/vae_%x_%j.out
#SBATCH --error=logs/vae_%x_%j.err

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch scripts/vae_slurm.sh SIZE CONFIG SEED" >&2
    exit 2
fi

SIZE=$1
CONFIG=$2
SEED=$3

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PROJECT_ROOT/logs"

# Cache redirects (the author's home quota was full on Apocrita). These default
# under the project root and can be overridden per cluster.
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_ROOT/conda_cache/pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"

# PLINK lives in the system module, not the conda env
module load plink/2.00a6LM 2>/dev/null || true

# GPU Python interpreter; override with VAE_PYTHON for a different environment.
PYBIN="${VAE_PYTHON:-$PROJECT_ROOT/conda_env_gpu/bin/python}"

cd "$PROJECT_ROOT"
echo "[$(date -Is)] starting: size=$SIZE config=$CONFIG seed=$SEED"
"$PYBIN" scripts/05_train_vae.py "$SIZE" "$CONFIG" "$SEED"
echo "[$(date -Is)] done"