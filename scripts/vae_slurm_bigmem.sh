#!/usr/bin/env bash
# =============================================================================
# Slurm wrapper for 05_train_vae.py: gpushort, 1 GPU, mem=64G, 55 min wall
# (high-variant configs: 470k Config 1 and Config 4).
# Run order: launcher for step 05 (VAE branch; bigmem variant).
# Inputs:    args SIZE CONFIG SEED -> runs scripts/05_train_vae.py
# Outputs:   one VAE seed-run (see 05_train_vae.py) + logs/vae_*.{out,err}
# Slurm submission wrapper, organised and commented for submission; logic unchanged.
# =============================================================================
#SBATCH --partition=gpushort
#SBATCH --account=pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:55:00
#SBATCH --output=logs/vae_%x_%j.out
#SBATCH --error=logs/vae_%x_%j.err
#
# Submit with:
#   sbatch --job-name=vae_<size>_c<cfg>_s<seed> scripts/vae_slurm_bigmem.sh <size> <cfg> <seed>

set -euo pipefail

SIZE=$1
CONFIG=$2
SEED=$3

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$PROJECT_ROOT/logs"

module load miniforge/25.3.0
module load plink/2.00a6LM
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_ROOT/conda_cache/pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.cache/matplotlib}"
source activate "${VAE_CONDA_ENV:-$PROJECT_ROOT/conda_env_gpu}"

cd "$PROJECT_ROOT"
echo "=== Starting VAE: size=${SIZE} config=${CONFIG} seed=${SEED} ==="
echo "Node: $(hostname), GPU:"
nvidia-smi --query-gpu=name,memory.free --format=csv | head -2

python scripts/05_train_vae.py ${SIZE} ${CONFIG} ${SEED}

echo "=== Done ==="
