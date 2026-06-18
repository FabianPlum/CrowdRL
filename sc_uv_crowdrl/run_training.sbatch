#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Usage (from the repo root):
#   sbatch sc_uv_crowdrl/run_training.sbatch configs/my_experiment.yaml
#   sbatch sc_uv_crowdrl/run_training.sbatch configs/my_experiment.yaml --init_from path/to/checkpoint.pt

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch sc_uv_crowdrl/run_training.sbatch <config.yaml> [extra train_mappo.py args...]" >&2
    exit 2
fi

CONFIG="$1"
shift

source "${SLURM_SUBMIT_DIR}/sc_uv_crowdrl/activate.sh"

cd "${CROWDRL_DIR}"
mkdir -p logs

# Redirect all caches to project space (home quota is too small for torch.compile)
export XDG_CACHE_HOME="${CROWDRL_DIR}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torch_inductor"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

srun --export=ALL --cpu_bind=none \
    python train_mappo.py \
        --config "$CONFIG" \
        "$@"
