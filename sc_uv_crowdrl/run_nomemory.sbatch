#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --time=1:00:00
#SBATCH --output=logs/nomemory_%j.out
#SBATCH --error=logs/nomemory_%j.err

# Submit from the repo root:
#   mkdir -p logs && sbatch sc_uv_crowdrl/run_nomemory.sbatch

source "${SLURM_SUBMIT_DIR}/sc_uv_crowdrl/activate.sh"

cd "${CROWDRL_DIR}"
mkdir -p logs

# Redirect all caches to project space (home quota is too small for torch.compile)
export XDG_CACHE_HOME="${CROWDRL_DIR}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torch_inductor"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export MPLCONFIGDIR="${XDG_CACHE_HOME}/matplotlib"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${MPLCONFIGDIR}"

CONFIG="configs/exp_nogoaldir_stable_bigrooms_nomemory.yaml"
INIT_FROM="results_exp_nogoaldir_stable_bigrooms/checkpoint_rollout_0150.pt"

srun --export=ALL --cpu_bind=none \
    python train_mappo.py \
        --config "$CONFIG" \
        --init_from "$INIT_FROM"
