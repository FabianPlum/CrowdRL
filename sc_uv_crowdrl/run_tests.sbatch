#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu-devel
#SBATCH --time=00:10:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# Submit from the repo root, e.g.:  sbatch sc_uv_crowdrl/run_tests.sbatch
# activate.sh sources the gitignored site.sh (UV_ROOT) and config.sh (CROWDRL_DIR),
# so the project path lives only in site.sh and is never committed here.
source "${SLURM_SUBMIT_DIR}/sc_uv_crowdrl/activate.sh"

cd "${CROWDRL_DIR}"

srun --export=ALL --cpu_bind=none python -m pytest
