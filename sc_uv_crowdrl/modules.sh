module purge
module load Stages/2026
module load GCC OpenMPI
module load IPython git
# No CUDA module — PyTorch cu126 bundles its own CUDA 12 runtime (nvidia-*-cu12 wheels)
