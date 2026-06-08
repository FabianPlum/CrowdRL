# sc_uv_crowdrl — HPC environment for CrowdRL

Python 3.12 + PyTorch 2.6 (cu126) environment managed by [uv](https://docs.astral.sh/uv/),
designed for JURECA/JUWELS.  Mirrors the `sc_venv_*` pattern used elsewhere in this project
but uses uv workspaces instead of a plain venv + pip.

## One-time setup

### 1. Create your site configuration

```bash
cd sc_uv_crowdrl
cp site.sh.template site.sh
# edit site.sh and set UV_ROOT to your project allocation path, e.g.:
#   export UV_ROOT="/p/project1/<your-project-id>"
```

`site.sh` is gitignored — each user keeps their own copy.

### 2. Run the setup script (login node only)

```bash
bash sc_uv_crowdrl/setup.sh
```

This will:
- Install `uv` to `$UV_ROOT/bin` if not already present
- Download CPython 3.12 via uv (to `$UV_ROOT/.uv/python`)
- Create `.venv/` in the CrowdRL workspace root
- Install all workspace packages and the `dev` dependency group
  (torch 2.6+cu126, triton, nvidia-\*-cu12 CUDA runtime wheels, jupyter, etc.)

All uv state (cache, data, Python installs) lives under `$UV_ROOT/.uv/` —
never in the home directory, which has a small quota on JURECA.

### 3. Set up Jupyter kernel (optional)

```bash
bash sc_uv_crowdrl/create_kernel.sh
```

### 4. Set up VSCode Python interpreter (optional)

```bash
bash sc_uv_crowdrl/create_python_for_vscode.sh
# Then point VSCode to: sc_uv_crowdrl/python
```

## Test the installation

Once setup is complete, smoke-test the environment on a compute node by submitting
the bundled Slurm job from the repo root:

```bash
sbatch sc_uv_crowdrl/run_tests.sbatch
```

This requests a single GPU on the `dc-gpu-devel` partition, activates the env, and
runs the test suite (`python -m pytest`). Job logs land in `<jobid>.out` and
`<jobid>.err` in the directory you submit from.

If your compute budget differs, edit the `#SBATCH --account=...` line in
`run_tests.sbatch` to your own project before submitting.

## Daily use

```bash
source sc_uv_crowdrl/activate.sh
```

This loads the HPC modules (Stages/2026, GCC, OpenMPI, IPython, git), activates the uv
venv, and adds the bundled CUDA runtime libraries to `LD_LIBRARY_PATH`.

## Design notes

| Concern | Decision |
|---|---|
| HPC stage | `Stages/2026` (Python 3.13 system) — but Python 3.12 is managed by uv itself |
| Python version | **3.12** — triton 3.1.0 (bundled with torch 2.6) has no cp313 wheel |
| CUDA | **cu126 wheels are self-contained** — no `module load CUDA` needed; PyTorch ships its own runtime via `nvidia-*-cu12` packages |
| uv state | All redirected to `$UV_ROOT/.uv/` via `UV_CACHE_DIR`, `UV_DATA_DIR`, `UV_PYTHON_INSTALL_DIR` |
| `--system-site-packages` | **Not used** — uv manages the full environment; no HPC-installed Python packages are inherited |
