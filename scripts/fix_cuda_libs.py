#!/usr/bin/env python3
"""Install a .pth file that preloads CUDA libs at Python startup.

The pip-installed nvidia-* packages place shared libraries under
site-packages/nvidia/*/lib/, which is outside the default linker search
path.  PyTorch's `torch._C` extension fails to load without them.

This script writes a one-line .pth file into the active site-packages
directory so that libcusparseLt is loaded via ctypes before any import
of torch.  Re-run after `uv sync --reinstall` or venv recreation.

Usage:
    uv run python scripts/fix_cuda_libs.py
"""

import site
import sys
from pathlib import Path

PTH_NAME = "_nvidia_cuda_preload.pth"
PTH_CONTENT = (
    'import ctypes, importlib.util, os;'
    ' spec = importlib.util.find_spec("nvidia.cusparselt");'
    ' ctypes.cdll.LoadLibrary('
    'os.path.join(spec.submodule_search_locations[0], "lib", "libcusparseLt.so.0")'
    ') if spec else None\n'
)


def main() -> None:
    if sys.platform != "linux":
        print("Skipping — CUDA preload only needed on Linux")
        return

    site_dir = Path(site.getsitepackages()[0])
    pth_path = site_dir / PTH_NAME
    pth_path.write_text(PTH_CONTENT)
    print(f"Wrote {pth_path}")

    # Quick sanity check
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia.cusparselt")
        if spec is None:
            print("Warning: nvidia-cusparselt not installed — run 'uv sync --all-packages'")
        else:
            print("nvidia.cusparselt found — preload will work on next Python start")
    except Exception as e:
        print(f"Warning: {e}")


if __name__ == "__main__":
    main()
