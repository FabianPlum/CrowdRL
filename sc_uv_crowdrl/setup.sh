#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/site.sh
source "${ABSOLUTE_PATH}"/modules.sh

# Install uv to project space (home quota is limited on JURECA)
mkdir -p "${UV_ROOT}/bin"
export PATH="${UV_ROOT}/bin:$PATH"

# Redirect all uv state away from home (cache, data, config, Python installs)
export UV_CACHE_DIR="${UV_ROOT}/.uv/cache"
export UV_DATA_DIR="${UV_ROOT}/.uv/data"
export UV_PYTHON_INSTALL_DIR="${UV_ROOT}/.uv/python"

if ! command -v uv &> /dev/null; then
    echo "uv not found — installing to ${UV_ROOT}/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="${UV_ROOT}/bin" sh
fi

echo "uv:     $(which uv)  ($(uv --version))"
echo "Python: $(python3 --version)"

# Install all workspace packages + dev dependency group (includes torch, triton, nvidia-*-cu12)
cd "${CROWDRL_DIR}"
# Pin to Python 3.12: triton 3.1.0 (bundled with torch 2.6) has no cp313 wheel
uv sync --python 3.12 --all-packages --group dev
