#!/bin/bash

# See https://stackoverflow.com/a/28336473
SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

[[ "$0" != "${SOURCE_PATH}" ]] && echo "The activation script must be sourced, otherwise the virtual environment will not work." || ( echo "Vars script must be sourced." && exit 1) ;

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/site.sh
source "${ABSOLUTE_PATH}"/modules.sh

# Ensure uv is in PATH and all uv state is kept in project space (home quota is limited on JURECA)
export PATH="${UV_ROOT}/bin:$PATH"
export UV_CACHE_DIR="${UV_ROOT}/.uv/cache"
export UV_DATA_DIR="${UV_ROOT}/.uv/data"
export UV_PYTHON_INSTALL_DIR="${UV_ROOT}/.uv/python"

export PYTHONPATH="$(echo "${ENV_DIR}"/lib/python*/site-packages):${PYTHONPATH}"

# Add bundled CUDA runtime libs (nvidia-*-cu12 wheels) to the dynamic linker path
for _nvidia_lib_dir in "${ENV_DIR}"/lib/python*/site-packages/nvidia/*/lib; do
    export LD_LIBRARY_PATH="${_nvidia_lib_dir}:${LD_LIBRARY_PATH}"
done
unset _nvidia_lib_dir

source "${ENV_DIR}"/bin/activate
