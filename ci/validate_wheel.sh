#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

# some packages are much larger on CUDA 11 than on CUDA 12
if [[ "${package_dir}" == "python/cugraph" ]] || [[ "${package_dir}" == "python/pylibcugraph" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "11" ]]; then
        PYDISTCHECK_ARGS=(
            --max-allowed-size-compressed '1.5G'
        )
    else
        PYDISTCHECK_ARGS=(
            --max-allowed-size-compressed '975M'
        )
    fi
else
    PYDISTCHECK_ARGS=()
fi

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo ${wheel_dir_relative_path}/*.whl)"
