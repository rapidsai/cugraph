#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

# Download the pylibcugraph built in the previous step
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibcugraph-dep
python -m pip install --no-deps ./local-pylibcugraph-dep/pylibcugraph*.whl

# Always install latest dask for testing
python -m pip install git+https://github.com/dask/dask.git@main git+https://github.com/dask/distributed.git@main git+https://github.com/rapidsai/dask-cuda.git@branch-23.10

# Only download test data for x86
arch=$(uname -m)
if [[ "${arch}" == "x86_64" ]]; then
    pushd ./datasets
    bash ./get_test_data.sh
    popd
fi

./ci/test_wheel.sh cugraph python/cugraph
