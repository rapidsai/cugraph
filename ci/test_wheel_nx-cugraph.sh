#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

source ./ci/use_wheels_from_prs.sh

# Download wheels built during this job.
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
python -m pip install ./local-deps/*.whl

./ci/test_wheel.sh nx-cugraph python/nx-cugraph
