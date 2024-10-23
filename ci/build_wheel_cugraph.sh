#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="pylibcugraph"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

# Download the pylibcugraph wheel built in the previous step and install it in the build environment.
#
# Ensures 'cugraph' wheel builds always use the 'pylibcugraph' just built in the same CI run.
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 /tmp/pylibcugraph_dist)
echo "$(echo ${CPP_WHEELHOUSE}/pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" >> /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

PARALLEL_LEVEL=$(python -c \
  "from math import ceil; from multiprocessing import cpu_count; print(ceil(cpu_count()/4))")

case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=ON"
  ;;
  11.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=OFF"
  ;;
esac

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF;-DCPM_cugraph-ops_SOURCE=${GITHUB_WORKSPACE}/cugraph-ops/${EXTRA_CMAKE_ARGS}"
export SKBUILD_BUILD_TOOL_ARGS="-j${PARALLEL_LEVEL};-l${PARALLEL_LEVEL}"

./ci/build_wheel.sh cugraph python/cugraph
