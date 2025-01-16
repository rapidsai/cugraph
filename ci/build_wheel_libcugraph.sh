#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="libcugraph"
package_dir="python/libcugraph"

rapids-logger "Generating build requirements"
matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

# TODO(jameslamb): remove this when https://github.com/rapidsai/raft/pull/2531 is merged
source ./ci/use_wheels_from_prs.sh

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

rapids-logger "Done build requirements"

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

PARALLEL_LEVEL=$(python -c \
  "from math import ceil; from multiprocessing import cpu_count; print(ceil(cpu_count()/4))")

export SKBUILD_BUILD_TOOL_ARGS="-j${PARALLEL_LEVEL};-l${PARALLEL_LEVEL}"

./ci/build_wheel.sh libcugraph ${package_dir} cpp
./ci/validate_wheel.sh ${package_dir} final_dist
