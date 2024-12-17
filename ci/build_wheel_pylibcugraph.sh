#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/pylibcugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the libcugraph and wheel built in the previous step and make it
# available for pip to find.
#
# ensure 'cugraph' wheel builds always use the 'libcugraph' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist)

# TODO(jameslamb): remove this stuff from https://github.com/rapidsai/raft/pull/2531
RAFT_COMMIT="f492d59978af3390e418796228aedb2601d03efc"
LIBRAFT_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}"
)

cat > ./constraints.txt <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUGRAPH_WHEELHOUSE}/libcugraph_*.whl)
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_CHANNEL}/libraft_*.whl)
EOF

export PIP_CONSTRAINT="${PWD}/constraints.txt"

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

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF${EXTRA_CMAKE_ARGS}"

./ci/build_wheel.sh pylibcugraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} final_dist
