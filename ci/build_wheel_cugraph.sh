#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the libcugraph and pylibcugraph wheels built in the previous step and make them
# available for pip to find.
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist)
PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python /tmp/pylibcugraph_dist)

LIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 cpp wheel)
PYLIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 python wheel)
LIBCUDF_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact cudf 17899 cpp wheel)
PYLIBCUDF_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact cudf 17899 python wheel)
CUDF_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="cudf_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact cudf 17899 python wheel)
LIBRAFT_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact raft 2566 cpp wheel)
PYLIBRAFT_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact raft 2566 python wheel)

cat >> ./constraints.txt <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUGRAPH_WHEELHOUSE}/libcugraph_*.whl)
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUGRAPH_WHEELHOUSE}/pylibcugraph_*.whl)
EOF

echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEEL_DIR}/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRMM_WHEEL_DIR}/rmm_*.whl)" >> /tmp/constraints.txt
echo "libcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUDF_WHEEL_DIR}/libcudf_*.whl)" >> /tmp/constraints.txt
echo "cudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUDF_WHEEL_DIR}/cudf_*.whl)" >> /tmp/constraints.txt
echo "pylibcudf-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUDF_WHEEL_DIR}/pylibcudf_*.whl)" >> /tmp/constraints.txt
echo "libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_WHEEL_DIR}/libraft_*.whl)" >> /tmp/constraints.txt
echo "raft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRAFT_WHEEL_DIR}/raft_*.whl)" >> /tmp/constraints.txt

# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
export PIP_CONSTRAINT="${PWD}/constraints.txt"

./ci/build_wheel.sh cugraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} final_dist
