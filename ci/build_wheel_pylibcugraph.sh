#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/pylibcugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcugraph wheel built in the previous step and make it
# available for pip to find.
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist)

cat >> ./constraints.txt <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUGRAPH_WHEELHOUSE}"/libcugraph_*.whl)
EOF

# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
export PIP_CONSTRAINT="${PWD}/constraints.txt"

./ci/build_wheel.sh pylibcugraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
