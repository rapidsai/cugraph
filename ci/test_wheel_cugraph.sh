#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

# Download the packages built in the previous step
mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-pylibcugraph-dep
RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libcugraph-dep

# TODO(jameslamb): remove this stuff from https://github.com/rapidsai/raft/pull/2531
RAFT_COMMIT="f492d59978af3390e418796228aedb2601d03efc"
LIBRAFT_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}"
)

cat > ./constraints.txt <<EOF
libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_CHANNEL}/libraft_*.whl)
EOF

export PIP_CONSTRAINT="${PWD}/constraints.txt"

python -m pip install \
    "$(echo ./dist/cugraph*.whl)[test]" \
    ./local-pylibcugraph-dep/pylibcugraph*.whl \
    ./local-libcugraph-dep/libcugraph*.whl

./ci/test_wheel.sh cugraph
