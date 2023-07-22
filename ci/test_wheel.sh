#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

package_name=$1
package_dir=$2

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# echo to expand wildcard before adding `[extra]` requires for pip
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist
python -m pip install $(echo ./dist/${package_name}*.whl)[test]

# Run smoke tests for aarch64 pull requests
arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_${package_name}.py
else
    RAPIDS_DATASET_ROOT_DIR=`pwd`/datasets python -m pytest ./python/${package_name}/${package_name}/tests
fi
