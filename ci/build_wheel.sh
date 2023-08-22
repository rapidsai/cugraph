#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-{RAPIDS_PY_CUDA_SUFFIX}"

# Manually install dependencies because we're building without isolation.
rapids-dependency-file-generator \
  --output requirements \
  --file_key py_build_${package_name} \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee requirements.txt

# pyproject.toml updates
sed -i "s/^version = .*/version = \"${version_override}\"/g" \
  python/cugraph/pyproject.toml \
  python/cugraph-dgl/pyproject.toml \
  python/cugraph-pyg/pyproject.toml \
  python/cugraph-service/client/pyproject.toml \
  python/cugraph-service/server/pyproject.toml \
  python/pylibcugraph/pyproject.toml

# pylibcugraph pyproject.toml cuda suffixes
sed -i "s/name = \"pylibcugraph\"/name = \"pylibcugraph${PACKAGE_CUDA_SUFFIX}\"/g" python/pylibcugraph/pyproject.toml
sed -i "s/rmm/rmm${PACKAGE_CUDA_SUFFIX}/g" python/pylibcugraph/pyproject.toml
sed -i "s/pylibraft/pylibraft${PACKAGE_CUDA_SUFFIX}/g" python/pylibcugraph/pyproject.toml
sed -i "s/cudf/cudf${PACKAGE_CUDA_SUFFIX}/g" python/pylibcugraph/pyproject.toml

# cugraph pyproject.toml cuda suffixes
sed -i "s/name = \"cugraph\"/name = \"cugraph${PACKAGE_CUDA_SUFFIX}\"/g" python/cugraph/pyproject.toml
sed -i "s/rmm/rmm${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml
sed -i "s/cudf/cudf${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml
sed -i "s/raft-dask/raft-dask${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml
sed -i "s/pylibcugraph/pylibcugraph${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml
sed -i "s/pylibraft/pylibraft${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml
sed -i "s/ucx-py/ucx-py${PACKAGE_CUDA_SUFFIX}/g" python/cugraph/pyproject.toml

if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" python/cugraph/pyproject.toml
fi

# TODO: Remove this once the dependency file generator supports matrix entries,
# https://github.com/rapidsai/dependency-file-generator/pull/48
sed -i "s/rmm==/rmm${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt
sed -i "s/cudf==/cudf${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt
sed -i "s/raft-dask==/raft-dask${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt
sed -i "s/pylibcugraph==/pylibcugraph${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt
sed -i "s/pylibraft==/pylibraft${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt
sed -i "s/ucx-py==/ucx-py${PACKAGE_CUDA_SUFFIX}==/g" requirements.txt

python -m pip install -r requirements.txt

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --no-build-isolation

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
