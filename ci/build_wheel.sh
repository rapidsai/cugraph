#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

echo "${version}" > VERSION

# For testing, will be removed when the package is released
git clone -b main https://github.com/rapidsai/rapids-build-backend.git
pushd rapids-build-backend
sed -e 's/^version =.*/version = "0.0.2"/' -i pyproject.toml
python -m pip wheel . --no-deps
popd

cd "${package_dir}"

PIP_FIND_LINKS="$PWD/rapids-build-backend" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

# pure-python packages should be marked as pure, and not have auditwheel run on them.
if [[ ${package_name} == "nx-cugraph" ]] || \
   [[ ${package_name} == "cugraph-dgl" ]] || \
   [[ ${package_name} == "cugraph-pyg" ]] || \
   [[ ${package_name} == "cugraph-equivariant" ]]; then
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 dist
else
    mkdir -p final_dist
    python -m auditwheel repair -w final_dist dist/*
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
fi
