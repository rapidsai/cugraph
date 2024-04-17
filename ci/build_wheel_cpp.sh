#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="libcugraph"
package_dir="python/libcugraph"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

pyproject_file="${package_dir}/pyproject.toml"
version_file="${package_dir}/${package_name}/_version.py"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

for dep in librmm libraft libcugraphops; do
    sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
done

cd "${package_dir}"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 cpp 0435e6b)
libcugraphops_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph-ops 629 cpp e7c6f06)

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --find-links ${librmm_wheelhouse} --find-links ${libraft_wheelhouse} --find-links ${libcugraphops_wheelhouse}

mkdir -p final_dist
python -m auditwheel repair --exclude libraft.so --exclude libcugraph-ops++.so -w final_dist dist/*
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp final_dist
