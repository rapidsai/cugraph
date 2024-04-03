#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="libcugraph"
package_dir="python/libcugraph"

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
fi
export PATH="/tmp/gha-tools/tools:${PATH}"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1512 cpp)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2251 cpp)

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

for dep in librmm libraft; do
    sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
done

cd "${package_dir}"

PIP_FIND_LINKS="${librmm_wheelhouse} ${libraft_wheelhouse}" python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

# Don't repair raft into this wheel. Everything else is fair game.
# TODO: Check if this works transitively, i.e. if raft links to something that
# it bundles in then will cugraph try to pull that into its own libs? If so,
# we'll need to programatically exclude raft's dependencies from cugraph's.
mkdir -p final_dist
python -m auditwheel repair --exclude libraft.so -w final_dist dist/*
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp final_dist
