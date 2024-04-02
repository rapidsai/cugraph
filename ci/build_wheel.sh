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
RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_VERSION="3.11" rapids-download-wheels-from-s3 /tmp/libcugraph_dist

artifact_name=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_REPOSITORY=rmm RAPIDS_PY_VERSION="3.11" rapids-package-name wheel_python)
commit=$(git ls-remote https://github.com/rapidsai/rmm.git refs/heads/pull-request/1512 | cut -c1-7)
librmm_wheelhouse=$(rapids-get-artifact "ci/rmm/pull-request/1512/${commit}/${artifact_name}")

artifact_name=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_REPOSITORY=raft RAPIDS_PY_VERSION="3.11" rapids-package-name wheel_python)
commit=$(git ls-remote https://github.com/rapidsai/raft.git refs/heads/pull-request/2251 | cut -c1-7)
libraft_wheelhouse=$(rapids-get-artifact "ci/raft/pull-request/2251/${commit}/${artifact_name}")

artifact_name=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_REPOSITORY=raft rapids-package-name wheel_python)
commit=$(git ls-remote https://github.com/rapidsai/raft.git refs/heads/pull-request/2251 | cut -c1-7)
pylibraft_wheelhouse=$(rapids-get-artifact "ci/raft/pull-request/2251/${commit}/${artifact_name}")

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
version_package_name="$underscore_package_name"
if [[ "${version_package_name}" = "nx_cugraph" ]]; then
    version_package_name="_nx_cugraph"
fi
pyproject_file="${package_dir}/pyproject.toml"
version_file="${package_dir}/${version_package_name}/_version.py"

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

for dep in rmm cudf cugraph raft-dask pylibcugraph pylibcugraphops pylibraft ucx-py; do
    sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
done

# dask-cuda & rapids-dask-dependency doesn't get a suffix, but it does get an alpha spec.
for dep in dask-cuda rapids-dask-dependency; do
    sed -r -i "s/${dep}==(.*)\"/${dep}==\1${alpha_spec}\"/g" ${pyproject_file}
done


if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

cd "${package_dir}"

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

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
