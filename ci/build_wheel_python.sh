#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist)

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 cpp efafdb6)
libcugraphops_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph-ops 629 cpp e7c6f06)
pylibraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 python efafdb6)

PYTHON_WHEELHOUSE="${PWD}/dist/"
PYTHON_PURE_WHEELHOUSE="${PWD}/pure_dist/"
PYTHON_AUDITED_WHEELHOUSE="${PWD}/final_dist/"

WHEELHOUSES=("${librmm_wheelhouse}" "${libraft_wheelhouse}" "${libcugraphops_wheelhouse}" "${pylibraft_wheelhouse}" "${CPP_WHEELHOUSE}" "${PYTHON_WHEELHOUSE}" "${PYTHON_PURE_WHEELHOUSE}")
mkdir -p "${PYTHON_AUDITED_WHEELHOUSE}"

FIND_LINKS=""
# Iterate over the array
for wheelhouse in "${WHEELHOUSES[@]}"; do
    FIND_LINKS+="--find-links ${wheelhouse} "
done

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

echo "${version}" > VERSION

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

build_wheel () {
    local package_name="${1}"
    local current_wheelhouse="${2}"

    local underscore_package_name=$(echo "${package_name}" | tr "-" "_")
    local version_package_name="$underscore_package_name"
    if [[ "${version_package_name}" = "nx_cugraph" ]]; then
        version_package_name="_nx_cugraph"
    fi

    local package_dir="python/${package_name}"
    local pyproject_file="${package_dir}/pyproject.toml"
    local version_file="${package_dir}/${version_package_name}/_version.py"

    sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
    sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

    for dep in rmm cudf cugraph libcugraph raft-dask pylibcugraph pylibcugraphops pylibraft ucx-py; do
        sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
    done

    # dask-cuda & rapids-dask-dependency don't get a suffix, but they do get an alpha spec.
    for dep in dask-cuda rapids-dask-dependency; do
        sed -r -i "s/${dep}==(.*)\"/${dep}==\1${alpha_spec}\"/g" ${pyproject_file}
    done

    if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
        sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
    fi

    python -m pip wheel "${package_dir}" -w ${current_wheelhouse} -vvv --no-deps --disable-pip-version-check ${FIND_LINKS}
}

build_wheel pylibcugraph "${PYTHON_WHEELHOUSE}"
build_wheel cugraph "${PYTHON_WHEELHOUSE}"
build_wheel nx-cugraph "${PYTHON_PURE_WHEELHOUSE}"
build_wheel cugraph-dgl "${PYTHON_PURE_WHEELHOUSE}"
build_wheel cugraph-pyg "${PYTHON_PURE_WHEELHOUSE}"
build_wheel cugraph-equivariant "${PYTHON_PURE_WHEELHOUSE}"

RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python ${PYTHON_PURE_WHEELHOUSE}

python -m auditwheel repair -w "${PYTHON_AUDITED_WHEELHOUSE}" --exclude libcugraph.so --exclude libcugraph_c.so --exclude libraft.so --exclude libcugraph-ops++.so ${PYTHON_WHEELHOUSE}/*
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python ${PYTHON_AUDITED_WHEELHOUSE}
