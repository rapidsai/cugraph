#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

touch /tmp/requirements-build.txt

if [[ "${package_name}" == "cuspatial" ]]; then
    # Download the pylibcugraph wheel built in the previous step and ensure
    # it's installed in the build environment.
    PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python /tmp/pylibcugraph_dist)
    echo ${PYLIBCUGRAPH_WHEELHOUSE}/pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl >> /tmp/requirements-build.txt
fi

rapids-logger "Generating build requirements"
declare -r matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name/-/_}" \
  --matrix "${matrix_selectors}" \
| tee -a /tmp/requirements-build.txt

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_rapids_build_${package_name/-/_}" \
  --matrix "${matrix_selectors}" \
| tee -a /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

rapids-generate-version > ./VERSION

cd "${package_dir}"

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-build-isolation \
    --no-deps \
    --disable-pip-version-check \
    --extra-index-url https://pypi.nvidia.com \
    .

sccache --show-adv-stats

# pure-python packages should be marked as pure, and not have auditwheel run on them.
if [[ ${package_name} == "nx-cugraph" ]] || \
   [[ ${package_name} == "cugraph-dgl" ]] || \
   [[ ${package_name} == "cugraph-pyg" ]] || \
   [[ ${package_name} == "cugraph-equivariant" ]]; then
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 dist
else
    case "${RAPIDS_CUDA_VERSION}" in
        12.*)
            EXCLUDE_ARGS=(
                --exclude "libcublas.so.12"
                --exclude "libcublasLt.so.12"
                --exclude "libcurand.so.10"
                --exclude "libcusolver.so.11"
                --exclude "libcusparse.so.12"
                --exclude "libnvJitLink.so.12"
            )
        ;;
        11.*)
            EXCLUDE_ARGS=()
        ;;
    esac

    mkdir -p final_dist
    python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
fi
