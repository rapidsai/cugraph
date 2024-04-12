#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
fi
export PATH="/tmp/gha-tools/tools:${PATH}"

# TODO: Enable dask query planning (by default) once some bugs are fixed.
# xref: https://github.com/rapidsai/cudf/issues/15027
export DASK_DATAFRAME__QUERY_PLANNING=False

package_name=$1
package_dir=$2

python_package_name=$(echo ${package_name}|sed 's/-/_/g')

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# nx-cugraph is a pure wheel, which is part of generating the download path
if [[ "${package_name}" == "nx-cugraph" ]]; then
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist
else
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
fi

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
rmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 python)
libraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 cpp)
pylibraft_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact raft 2264 python)
libcugraphops_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="libcugraphops_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph-ops 629 cpp f843746) 

RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist

# use 'ls' to expand wildcard before adding `[extra]` requires for pip
# pip creates wheels using python package names
python -m pip install $(ls ./dist/${python_package_name}*.whl)[test] --find-links /tmp/libcugraph_dist --find-links ${librmm_wheelhouse} --find-links ${rmm_wheelhouse} --find-links ${libraft_wheelhouse} --find-links ${pylibraft_wheelhouse} --find-links ${libcugraphops_wheelhouse}

# Run smoke tests for aarch64 pull requests
arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_${package_name}.py
else
    # Test runs that include tests that use dask require
    # --import-mode=append. See test_python.sh for details.
    # FIXME: Adding PY_IGNORE_IMPORTMISMATCH=1 to workaround conftest.py import
    # mismatch error seen by nx-cugraph after using pytest 8 and
    # --import-mode=append.
    RAPIDS_DATASET_ROOT_DIR=`pwd`/datasets \
    PY_IGNORE_IMPORTMISMATCH=1 \
    DASK_WORKER_DEVICES="0" \
    DASK_DISTRIBUTED__SCHEDULER__WORKER_TTL="1000s" \
    DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="1000s" \
    DASK_CUDA_WAIT_WORKERS_MIN_TIMEOUT="1000s" \
    python -m pytest \
       -v \
       --import-mode=append \
       --benchmark-disable \
       -k "not test_property_graph_mg and not test_bulk_sampler_io" \
       ./python/${package_name}/${python_package_name}/tests
fi
