#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1404 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1404 python)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14576 cpp)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14576 python)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2049 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2049 python)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version}
echo "${version}" > VERSION

rapids-logger "Begin py build"

package_dir="python"
for package_name in pylibcugraph cugraph nx-cugraph cugraph-pyg cugraph-dgl; do
  underscore_package_name=$(echo "${package_name}" | tr "-" "_")
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/${underscore_package_name}/_version.py"
done

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  conda/recipes/pylibcugraph

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/nx-cugraph

# NOTE: nothing in the cugraph-service packages are CUDA-specific, but they are
# built on each CUDA platform to ensure they are included in each set of
# artifacts, since test scripts only install from one set of artifacts based on
# the CUDA version used for the test run.
version_file_cugraph_service_client="python/cugraph-service/client/cugraph_service_client/_version.py"
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_service_client}
version_file_cugraph_service_server="python/cugraph-service/server/cugraph_service_server/_version.py"
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_service_server}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-service

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    --channel "${LIBCUDF_CHANNEL}" \
    --channel "${CUDF_CHANNEL}" \
    --channel "${LIBRAFT_CHANNEL}" \
    --channel "${RAFT_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel pyg \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-pyg

  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    --channel "${LIBCUDF_CHANNEL}" \
    --channel "${CUDF_CHANNEL}" \
    --channel "${LIBRAFT_CHANNEL}" \
    --channel "${RAFT_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel dglteam \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-dgl
fi

rapids-upload-conda-to-s3 python
