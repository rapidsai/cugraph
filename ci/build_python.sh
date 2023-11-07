#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

REPO="rmm"
PR_NUMBER="1095"
COMMIT=$(git ls-remote https://github.com/rapidsai/${REPO}.git refs/heads/pull-request/${PR_NUMBER} | cut -c1-7)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')
LIBRMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)


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
  conda/recipes/pylibcugraph

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cugraph

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
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
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  conda/recipes/cugraph-service

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel pyg \
    --channel pytorch \
    --channel pytorch-nightly \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    conda/recipes/cugraph-pyg

  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel dglteam \
    --channel pytorch \
    --channel pytorch-nightly \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    conda/recipes/cugraph-dgl
fi

rapids-upload-conda-to-s3 python
