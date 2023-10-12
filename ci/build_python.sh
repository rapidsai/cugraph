#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version} 

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
version_file_pylibcugraph="python/pylibcugraph/pylibcugraph/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_pylibcugraph}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_pylibcugraph}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibcugraph

version_file_cugraph="python/cugraph/cugraph/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_cugraph}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
version_file_nx_cugraph="python/nx-cugraph/nx_cugraph/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_nx_cugraph}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_nx_cugraph}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/nx-cugraph

# NOTE: nothing in the cugraph-service packages are CUDA-specific, but they are
# built on each CUDA platform to ensure they are included in each set of
# artifacts, since test scripts only install from one set of artifacts based on
# the CUDA version used for the test run.
version_file_cugraph_service_client="python/cugraph-service/client/cugraph_service_client/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_cugraph_service_client}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_service_client}
version_file_cugraph_service_server="python/cugraph-service/server/cugraph_service_server/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_cugraph_service_server}
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_service_server}
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph-service

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  # Only CUDA 11 is supported right now due to PyTorch requirement.
  version_file_cugraph_pyg="python/cugraph-pyg/cugraph_pyg/_version.py"
  sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_cugraph_pyg}
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_pyg}
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel pyg \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-pyg

  # Only CUDA 11 is supported right now due to PyTorch requirement.
  version_file_cugraph_dgl="python/cugraph-dgl/cugraph_dgl/_version.py"
  sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file_cugraph_dgl}
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_cugraph_dgl}
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel dglteam \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-dgl
fi

rapids-upload-conda-to-s3 python
