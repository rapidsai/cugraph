#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
VERSION_NUMBER=$(rapids-get-rapids-version-from-git)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcugraph \
  pylibcugraph \
  cugraph \
  cugraph-pyg \
  cugraph-service-server \
  cugraph-service-client \
  libcugraph_etl

# This command installs `cugraph-dgl` without its dependencies
# since this package can currently only run in `11.6` CTK environments
# due to the dependency version specifications in its conda recipe.
rapids-logger "Install cugraph-dgl"
set -x
rapids-mamba-retry install "${PYTHON_CHANNEL}/linux-64/cugraph-dgl-*.tar.bz2"
set +x

rapids-logger "Build Doxygen docs"
pushd cpp/doxygen
doxygen Doxyfile
popd

rapids-logger "Build Sphinx docs"
pushd docs/cugraph
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
popd


if [[ "${RAPIDS_BUILD_TYPE}" == "branch" ]]; then
  rapids-logger "Upload Docs to S3"
  aws s3 sync --no-progress --delete docs/cugraph/_html "s3://rapidsai-docs/cugraph/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete docs/cugraph/_text "s3://rapidsai-docs/cugraph/${VERSION_NUMBER}/txt"
  aws s3 sync --no-progress --delete cpp/doxygen/html "s3://rapidsai-docs/libcugraph/${VERSION_NUMBER}/html"
fi
