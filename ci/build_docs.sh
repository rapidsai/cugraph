#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  CONDA_CUDA_VERSION="11.8"
  DGL_CHANNEL="dglteam/label/cu118"
else
  CONDA_CUDA_VERSION="12.1"
  DGL_CHANNEL="dglteam/label/cu121"
fi

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel conda-forge \
  --channel pyg \
  --channel nvidia \
  --channel "${DGL_CHANNEL}" \
  libcugraph \
  pylibcugraph \
  cugraph \
  cugraph-pyg \
  cugraph-dgl \
  cugraph-service-server \
  cugraph-service-client \
  libcugraph_etl \
  pylibcugraphops \
  pylibwholegraph \
  pytorch \
  "cuda-version=${CONDA_CUDA_VERSION}"

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

for PROJECT in libcugraphops libwholegraph; do
  rapids-logger "Download ${PROJECT} xml_tar"
  TMP_DIR=$(mktemp -d)
  export XML_DIR_${PROJECT^^}="$TMP_DIR"
  curl "https://d1664dvumjb44w.cloudfront.net/${PROJECT}/xml_tar/${RAPIDS_VERSION_NUMBER}/xml.tar.gz" | tar -xzf - -C "${TMP_DIR}"
done

rapids-logger "Build Doxygen docs"
pushd cpp
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-logger "Output temp dir: ${RAPIDS_DOCS_DIR}"

rapids-upload-docs
