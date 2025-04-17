#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  DGL_CHANNEL="dglteam/label/th23_cu118"
else
  DGL_CHANNEL="dglteam/label/th23_cu121"
fi

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION
RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_MAJOR_MINOR
RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"
export RAPIDS_VERSION_NUMBER

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --prepend-channel conda-forge \
  --prepend-channel nvidia \
  --prepend-channel "${DGL_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
XML_DIR_LIBCUGRAPH="$(pwd)/xml"
export XML_DIR_LIBCUGRAPH
mkdir -p "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-upload-docs
