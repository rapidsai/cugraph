#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  CONDA_CUDA_VERSION="11.8"
  DGL_CHANNEL="dglteam/label/th23_cu118"
else
  CONDA_CUDA_VERSION="12.1"
  DGL_CHANNEL="dglteam/label/th23_cu121"
fi

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"

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

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

rapids-print-env

rapids-mamba-retry install \

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
export XML_DIR_LIBCUGRAPH="$(pwd)/xml"
mkdir -p "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-upload-docs
