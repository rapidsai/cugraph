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
rapids-mamba-retry install "${PYTHON_CHANNEL}/linux-64/cugraph-dgl-*.tar.bz2"

export RAPIDS_VERSION_NUMBER="23.10"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libcugraph/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcugraph/html"
popd

rapids-logger "Build Python docs"
pushd docs/cugraph
# Ensure cugraph is importable, since sphinx does not report details about this
# type of failure well.
python -c "import cugraph; print(f'Using cugraph: {cugraph}')"
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
mkdir -p "${RAPIDS_DOCS_DIR}/cugraph/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/cugraph/html"
mv _text/* "${RAPIDS_DOCS_DIR}/cugraph/txt"
popd

rapids-upload-docs
