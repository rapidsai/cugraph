#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cugraph cugraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

# A skipped 'conda-cpp-build' job (libcugraph unaffected by this PR) means no
# artifact was uploaded this run. Fail fast (short retry budget) rather than
# waiting out the full retry window, then fall back to resolving libcugraph
# from the rapidsai-nightly channel (already in the default channel list)
# via its own '==26.12.*,>=0.0.0a0'-style dependency constraint.
DEPENDENCY_FILE_GENERATOR_ARGS=(--prepend-channel "${PYTHON_CHANNEL}")
if CPP_CHANNEL=$(RAPIDS_RETRY_MAX=1 RAPIDS_RETRY_SLEEP=15 rapids-download-from-github "$(rapids-artifact-name conda_cpp libcugraph cugraph --cuda "$RAPIDS_CUDA_VERSION")" 2>/tmp/libcugraph_channel_download.log); then
  DEPENDENCY_FILE_GENERATOR_ARGS+=(--prepend-channel "${CPP_CHANNEL}")
else
  cat /tmp/libcugraph_channel_download.log >&2
  rapids-logger "No libcugraph artifact found for this run; resolving it from the nightly channel"
fi

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

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
  "${DEPENDENCY_FILE_GENERATOR_ARGS[@]}" \
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
