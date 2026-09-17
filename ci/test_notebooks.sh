#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

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

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  "${DEPENDENCY_FILE_GENERATOR_ARGS[@]}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
NOTEBOOK_LIST="$(realpath "$(dirname "$0")/notebook_list.py")"
EXITCODE=0
trap "EXITCODE=1" ERR


pushd notebooks
TOPLEVEL_NB_FOLDERS="$(find . -name "*.ipynb" | cut -d'/' -f2 | sort -u)"
set +e
# Always run nbtest in all TOPLEVEL_NB_FOLDERS, set EXITCODE to failure
# if any run fails
for folder in ${TOPLEVEL_NB_FOLDERS}; do
    rapids-logger "Folder: ${folder}"
    pushd "${folder}"
    NBLIST=$(python "${NOTEBOOK_LIST}" ci)
    for nb in ${NBLIST}; do
        nbBasename=$(basename "${nb}")
        pushd "$(dirname "${nb}")"
        nvidia-smi
        ${NBTEST} "${nbBasename}"
        echo "Ran nbtest for $nb : return code was: $?, test script exit code is now: $EXITCODE"
        echo
        popd
    done
    popd
done

nvidia-smi

echo "Notebook test script exiting with value: ${EXITCODE}"
exit ${EXITCODE}
