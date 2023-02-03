#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -Eeuo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcugraph pylibcugraph cugraph

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
NOTEBOOK_LIST="$(realpath "$(dirname "$0")/gpu/notebook_list.py")"
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
