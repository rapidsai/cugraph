#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

conda config --remove custom_multichannels
conda config --get
cat /opt/conda/.condarc

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
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
