#!/bin/bash
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Any failing command will set EXITCODE to non-zero
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

NOTEBOOKS_DIR=${WORKSPACE}/notebooks
NBTEST=${WORKSPACE}/ci/utils/nbtest.sh
LIBCUDF_KERNEL_CACHE_PATH=${WORKSPACE}/.jitcache
EXITCODE=0
RUN_TYPE=$1

cd ${NOTEBOOKS_DIR}
TOPLEVEL_NB_FOLDERS=$(find . -name *.ipynb |cut -d'/' -f2|sort -u)

## Check env
env

# Do not abort the script on error. This allows all tests to run regardless of
# pass/fail but relies on the ERR trap above to manage the EXITCODE for the
# script.
set +e

# Always run nbtest in all TOPLEVEL_NB_FOLDERS, set EXITCODE to failure
# if any run fails
for folder in ${TOPLEVEL_NB_FOLDERS}; do
    echo "========================================"
    echo "FOLDER: ${folder}"
    echo "========================================"
    cd ${NOTEBOOKS_DIR}/${folder}
    if [ -z "$1" ]
        then
            runtype="all"
        else
            runtype=${1}
    fi
    NBLIST=$(python ${WORKSPACE}/ci/gpu/notebook_list.py ${runtype})
    for nb in ${NBLIST}; do
        nbBasename=$(basename ${nb})
        cd $(dirname ${nb})
        nvidia-smi
        ${NBTEST} ${nbBasename}
        echo "Ran nbtest for $nb : return code was: $?, test script exit code is now: $EXITCODE"
        echo
        rm -rf ${LIBCUDF_KERNEL_CACHE_PATH}/*
        cd ${NOTEBOOKS_DIR}/${folder}
    done
done

nvidia-smi

echo "Notebook test script exiting with value: $EXITCODE"
exit ${EXITCODE}
