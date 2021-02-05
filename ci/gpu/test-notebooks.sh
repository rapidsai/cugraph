#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#RAPIDS_DIR=/rapids
NOTEBOOKS_DIR=${WORKSPACE}/notebooks
NBTEST=${WORKSPACE}/ci/utils/nbtest.sh
LIBCUDF_KERNEL_CACHE_PATH=${WORKSPACE}/.jitcache

cd ${NOTEBOOKS_DIR}
TOPLEVEL_NB_FOLDERS=$(find . -name *.ipynb |cut -d'/' -f2|sort -u)

## Check env
env

EXITCODE=0

# Always run nbtest in all TOPLEVEL_NB_FOLDERS, set EXITCODE to failure
# if any run fails
for folder in ${TOPLEVEL_NB_FOLDERS}; do
    echo "========================================"
    echo "FOLDER: ${folder}"
    echo "========================================"
    cd ${NOTEBOOKS_DIR}/${folder}
    for nb in $(python ${WORKSPACE}/ci/gpu/notebook_list.py); do
        nbBasename=$(basename ${nb})
        cd $(dirname ${nb})
        nvidia-smi
        ${NBTEST} ${nbBasename}
        EXITCODE=$((EXITCODE | $?))
        rm -rf ${LIBCUDF_KERNEL_CACHE_PATH}/*
        cd ${NOTEBOOKS_DIR}/${folder}
    done
done

nvidia-smi

exit ${EXITCODE}
