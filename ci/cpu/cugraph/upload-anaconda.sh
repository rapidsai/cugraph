#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

set -e

if [[ "$BUILD_CUGRAPH" == "1" && "$UPLOAD_CUGRAPH" == "1" ]]; then
  export UPLOADFILE=`conda build conda/recipes/cugraph -c rapidsai -c nvidia -c numba -c conda-forge -c defaults --python=$PYTHON --output`

  SOURCE_BRANCH=master

  # Have to label all CUDA versions due to the compatibility to work with any CUDA
  if [ "$LABEL_MAIN" == "1" ]; then
    LABEL_OPTION="--label main"
  elif [ "$LABEL_MAIN" == "0" ]; then
    LABEL_OPTION="--label dev"
  else
    echo "Unknown label configuration LABEL_MAIN='$LABEL_MAIN'"
    exit 1
  fi
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${UPLOADFILE}

  # Restrict uploads to master branch
  if [ ${GIT_BRANCH} != ${SOURCE_BRANCH} ]; then
    echo "Skipping upload"
    return 0
  fi

  if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
  fi

  echo "Upload"
  echo ${UPLOADFILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${UPLOADFILE}
else
    echo "Skipping cugraph upload"
    return 0
fi
