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

if [[ "$BUILD_LIBCUGRAPH" == "1" && "$UPLOAD_LIBCUGRAPH" == "1" ]]; then
  CUDA_REL=${CUDA_VERSION%.*}

  export UPLOADFILE=`conda build conda/recipes/libcugraph --output`

  SOURCE_BRANCH=master

  LABEL_OPTION="--label main"
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
    echo "Skipping libcugraph upload"
fi
