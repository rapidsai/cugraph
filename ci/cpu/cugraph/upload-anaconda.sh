#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$UPLOAD_CUGRAPH" == "1" ]; then
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
  anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --force ${UPLOADFILE}
else
    echo "Skipping cugraph upload"
    return 0
fi
