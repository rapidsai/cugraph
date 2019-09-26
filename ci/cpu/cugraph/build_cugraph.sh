#!/usr/bin/env bash
set -e

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

if [ "$BUILD_CUGRAPH" == "1" ]; then
  echo "Building cugraph"
  CUDA_REL=${CUDA_VERSION%.*}

  conda build conda/recipes/cugraph --python=$PYTHON
fi
