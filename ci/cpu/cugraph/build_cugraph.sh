#!/usr/bin/env bash
set -e

if [ "$BUILD_CUGRAPH" == "1" ]; then
  echo "Building cugraph"
  CUDA_REL=${CUDA_VERSION%.*}

  conda build conda/recipes/cugraph --python=$PYTHON
fi
