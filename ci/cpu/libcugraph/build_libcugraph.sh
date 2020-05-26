#!/usr/bin/env bash
set -e

if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  echo "Building libcugraph"
  CUDA_REL=${CUDA_VERSION%.*}
  
  conda build conda/recipes/libcugraph
fi
