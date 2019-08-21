#!/usr/bin/env bash
set -e

if [ "$BUILD_LIBCUGRAPH" == '1' ]; then
  echo "Building libcugraph"
  CUDA_REL=${CUDA_VERSION%.*}

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/libcugraph -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c numba -c conda-forge -c defaults
  else
    conda build conda/recipes/libcugraph -c nvidia/label/cf201901-cuda${CUDA_REL} -c rapidsai/label/cf201901-cuda${CUDA_REL} -c numba -c conda-forge/label/cf201901 -c defaults
  fi
fi
