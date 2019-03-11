#!/usr/bin/env bash
set -e

if [ "$BUILD_CUGRAPH" == "1" ]; then
  echo "Building cugraph"
  CUDA_REL=${CUDA:0:3}
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
  fi

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/cugraph -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c numba -c conda-forge -c defaults --python=$PYTHON
  else
    conda build conda/recipes/cugraph -c rapidsai/label/cf201901 -c nvidia/label/cf201901 -c numba -c conda-forge/label/cf201901 -c defaults --python=$PYTHON
  fi
fi
