#!/usr/bin/env bash

# Copyright (c) 2022, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory

# The standard "./build.sh cugraph_service" cannot be used here since a custom
# setup.py invocation must be done.  This is required to work around the bug
# with split packages described here:
# https://github.com/conda/conda-build/issues/3993
# This workaround was copied from this PR:
# https://github.com/rapidsai/ucx-split-feedstock/pull/28

cd "${SRC_DIR}/python/cugraph-service/server"
mkdir -p pip_cache
$PYTHON -m pip install --no-index --no-deps --ignore-installed --cache-dir ./pip_cache . -vv

#./build.sh cugraph-service
