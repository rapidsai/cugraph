#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

cp -r ${PREFIX}/tmp/install/libcugraph_etl/* ${PREFIX}/

# The libarrow package contains a file that is somehow being installed accidentally, see
# https://github.com/prefix-dev/rattler-build/issues/979
# https://github.com/conda-forge/arrow-cpp-feedstock/issues/1478
rm -rf "${PREFIX}/share"
