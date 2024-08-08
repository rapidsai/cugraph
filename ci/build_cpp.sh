#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

# TODO: Upstream this to the image.
mamba install rattler-build -c conda-forge

# Notes on the comments in the command below (some things like file renamings
# should be done before merging):
# - rattler-build uses recipe.yaml by default, not meta.yaml.
# - rattler-build uses variants.yaml by default, not conda_build_config.yaml
# - rattler-build does not respect .condarc, so channels and the output dir
#   must be explicitly specified
# - The multi-output cache is currently an experimental feature.
#   (https://prefix-dev.github.io/rattler-build/dev/multiple_output_cache/)
# - By default rattler-build adds a timestamp that defeats sccache caching,
#   which --no-build-id turns off
RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rattler-build build \
    --recipe conda/recipes/libcugraph/meta.yaml \
    --variant-config conda/recipes/libcugraph/conda_build_config.yaml \
    -c rapidsai-nightly -c conda-forge \
    --output-dir ${RAPIDS_CONDA_BLD_OUTPUT_DIR} \
    --experimental \
    --no-build-id

echo "sccache stats:"
sccache -s

rapids-upload-conda-to-s3 cpp
