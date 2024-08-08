#!/usr/bin/env bash
# Copyright (c) 2019-2024, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory

# NOTE: the "libcugraph" target also builds all single-gpu gtest binaries, and
# the "cpp-mgtests" target builds the multi-gpu gtest binaries (and requires the
# openmpi build dependencies). The conda package does NOT include these test
# binaries or extra dependencies, but these are built here for use in CI runs.

export LIBCUGRAPH_BUILD_DIR="${PREFIX}/tmp/cugraph_build"
export LIBCUGRAPH_ETL_BUILD_DIR="${PREFIX}/tmp/cugraph_etl_build"
./build.sh libcugraph libcugraph_etl cpp-mgtests -n -v --allgpuarch

# The cccl libcudacxx dir contains some broken symlinks and that causes
# rattler-build to fail out when trying to follow them to copy the files.
find ${LIBCUGRAPH_BUILD_DIR}/ -xtype l -delete

# The libarrow package contains a file that is somehow being installed accidentally, see
# https://github.com/prefix-dev/rattler-build/issues/979
# https://github.com/conda-forge/arrow-cpp-feedstock/issues/1478
rm -rf "${PREFIX}/share"

cmake --install ${LIBCUGRAPH_BUILD_DIR} --prefix ${PREFIX}/tmp/install/libcugraph/
cmake --install ${LIBCUGRAPH_ETL_BUILD_DIR} --prefix ${PREFIX}/tmp/install/libcugraph_etl/

for component in testing testing_c testing_mg; do
    cmake --install ${LIBCUGRAPH_BUILD_DIR} --component ${component} --prefix ${PREFIX}/tmp/install/libcugraph_components/${component}/
done

# This is a nonexistent component that we've been installing for no reason...
#cmake --install ${LIBCUGRAPH_ETL_BUILD_DIR} --component testing --prefix ${PREFIX}/tmp/install/libcugraph_etl_components/testing/
