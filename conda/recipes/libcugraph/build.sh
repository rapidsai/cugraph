#!/usr/bin/env bash
CMAKE_COMMON_VARIABLES=" -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX11_ABI=$BUILD_ABI -DNVG_PLUGIN=True"

# show environment
printenv
# Cleanup local git
if [ -d .git ]; then
    git clean -xdf
fi
# Use CMake-based build procedure
mkdir -p cpp/build
cd cpp/build
# configure
cmake $CMAKE_COMMON_VARIABLES ..
# build
make -j${PARALLEL_LEVEL} VERBOSE=1 install
