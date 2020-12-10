#!/bin/bash

# Copyright (c) 2019-2020, NVIDIA CORPORATION.

# cugraph build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)
LIBCUGRAPH_BUILD_DIR=${LIBCUGRAPH_BUILD_DIR:=${REPODIR}/cpp/build}

VALIDARGS="clean libcugraph cugraph docs -v -g -n --allgpuarch --show_depr_warn -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libcugraph       - build the cugraph C++ code
   cugraph          - build the cugraph Python package
   docs             - build the docs
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   -n               - no install step
   --allgpuarch     - build for all supported GPU architectures
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build and install 'libcugraph' then 'cugraph' then 'docs' targets

 libcugraph build dir is: ${LIBCUGRAPH_BUILD_DIR}

 Set env var LIBCUGRAPH_BUILD_DIR to override libcugraph build dir.
"
CUGRAPH_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUGRAPH_BUILD_DIR} ${CUGRAPH_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_DISABLE_DEPRECATION_WARNING=ON
GPU_ARCH=""

# Set defaults for vars that may not have been defined externally
#  FIXME: if PREFIX is not set, check CONDA_PREFIX, but there is no fallback
#  from there!
INSTALL_PREFIX=${PREFIX:=${CONDA_PREFIX}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildAll {
    (( ${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-]\+ ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
	if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
	    echo "Invalid option: ${a}"
	    exit 1
	fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    GPU_ARCH="-DGPU_ARCHS=ALL"
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi

# If clean given, run it prior to any other steps
if hasArg clean; then
    # FIXME: ideally the "setup.py clean" command below would also be run to
    # remove all the "inplace" python build artifacts, but currently, running
    # any setup.py command has side effects (eg. cloning repos).
    #(cd ${REPODIR}/python && python setup.py clean)

    # If the dirs to clean are mounted dirs in a container, the contents should
    # be removed but the mounted dirs will remain.  The find removes all
    # contents but leaves the dirs, the rmdir attempts to remove the dirs but
    # can fail safely.
    for bd in ${BUILD_DIRS}; do
	if [ -d ${bd} ]; then
	    find ${bd} -mindepth 1 -delete
	    rmdir ${bd} || true
	fi
    done
fi

################################################################################
# Configure, build, and install libcugraph
if buildAll || hasArg libcugraph; then
    if [[ ${GPU_ARCH} == "" ]]; then
        echo "Building for the architecture of the GPU in the system..."
    else
        echo "Building for *ALL* supported GPU architectures..."
    fi
    mkdir -p ${LIBCUGRAPH_BUILD_DIR}
    cd ${LIBCUGRAPH_BUILD_DIR}
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        ${GPU_ARCH} \
        -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${REPODIR}/cpp
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} ${INSTALL_TARGET}
fi

# Build and install the cugraph Python package
if buildAll || hasArg cugraph; then
    cd ${REPODIR}/python
    # setup.py references an env var CUGRAPH_BUILD_PATH to find the libcugraph
    # build. If not set by the user, set it to LIBCUGRAPH_BUILD_DIR
    CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH:=${LIBCUGRAPH_BUILD_DIR}}
    env CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH} python setup.py build_ext --inplace --library-dir=${LIBCUGRAPH_BUILD_DIR}
    if [[ ${INSTALL_TARGET} != "" ]]; then
	env CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH} python setup.py install
    fi
fi

# Build the docs
if buildAll || hasArg docs; then
    if [ ! -d ${LIBCUGRAPH_BUILD_DIR} ]; then
        mkdir -p ${LIBCUGRAPH_BUILD_DIR}
        cd ${LIBCUGRAPH_BUILD_DIR}
        cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
            -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${REPODIR}/cpp
    fi
    cd ${LIBCUGRAPH_BUILD_DIR}
    make -j${PARALLEL_LEVEL} VERBOSE=${VERBOSE} docs_cugraph
    cd ${REPODIR}/docs
    make html
fi
