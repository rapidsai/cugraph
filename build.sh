#!/bin/bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION.

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

VALIDARGS="clean uninstall libcugraph cugraph docs -v -g -n --allgpuarch --buildfaiss --show_depr_warn -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   uninstall        - uninstall libcugraph and cugraph from a prior build/install (see also -n)
   libcugraph       - build the cugraph C++ code
   cugraph          - build the cugraph Python package
   cpp-mgtests      - build libcugraph mnmg tests. Builds MPI communicator, adding MPI as a dependency.
   docs             - build the docs
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   -n               - do not install after a successful build
   --allgpuarch     - build for all supported GPU architectures
   --buildfaiss     - build faiss statically into cugraph
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build and install 'libcugraph' then 'cugraph' then 'docs' targets

 libcugraph build dir is: ${LIBCUGRAPH_BUILD_DIR}

 Set env var LIBCUGRAPH_BUILD_DIR to override libcugraph build dir.
"
CUGRAPH_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUGRAPH_BUILD_DIR} ${CUGRAPH_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET=install
BUILD_DISABLE_DEPRECATION_WARNING=ON
BUILD_CPP_MG_TESTS=OFF
BUILD_STATIC_FAISS=OFF
BUILD_ALL_GPU_ARCH=0

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
    VERBOSE_FLAG="-v"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --buildfaiss; then
        BUILD_STATIC_FAISS=ON
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi
if hasArg cpp-mgtests; then
    BUILD_CPP_MG_TESTS=ON
fi

# If clean or uninstall given, run them prior to any other steps
if hasArg uninstall; then
    # uninstall libcugraph
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf ${INSTALL_PREFIX}/include/cugraph
        rm -f ${INSTALL_PREFIX}/lib/libcugraph.so
    fi
    # This may be redundant given the above, but can also be used in case
    # there are other installed files outside of the locations above.
    if [ -e ${LIBCUGRAPH_BUILD_DIR}/install_manifest.txt ]; then
        xargs rm -f < ${LIBCUGRAPH_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
    fi
    # uninstall cugraph installed from a prior "setup.py install"
    pip uninstall -y cugraph
fi

if hasArg clean; then
    # remove artifacts generated inplace
    # FIXME: ideally the "setup.py clean" command would be used for this, but
    # currently running any setup.py command has side effects (eg. cloning
    # repos).
    # (cd ${REPODIR}/python && python setup.py clean)
    if [[ -d ${REPODIR}/python ]]; then
        pushd ${REPODIR}/python > /dev/null
        rm -rf dist dask-worker-space cugraph/raft *.egg-info
        find . -name "__pycache__" -type d -exec rm -rf {} \; > /dev/null 2>&1
        find . -name "*.cpp" -type f -delete
        find . -name "*.cpython*.so" -type f -delete
        popd > /dev/null
    fi

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
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUGRAPH_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUGRAPH_CMAKE_CUDA_ARCHITECTURES="ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi
    mkdir -p ${LIBCUGRAPH_BUILD_DIR}
    cd ${LIBCUGRAPH_BUILD_DIR}
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUGRAPH_CMAKE_CUDA_ARCHITECTURES} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_STATIC_FAISS=${BUILD_STATIC_FAISS} \
          -DBUILD_CUGRAPH_MG_TESTS=${BUILD_CPP_MG_TESTS} \
          ${REPODIR}/cpp
    cmake --build "${LIBCUGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} --target ${INSTALL_TARGET} ${VERBOSE_FLAG}
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
            -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${REPODIR}/cpp \
            -DBUILD_STATIC_FAISS=${BUILD_STATIC_FAISS}
    fi
    cd ${LIBCUGRAPH_BUILD_DIR}
    cmake --build "${LIBCUGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} --target docs_cugraph ${VERBOSE_FLAG}
    cd ${REPODIR}/docs/cugraph
    make html
fi
