#!/bin/bash

# Copyright (c) 2019-2022, NVIDIA CORPORATION.

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
LIBCUGRAPH_ETL_BUILD_DIR=${LIBCUGRAPH_ETL_BUILD_DIR:=${REPODIR}/cpp/libcugraph_etl/build}

VALIDARGS="clean uninstall uninstall_cmake_deps libcugraph libcugraph_etl cugraph pylibcugraph cpp-mgtests docs -v -g -n --allgpuarch --skip_cpp_tests -h --help"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                - remove all existing build artifacts and configuration (start over)
   uninstall            - uninstall libcugraph and cugraph from a prior build/install (see also -n)
   uninstall_cmake_deps - uninstall headers from external dependencies installed by cmake (raft, rmm, cuco, etc.) (see also -n)
   libcugraph           - build libcugraph.so and SG test binaries
   libcugraph_etl       - build libcugraph_etl.so and SG test binaries
   cugraph              - build the cugraph Python package
   pylibcugraph         - build the pylibcugraph Python package
   cpp-mgtests          - build libcugraph and libcugraph_etl MG tests. Builds MPI communicator, adding MPI as a dependency.
   docs                 - build the docs
 and <flag> is:
   -v                   - verbose build mode
   -g                   - build for debug
   -n                   - do not install after a successful build
   --allgpuarch         - build for all supported GPU architectures
   --skip_cpp_tests     - do not build the SG test binaries as part of the libcugraph and libcugraph_etl targets
   -h                   - print this text

 default action (no args) is to build and install 'libcugraph' then 'libcugraph_etl' then 'pylibcugraph' then 'cugraph' then 'docs' targets

 libcugraph build dir is: ${LIBCUGRAPH_BUILD_DIR}

 Set env var LIBCUGRAPH_BUILD_DIR to override libcugraph build dir.
"
CUGRAPH_BUILD_DIR=${REPODIR}/python/build
BUILD_DIRS="${LIBCUGRAPH_BUILD_DIR} ${LIBCUGRAPH_ETL_BUILD_DIR} ${CUGRAPH_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
CMAKE_VERBOSE_OPTION=""
BUILD_TYPE=Release
INSTALL_TARGET="--target install"
BUILD_CPP_TESTS=ON
BUILD_CPP_MG_TESTS=OFF
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
    CMAKE_VERBOSE_OPTION="--log-level=VERBOSE"
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
if hasArg --skip_cpp_tests; then
    BUILD_CPP_TESTS=OFF
fi
if hasArg cpp-mgtests; then
    BUILD_CPP_MG_TESTS=ON
fi

# If clean or uninstall targets given, run them prior to any other steps
if hasArg uninstall; then
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf ${INSTALL_PREFIX}/include/cugraph
        rm -f ${INSTALL_PREFIX}/lib/libcugraph.so
        rm -rf ${INSTALL_PREFIX}/include/cugraph_c
        rm -f ${INSTALL_PREFIX}/lib/libcugraph_c.so
        rm -rf ${INSTALL_PREFIX}/include/cugraph_etl
        rm -f ${INSTALL_PREFIX}/lib/libcugraph_etl.so
        rm -rf ${INSTALL_PREFIX}/lib/cmake/cugraph
        rm -rf ${INSTALL_PREFIX}/lib/cmake/cugraph_etl
    fi
    # This may be redundant given the above, but can also be used in case
    # there are other installed files outside of the locations above.
    if [ -e ${LIBCUGRAPH_BUILD_DIR}/install_manifest.txt ]; then
        xargs rm -f < ${LIBCUGRAPH_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
    fi
    # uninstall cugraph and pylibcugraph installed from a prior "setup.py
    # install"
    # FIXME: if multiple versions of these packages are installed, this only
    # removes the latest one and leaves the others installed. build.sh uninstall
    # can be run multiple times to remove all of them, but that is not obvious.
    pip uninstall -y cugraph pylibcugraph
fi

if hasArg uninstall_cmake_deps; then
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf ${INSTALL_PREFIX}/include/raft
        rm -rf ${INSTALL_PREFIX}/lib/cmake/raft
        rm -rf ${INSTALL_PREFIX}/include/cub
        rm -rf ${INSTALL_PREFIX}/lib/cmake/cub
        rm -rf ${INSTALL_PREFIX}/include/cuco
        rm -rf ${INSTALL_PREFIX}/lib/cmake/cuco
        rm -rf ${INSTALL_PREFIX}/include/rmm
        rm -rf ${INSTALL_PREFIX}/lib/cmake/rmm
    fi
fi

if hasArg clean; then
    # Ignore errors for clean since missing files, etc. are not failures
    set +e
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
        find . -type d -name _external_repositories -exec rm -rf {} \; > /dev/null 2>&1
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
    # Go back to failing on first error for all other operations
    set -e
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
    cmake -B "${LIBCUGRAPH_BUILD_DIR}" -S "${REPODIR}/cpp" \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUGRAPH_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_TESTS=${BUILD_CPP_TESTS} \
          -DBUILD_CUGRAPH_MG_TESTS=${BUILD_CPP_MG_TESTS} \
          ${CMAKE_VERBOSE_OPTION}
    cmake --build "${LIBCUGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} ${INSTALL_TARGET} ${VERBOSE_FLAG}
fi

# Configure, build, and install libcugraph_etl
if buildAll || hasArg libcugraph_etl; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        CUGRAPH_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        CUGRAPH_CMAKE_CUDA_ARCHITECTURES="ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi
    mkdir -p ${LIBCUGRAPH_ETL_BUILD_DIR}
     cd ${LIBCUGRAPH_ETL_BUILD_DIR}
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${CUGRAPH_CMAKE_CUDA_ARCHITECTURES} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DBUILD_TESTS=${BUILD_CPP_TESTS} \
          -DBUILD_CUGRAPH_MG_TESTS=${BUILD_CPP_MG_TESTS} \
          ${CMAKE_VERBOSE_OPTION} \
          ${REPODIR}/cpp/libcugraph_etl
    cmake --build "${LIBCUGRAPH_ETL_BUILD_DIR}" -j${PARALLEL_LEVEL} ${INSTALL_TARGET} ${VERBOSE_FLAG}
fi

# Build, and install pylibcugraph
if buildAll || hasArg pylibcugraph; then
    cd ${REPODIR}/python/pylibcugraph
    # setup.py references an env var CUGRAPH_BUILD_PATH to find the libcugraph
    # build. If not set by the user, set it to LIBCUGRAPH_BUILD_DIR
    CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH:=${LIBCUGRAPH_BUILD_DIR}}
    env CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH} python setup.py build_ext --inplace --library-dir=${LIBCUGRAPH_BUILD_DIR}
    if [[ ${INSTALL_TARGET} != "" ]]; then
	env CUGRAPH_BUILD_PATH=${CUGRAPH_BUILD_PATH} python setup.py install
    fi
fi

# Build and install the cugraph Python package
if buildAll || hasArg cugraph; then
    cd ${REPODIR}/python/cugraph
    # FIXME: this needs to eventually reference the pylibcugraph build
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
        cmake -B "${LIBCUGRAPH_BUILD_DIR}" -S "${REPODIR}/cpp" \
              -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              ${CMAKE_VERBOSE_OPTION}
    fi
    cd ${LIBCUGRAPH_BUILD_DIR}
    cmake --build "${LIBCUGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} --target docs_cugraph ${VERBOSE_FLAG}
    cd ${REPODIR}/docs/cugraph
    make html
fi
