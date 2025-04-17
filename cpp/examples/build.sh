#!/bin/bash

# Copyright (c) 2024-2025, NVIDIA CORPORATION.

# script for building libcugraph examples

set -e

NUMARGS=$#
ARGS=$*

VALIDARGS="
   clean
   all
   -v
   -h
   --help
"

VERBOSE_FLAG=""
CMAKE_VERBOSE_OPTION=""

# Parallelism control
PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}

# Root of examples
EXAMPLES_ROOT_DIR=$(dirname "$(realpath "$0")")
EXAMPLES=(
    "users/single_gpu_application"
    "users/multi_gpu_application"
    "developers/vertex_and_edge_partition"
    "developers/graph_operations")

CUGRAPH_BUILD_DIR=${CUGRAPH_BUILD_DIR:-$(readlink -f "${EXAMPLES_ROOT_DIR}/../build")}

HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                      - remove all existing build artifacts and configuration (start over)
   all                        - build all of the examples from the following directories
                                  ${EXAMPLES_ROOT_DIR}/users
                                  ${EXAMPLES_ROOT_DIR}/developers
 where <flag> is:
   -v                         - verbose build mode
   -h                         - print this text
   --help                     - print this text
"

if (( NUMARGS == 0 )); then
    echo "${HELP}"
fi

# Check for valid usage
if (( NUMARGS != 0 )); then
    for a in ${ARGS}; do
        if ! (echo "${VALIDARGS}" | grep -q "^[[:blank:]]*${a}$"); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

if hasArg -v; then
    VERBOSE_FLAG="-v"
    CMAKE_VERBOSE_OPTION="--log-level=VERBOSE"
fi

if hasArg clean; then
    # Ignore errors for clean since missing files, etc. are not failures
    set +e
    for idx in "${!EXAMPLES[@]}"
    do
        current_example=${EXAMPLES[$idx]}
        build_dir="${EXAMPLES_ROOT_DIR}/${current_example}/build"
        if [ -d "${build_dir}" ]; then
            find "${build_dir}" -mindepth 1 -delete
            rmdir "${build_dir}" || true
            echo "Removed ${build_dir}"
        fi
    done
    # Go back to failing on first error for all other operations
    set -e
fi

build_example() {
  echo "building ${1}"
  example_dir=${1}
  example_dir="${EXAMPLES_ROOT_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S "${example_dir}" -B "${build_dir}" -Dcugraph_ROOT="${CUGRAPH_BUILD_DIR}" ${CMAKE_VERBOSE_OPTION}
  # Build
  cmake --build "${build_dir}" "-j${PARALLEL_LEVEL}" "${VERBOSE_FLAG}"
}

if hasArg all; then
    for idx in "${!EXAMPLES[@]}"
    do
        current_example=${EXAMPLES[$idx]}
        build_example "$current_example"
    done
fi
