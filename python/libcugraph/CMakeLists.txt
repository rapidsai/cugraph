# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcugraph-python)

project(
  libcugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Check if cugraph is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of Python packages that need libcugraph.
find_package(cugraph "${RAPIDS_VERSION}")

if(cugraph_FOUND)
  return()
endif()

unset(cugraph_FOUND)

# --- cugraph ---#
set(BUILD_TESTS OFF)
set(BUILD_CUGRAPH_MG_TESTS OFF)
set(CMAKE_CUDA_LINEINFO OFF)

# --- raft -- #
set(USE_RAFT_STATIC OFF)
set(CUGRAPH_COMPILE_RAFT_LIB ON)

# --- cuvs --- #
set(CUGRAPH_COMPILE_CUVS ON)
set(CUGRAPH_USE_CUVS_STATIC ON)

add_subdirectory(../../cpp cugraph-cpp)

# assumes libcugraph.so is installed 2 levels deep, e.g. site-packages/libcugraph/lib64/libcugraph.so
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # starting with CTK 13 wheels, all libraries are grouped in this 'nvidia/cu13/lib' directory
  set(rpaths "$ORIGIN" "$ORIGIN/../../nvidia/cu13/lib")
else()
  set(rpaths
      "$ORIGIN"
      "$ORIGIN/../../nvidia/cublas/lib"
      "$ORIGIN/../../nvidia/curand/lib"
      "$ORIGIN/../../nvidia/cusolver/lib"
      "$ORIGIN/../../nvidia/cusparse/lib"
      "$ORIGIN/../../nvidia/nvjitlink/lib"
  )
endif()

set_property(
  TARGET cugraph
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cugraph_common
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cugraph_mg
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cugraph_mtmg
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cugraph_c
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
