# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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

# --- CUDA -- #
set(CUDA_STATIC_RUNTIME ON)

# --- cugraph ---#
set(BUILD_TESTS OFF)
set(BUILD_CUGRAPH_MG_TESTS OFF)
set(CMAKE_CUDA_LINEINFO OFF)

# --- raft -- #
set(USE_RAFT_STATIC OFF)
set(CUGRAPH_COMPILE_RAFT_LIB ON)

add_subdirectory(../../cpp cugraph-cpp)
