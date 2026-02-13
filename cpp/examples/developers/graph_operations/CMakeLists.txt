#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(${CMAKE_CURRENT_LIST_DIR}/../../../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(graph_operations)

project(
  graph_operations
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../../fetch_dependencies.cmake)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../../cmake/thirdparty/get_nccl.cmake)

add_executable(graph_operations graph_operations.cu)

target_include_directories(graph_operations
  PRIVATE
    "../../../src"
)

set_target_properties(graph_operations PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(graph_operations PRIVATE cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)
target_compile_features(graph_operations PRIVATE cxx_std_17)
