#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(
  mg_graph_algorithms
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../../fetch_dependencies.cmake)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../../cmake/thirdparty/get_nccl.cmake)

add_executable(mg_graph_algorithms mg_graph_algorithms.cpp)
set_target_properties(mg_graph_algorithms PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(mg_graph_algorithms PRIVATE cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)
target_compile_features(mg_graph_algorithms PRIVATE cxx_std_17)
