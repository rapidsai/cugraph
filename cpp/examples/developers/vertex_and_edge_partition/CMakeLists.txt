#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(
  graph_partition
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../../fetch_dependencies.cmake)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../../cmake/thirdparty/get_nccl.cmake)

add_executable(vertex_and_edge_partition vertex_and_edge_partition.cu)
set_target_properties(vertex_and_edge_partition PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(vertex_and_edge_partition PRIVATE cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)
target_compile_features(vertex_and_edge_partition PRIVATE cxx_std_17)
