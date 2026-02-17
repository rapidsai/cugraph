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

rapids_cuda_init_architectures(sg_graph_operations)

project(
  sg_graph_algorithms
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../../fetch_dependencies.cmake)

add_executable(sg_graph_algorithms sg_graph_algorithms.cpp)
set_target_properties(sg_graph_algorithms PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(sg_graph_algorithms PRIVATE cugraph::cugraph)
target_compile_features(sg_graph_algorithms PRIVATE cxx_std_17)
