#=============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(
  mg_graph_algorithms
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../../fetch_dependencies.cmake)

add_executable(sg_graph_algorithms sg_graph_algorithms.cpp)
set_target_properties(sg_graph_algorithms PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(sg_graph_algorithms PRIVATE cugraph::cugraph)
target_compile_features(sg_graph_algorithms PRIVATE cxx_std_17)
