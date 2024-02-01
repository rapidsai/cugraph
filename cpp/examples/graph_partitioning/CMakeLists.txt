#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.23.1)

project(
  graph_partition_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../cmake/thirdparty/get_nccl.cmake)


add_library(csvfileutil STATIC ../../tests/utilities/csv_file_utilities.cu)

target_include_directories(csvfileutil
PUBLIC
"../../../thirdparty/mmio"
"../../tests"
PRIVATE
"../../include"
"../../src"
)

set_target_properties(csvfileutil PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(csvfileutil PUBLIC cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)

# Configure your project here
add_executable(graph_partition_examples graph_partition_examples.cpp)
target_link_libraries(graph_partition_examples PRIVATE csvfileutil)
target_compile_features(graph_partition_examples PRIVATE cxx_std_17)
