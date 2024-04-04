# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcugraph-python)

project(
  libcugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_CUGRAPH_OPS "Enable all functions that call cugraph-ops" ON)

if(NOT USE_CUGRAPH_OPS)
    message(STATUS "Disabling libcugraph functions that reference cugraph-ops")
    add_compile_definitions(NO_CUGRAPH_OPS)
endif()

set(BUILD_TESTS OFF)
set(BUILD_CUGRAPH_MG_TESTS OFF)
set(BUILD_CUGRAPH_OPS_CPP_TESTS OFF)
set(CUDA_STATIC_RUNTIME ON)
set(CUGRAPH_USE_CUGRAPH_OPS_STATIC ON)
set(CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL ON)
set(ALLOW_CLONE_CUGRAPH_OPS ON)

add_subdirectory(../../cpp cugraph-cpp)
