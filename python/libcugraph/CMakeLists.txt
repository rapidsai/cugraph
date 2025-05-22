# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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
