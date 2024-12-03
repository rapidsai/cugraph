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

project(
  libcugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

option(USE_NVCOMP_RUNTIME_WHEEL "Use the nvcomp wheel at runtime instead of the system library" OFF)

# Check if cugraph is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cugraph package.
find_package(cugraph "${RAPIDS_VERSION}")

if(cugraph_FOUND)
  return()
endif()

unset(cugraph_FOUND)

# Find Python early so that later commands can use it
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)

set(BUILD_TESTS OFF)
set(BUILD_CUGRAPH_MG_TESTS OFF)
set(CMAKE_CUDA_LINEINFO OFF)
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_STATIC_MATH_LIBRARIES ON)
set(USE_RAFT_STATIC ON)
set(CUGRAPH_COMPILE_RAFT_LIB ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
  set(CUDA_STATIC_MATH_LIBRARIES OFF)
elseif(USE_CUDA_MATH_WHEELS)
  message(FATAL_ERROR "Cannot use CUDA math wheels with CUDA < 12.0")
endif()

add_subdirectory(../../cpp cugraph-cpp)

if(NOT CUDA_STATIC_MATH_LIBRARIES AND USE_CUDA_MATH_WHEELS)
  set(rpaths
    "$ORIGIN/../nvidia/cublas/lib"
    "$ORIGIN/../nvidia/cufft/lib"
    "$ORIGIN/../nvidia/curand/lib"
    "$ORIGIN/../nvidia/cusolver/lib"
    "$ORIGIN/../nvidia/cusparse/lib"
    "$ORIGIN/../nvidia/nvjitlink/lib"
  )
  set_property(TARGET cugraph PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()
