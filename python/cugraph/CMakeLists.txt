# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

# We always need CUDA for cuML because the raft dependency brings in a
# header-only cuco dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cugraph-python)

project(
  cugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_CUGRAPH_OPS "Enable all functions that call cugraph-ops" ON)

if(NOT USE_CUGRAPH_OPS)
    message(STATUS "Disabling libcugraph functions that reference cugraph-ops")
    add_compile_definitions(NO_CUGRAPH_OPS)
endif()

find_package(cugraph "${RAPIDS_VERSION}" REQUIRED)

# TODO: Wait until this is implemented in rapids-cmake. At that point also
# should have the llinking automated as part of rapids_cython_create_modules.
rapids_cython_register_site_packages_library(cugraph::cugraph)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(cugraph/components)
add_subdirectory(cugraph/dask/comms)
add_subdirectory(cugraph/dask/structure)
add_subdirectory(cugraph/internals)
add_subdirectory(cugraph/layout)
add_subdirectory(cugraph/linear_assignment)
add_subdirectory(cugraph/structure)
add_subdirectory(cugraph/tree)
add_subdirectory(cugraph/utilities)
