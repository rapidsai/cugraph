# =============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

set(cugraph_version 22.08.00)

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.08/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

project(
  cugraph-python
  VERSION ${cugraph_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX
)

################################################################################
# - User Options  --------------------------------------------------------------
option(FIND_CUGRAPH_CPP "Search for existing CUGRAPH C++ installations before defaulting to local files"
       OFF
)

# If the user requested it,  we attempt to find CUGRAPH.
if(FIND_CUGRAPH_CPP)
  find_package(cugraph ${cugraph_version} REQUIRED)
else()
  set(cugraph_FOUND OFF)
endif()


# FIXME: pylibcugraph should be installing libcugraph in its directory and cugraph should link
# to it
if(NOT cugraph_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
  # languages for the C++ project even if this project does not require those languges.
  include(rapids-cuda)
  rapids_cuda_init_architectures(cugraph-python)
  enable_language(CUDA)

  # FIXME: check of this is necessary
  # Since cugraph only enables CUDA optionally, we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  
  include("${CMAKE_PROJECT_cugraph-python_INCLUDE}")

  # FIXME: are those parameter necessary?
  
  set(BUILD_TESTS OFF)
  set(BUILD_BENCHMARKS OFF)
  message(STATUS "installing packages")
  add_subdirectory(../../cpp cugraph-cpp)

  # Since there are multiple subpackages of cudf._lib that require access to libcudf, we place the
  # library in the _lib/cpp directory as a single source of truth and modify the other rpaths
  # appropriately.
  install(TARGETS cugraph DESTINATION cugraph/library)
endif()


include(rapids-cython)
rapids_cython_init()

# FIXME: Update build directory
# add_subdirectory(cugraph)
add_subdirectory(cugraph/centrality)
add_subdirectory(cugraph/community)
add_subdirectory(cugraph/components)
add_subdirectory(cugraph/cores)
add_subdirectory(cugraph/dask/centrality)
add_subdirectory(cugraph/dask/comms)
add_subdirectory(cugraph/dask/community)
add_subdirectory(cugraph/dask/components)
add_subdirectory(cugraph/dask/link_analysis)
add_subdirectory(cugraph/dask/structure)
add_subdirectory(cugraph/generators)
add_subdirectory(cugraph/internals)
add_subdirectory(cugraph/layout)
add_subdirectory(cugraph/linear_assignment)
add_subdirectory(cugraph/link_analysis)
add_subdirectory(cugraph/link_prediction)
add_subdirectory(cugraph/sampling)
add_subdirectory(cugraph/structure)
add_subdirectory(cugraph/tree)
add_subdirectory(cugraph/utilities)
