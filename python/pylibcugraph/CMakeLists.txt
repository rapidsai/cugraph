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

if(POLICY CMP0094)
  # Make FindPython{,2,3} modules use LOCATION for lookup strategy.
  # https://cmake.org/cmake/help/latest/policy/CMP0094.html#policy:CMP0094
  cmake_policy(SET CMP0094 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0094 NEW)
endif()

include(../../fetch_rapids.cmake)

include(rapids-cuda)
# TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
# languages for the C++ project even if this project does not require those languges.
rapids_cuda_init_architectures(pylibcugraph-python)

set(pylibcugraph_version 22.10.00)

project(
  pylibcugraph-python
  VERSION ${pylibcugraph_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

################################################################################
# - User Options  --------------------------------------------------------------
option(FIND_CUGRAPH_CPP "Search for existing CUGRAPH C++ installations before defaulting to local files"
       OFF
)
option(CUGRAPH_BUILD_WHEELS "Whether we're building a wheel for pypi" OFF)

# If the user requested it we attempt to find CUGRAPH.

if(FIND_CUGRAPH_CPP)
  message(STATUS "Trying to find the package")
  find_package(cugraph ${pylibcugraph_version} REQUIRED)
else()
  set(BUILD_TESTS OFF)
  set(BUILD_CUGRAPH_MG_TESTS OFF)
  set(BUILD_CUGRAPH_OPS_CPP_TESTS OFF)

  # Statically link dependencies if building wheels
  set(CUDA_STATIC_RUNTIME ${CUGRAPH_BUILD_WHEELS})
  set(CUGRAPH_USE_CUGRAPH_OPS_STATIC ${CUGRAPH_BUILD_WHEELS})
  set(CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL ${CUGRAPH_BUILD_WHEELS})

  set(_exclude_from_all "")
  if(CUGRAPH_BUILD_WHEELS)
    # Don't install the cuML C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp cugraph-cpp ${_exclude_from_all})

  install(TARGETS cugraph cugraph_c DESTINATION pylibcugraph)
endif()

include(rapids-cython)
rapids_cython_init()

add_subdirectory(pylibcugraph)
