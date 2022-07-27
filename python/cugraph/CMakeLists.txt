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

include(../../fetch_rapids.cmake)

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

if(NOT cugraph_FOUND)
  # TODO: This will not be necessary once we upgrade to CMake 3.22, which will pull in the required
  # languages for the C++ project even if this project does not require those languges.
  include(rapids-cuda)
  rapids_cuda_init_architectures(CUGRAPH)
  enable_language(CUDA)

  # Since cugraph only enables CUDA optionally, we need to manually include the file that
  # rapids_cuda_init_architectures relies on `project` including.
  
  include("${CMAKE_PROJECT_cugraph-python_INCLUDE}")

  add_subdirectory(../../cpp cugraph-cpp)

  install(TARGETS cugraph DESTINATION cugraph/library)
endif()


include(rapids-cython)
rapids_cython_init()

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
