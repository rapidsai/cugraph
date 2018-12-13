#=============================================================================
# Copyright 2018 BlazingDB, Inc.
#     Copyright 2018 Percy Camilo Triveño Aucahuasi <percy@blazingdb.com>
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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NVGGDF VERSION 0.1.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQURIED ON)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

IF(CUDA_VERSION_MAJOR GREATER 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_60,code=sm_60)
ENDIF(CUDA_VERSION_MAJOR GREATER 7)

IF(CUDA_VERSION_MAJOR GREATER 8)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=sm_70)
ENDIF(CUDA_VERSION_MAJOR GREATER 8)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-lineinfo)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

###################################################################################################
# - add cudf --------------------------------------------------------------------------------------
if(NOT TARGET cudf)
    find_path(CUDF_INCLUDE "cudf"
          HINTS "$ENV{CUDF_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include")

    find_library(CUDF_LIBRARY "cudf"
          HINTS "$ENV{CUDF_ROOT}/lib"
                "$ENV{CONDA_PREFIX}/lib")

    add_library(cudf SHARED IMPORTED)
    if (CUDF_INCLUDE AND CUDF_LIBRARY)
        set_target_properties( cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
    endif (CUDF_INCLUDE AND CUDF_LIBRARY)
endif()

###################################################################################################
# - add cugraph -----------------------------------------------------------------------------------

if(NOT TARGET cugraph)
    add_library( cugraph SHARED IMPORTED)
    set_target_properties( cugraph PROPERTIES IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libcugraph.so)
    message(STATUS "cugraph found in ${CMAKE_INSTALL_PREFIX}/lib/libcugraph.so")
endif()

###################################################################################################
# - add nvgraph -----------------------------------------------------------------------------------
if(NOT TARGET NVGRAPH)
    find_path(NVGRAPH_INCLUDE "nvgraph"
          HINTS "$ENV{NVGRAPH_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include")
    find_library(NVGRAPH_LIBRARY "nvgraph_st"
             HINTS "$ENV{NVGRAPH_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

    add_library( nvgraph SHARED IMPORTED)
    if (NVGRAPH_INCLUDE AND NVGRAPH_LIBRARY)
        set_target_properties( nvgraph PROPERTIES IMPORTED_LOCATION ${NVGRAPH_LIBRARY})
        message(STATUS "nvgraph found in ${NVGRAPH_LIBRARY}")
    endif (NVGRAPH_INCLUDE AND NVGRAPH_LIBRARY)
endif()
###################################################################################################
# - include paths ---------------------------------------------------------------------------------
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    "${CUDA_INCLUDE_DIRS}" 
    "${CUDF_INCLUDE}"
    "${NVGRAPH_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}"/../../../include
)

###################################################################################################
# - library targets -------------------------------------------------------------------------------
cuda_add_library(nvggdf SHARED
    nvgraph_gdf.cu
)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------
target_link_libraries(nvggdf cudf)
target_link_libraries(nvggdf cugraph)
target_link_libraries(nvggdf nvgraph)

###################################################################################################
# - install targets -------------------------------------------------------------------------------
install(TARGETS nvggdf LIBRARY DESTINATION lib)
install(FILES nvgraph_gdf.h DESTINATION include/cugraph)

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
