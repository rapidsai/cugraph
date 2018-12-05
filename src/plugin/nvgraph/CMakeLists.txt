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

# Usage:
#    $ mkdir build       # create directory for out-of-source build
#    $ cd build          # enter the build directory
#    $ cmake ..          # configure build system
#    $ make              # make libnvggdf
#    $ make pytest       # trigger test
#    $ make install      # install libnvggdf

PROJECT(libnvggdf)

cmake_minimum_required(VERSION 2.8)  # not sure about version required

set(CMAKE_CXX_STANDARD 11)
message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

#IF(CMAKE_COMPILER_IS_GNUCXX)
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
#ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# Include CMake modules
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
#include(CTest)

# Include custom modules (see cmake directory)
#include(ConfigureGoogleTest)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

#find_package(CUDA)
#set_package_properties(
#    CUDA PROPERTIES TYPE REQUIRED
#    PURPOSE "NVIDIA CUDA® parallel computing platform and programming model."
#    URL "https://developer.nvidia.com/cuda-zone")
#
#if(CUDA_FOUND)
#    message(STATUS "CUDA ${CUDA_VERSION} found in ${CUDA_TOOLKIT_ROOT_DIR}")
#
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--expt-extended-lambda; -Xcompiler -rdynamic -lineinfo)
#    # Suppress SHFL warnings caused by modern GPU.
#    # TODO: remove this when modern GPU is removed or fixed to use shfl_sync
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
#        -Wno-deprecated-declarations;
#        -Xptxas --disable-warnings)
#    # set warnings as errors
#    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Werror cross-execution-space-call;-Xcompiler -Wall,-Werror,-Wno-error=sign-compare)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Werror cross-execution-space-call;-Xcompiler -Wall,-Wno-error=sign-compare)
#
#    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
#else()
#    message(FATAL_ERROR "CUDA not found, please check your settings.")
#endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    "${CUDA_INCLUDE_DIRS}" 
    "$ENV{CONDA_PREFIX}/include/cudf"
    "${CMAKE_CURRENT_SOURCE_DIR}"/../../../include
    SYSTEM "/usr/local/include/nvgraph"
)

IF(CUDA_VERSION_MAJOR GREATER 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_60,code=sm_60)
ENDIF(CUDA_VERSION_MAJOR GREATER 7)

IF(CUDA_VERSION_MAJOR GREATER 8)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=sm_70)
ENDIF(CUDA_VERSION_MAJOR GREATER 8)

IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-lineinfo)
ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)

cuda_add_library(nvggdf SHARED
    nvgraph_gdf.cu
)
if(NOT TARGET cudf)
add_library( cudf SHARED IMPORTED)
set_target_properties( gdf PROPERTIES IMPORTED_LOCATION $ENV{CONDA_PREFIX}/lib/libcudf.so)
endif()
target_link_libraries(nvggdf cudf)

if(NOT TARGET cugraph)
add_library( cugraph SHARED IMPORTED)
set_target_properties( cugraph PROPERTIES IMPORTED_LOCATION ${CMAKE_INSTALL_PREFIX}/lib/libcugraph.so)
endif()
target_link_libraries(nvggdf cugraph)
message(STATUS "cugraph found in /usr/local/lib/libcugraph.so")

if(NOT TARGET nvgraph)
add_library( nvgraph SHARED IMPORTED)
set_target_properties( nvgraph PROPERTIES IMPORTED_LOCATION /usr/local/lib/libnvgraph_st.so)
endif()
target_link_libraries(nvggdf nvgraph)
message(STATUS "nvgraph found in /usr/local/lib/libnvgraph_st.so")
# Command to symlink files into the build directory
#add_custom_command(  # link the include directory
#    OUTPUT include
#    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include include
#    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The install target
install(TARGETS nvggdf LIBRARY DESTINATION lib)
install(FILES nvgraph_gdf.h DESTINATION include/cugraph)

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
