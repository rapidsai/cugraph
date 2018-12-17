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
#    $ make              # make libcugraph
#    $ make pytest       # trigger test
#    $ make install      # install libcugraph

PROJECT(libcugraph)

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
include(CTest)

# Include custom modules (see cmake directory)
include(ConfigureGoogleTest)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

find_package(CUDA)
set_package_properties(
    CUDA PROPERTIES TYPE REQUIRED
    PURPOSE "NVIDIA CUDA® parallel computing platform and programming model."
    URL "https://developer.nvidia.com/cuda-zone")

if(CUDA_FOUND)
    message(STATUS "CUDA ${CUDA_VERSION} found in ${CUDA_TOOLKIT_ROOT_DIR}")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--expt-extended-lambda; -Xcompiler -rdynamic -lineinfo)
    # Suppress SHFL warnings caused by modern GPU.
    # TODO: remove this when modern GPU is removed or fixed to use shfl_sync
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -Wno-deprecated-declarations;
        -Xptxas --disable-warnings)
    # set warnings as errors
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Werror cross-execution-space-call;-Xcompiler -Wall,-Werror,-Wno-error=sign-compare)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Werror cross-execution-space-call;-Xcompiler -Wall,-Wno-error=sign-compare)

    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
else()
    message(FATAL_ERROR "CUDA not found, please check your settings.")
endif()

find_package(Boost 1.45.0 COMPONENTS system)

find_package(Git QUIET)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/")
    execute_process(COMMAND ${GIT_EXECUTABLE} log  -1 --pretty=format:"%ad" --date=format:%Y%m%d%H%M%S OUTPUT_VARIABLE LATEST_COMMIT_DATE WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/")
    set(REQUIRED_COMMIT_DATE "\"20181011022038\"")
    if(${REQUIRED_COMMIT_DATE} STRGREATER ${LATEST_COMMIT_DATE})
        execute_process(COMMAND ${GIT_EXECUTABLE} checkout master "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/")
        execute_process(COMMAND ${GIT_EXECUTABLE} pull WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/")
    endif()
else()
    execute_process(COMMAND ${GIT_EXECUTABLE} clone --recursive  https://github.com/gunrock/gunrock "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/")
endif()

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include" 
    "${CUDA_INCLUDE_DIRS}" 
    "$ENV{CONDA_PREFIX}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/externals/moderngpu/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../gunrock/externals/cub"
    ${Boost_INCLUDE_DIRS}
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

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



cuda_add_library(cugraph SHARED
    src/grmat.cu
    src/cugraph.cu
    src/pagerank.cu
    ../gunrock/gunrock/util/test_utils.cu
    ../gunrock/gunrock/util/error_utils.cu
    ../gunrock/gunrock/util/misc_utils.cu
    ../gunrock/externals/moderngpu/src/mgpucontext.cu
    ../gunrock/externals/moderngpu/src/mgpuutil.cpp
    ../gunrock/gunrock/util/gitsha1.c
)
add_library( cudf SHARED IMPORTED)
set_target_properties( cudf PROPERTIES IMPORTED_LOCATION $ENV{CONDA_PREFIX}/lib/libcudf.so)
target_link_libraries(cugraph cudf)
target_link_libraries(cugraph ${Boost_LIBRARIES})
message(STATUS "GDF found in $ENV{CONDA_PREFIX}/lib/libcudf.so")
# Command to symlink files into the build directory
add_custom_command(  # link the include directory
    OUTPUT include
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# The install target
install(TARGETS cugraph LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/cugraph)

# Configure the C++ tests
find_package(GTest QUIET)
set_package_properties(GTest PROPERTIES TYPE OPTIONAL
    PURPOSE "Google C++ Testing Framework (Google Test)."
    URL "https://github.com/google/googletest")

if(NOT NVG_PLUGIN)
    set(NVG_PLUGIN "FALSE")
else()
    if(NOT EXISTS "/usr/local/lib/libnvgraph_st.so")
        set(NVG_PLUGIN "FALSE")
    endif()
    message(STATUS "Could not find Nvgraph plugin in /usr/local/lib/libnvgraph_st.so")
endif()
message(STATUS "Nvgraph plugin : ${NVG_PLUGIN}")

if(NVG_PLUGIN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/nvgraph)
endif()

if(GTEST_FOUND)
    message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/tests)
else()
    message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
endif()

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
