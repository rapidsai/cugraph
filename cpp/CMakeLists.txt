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

project(cuGraph VERSION 0.1.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------
include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
include(CTest)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

#option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#if(NOT CMAKE_CXX11_ABI)
#		message(STATUS "CUDF: Disabling the GLIBCXX11 ABI")
#		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#		set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
#elseif(CMAKE_CXX11_ABI)
#		message(STATUS "CUDF: Enabling the GLIBCXX11 ABI")
#endif(NOT CMAKE_CXX11_ABI)
	
#set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")

#find_package(CUDA)
#set_package_properties(
#    CUDA PROPERTIES TYPE REQUIRED
#    PURPOSE "NVIDIA CUDA® parallel computing platform and programming model."
#    URL "https://developer.nvidia.com/cuda-zone")

#if(CUDA_FOUND)
#    message(STATUS "CUDA ${CUDA_VERSION} found in ${CUDA_TOOLKIT_ROOT_DIR}")

#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std=c++11;--expt-extended-lambda; -Xcompiler -rdynamic -lineinfo)
#    # Suppress SHFL warnings caused by modern GPU.
#    # TODO: remove this when modern GPU is removed or fixed to use shfl_sync
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Wno-deprecated-declarations; -Xptxas --disable-warnings)
#    # set warnings as errors
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Werror cross-execution-space-call;-Xcompiler -Wall,-Wno-error=sign-compare)

#    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
#else()
#    message(FATAL_ERROR "CUDA not found, please check your settings.")
#endif()

message(STATUS "Using C++ standard: c++${CMAKE_CXX_STANDARD}")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})
message(STATUS "CMAKE_MODULE_PATH:" "${CMAKE_MODULE_PATH}")

#IF(CUDA_VERSION_MAJOR GREATER 7)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_60,code=sm_60)
#ENDIF(CUDA_VERSION_MAJOR GREATER 7)

#IF(CUDA_VERSION_MAJOR GREATER 8)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_70,code=sm_70)
#ENDIF(CUDA_VERSION_MAJOR GREATER 8)

#IF(CMAKE_BUILD_TYPE MATCHES DEBUG)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-lineinfo)
#ENDIF(CMAKE_BUILD_TYPE MATCHES DEBUG)


if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" OFF)
    if(CMAKE_CXX11_ABI)
        message(STATUS "cuGraph: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "cuGraph: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)
	
# Keith testing space
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")
	
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler -rdynamic -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations -Xptxas --disable-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable")
	
# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -lineinfo")
endif(CMAKE_BUILD_TYPE MATCHES Debug)
	
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
	

###################################################################################################
# - cmake custom modules --------------------------------------------------------------------------
include(ConfigureGoogleTest)

###################################################################################################
# - Find and add different modules and supporting repos -------------------------------------------
find_package(Boost 1.45.0 COMPONENTS system)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Git QUIET)
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/gunrock/")
    execute_process(COMMAND ${GIT_EXECUTABLE} log  -1 --pretty=format:"%ad" --date=format:%Y%m%d%H%M%S OUTPUT_VARIABLE LATEST_COMMIT_DATE WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gunrock/")
    set(REQUIRED_COMMIT_DATE "\"20181011022038\"")
    if(${REQUIRED_COMMIT_DATE} STRGREATER ${LATEST_COMMIT_DATE})
        execute_process(COMMAND ${GIT_EXECUTABLE} checkout master "${CMAKE_CURRENT_BINARY_DIR}/gunrock/")
        execute_process(COMMAND ${GIT_EXECUTABLE} pull WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/gunrock/")
    endif()
else()
    execute_process(COMMAND ${GIT_EXECUTABLE} clone --recursive  https://github.com/gunrock/gunrock "${CMAKE_CURRENT_BINARY_DIR}/gunrock/")
endif()

###################################################################################################
# - add cudf --------------------------------------------------------------------------------------
find_path(CUDF_INCLUDE "cudf"
          HINTS "$ENV{CUDF_ROOT}/include"
                "$ENV{CONDA_PREFIX}/include")

find_library(CUDF_LIBRARY "cudf"
             HINTS "$ENV{CUDF_ROOT}/lib"
                   "$ENV{CONDA_PREFIX}/lib")

add_library(cudf SHARED IMPORTED)
if (CUDF_INCLUDE AND CUDF_LIBRARY)
set_target_properties( cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
message(STATUS "GDF found in ${CUDF_LIBRARY}")
endif (CUDF_INCLUDE AND CUDF_LIBRARY)

###################################################################################################
# - add nvgraph -----------------------------------------------------------------------------------
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

###################################################################################################
# - include paths ---------------------------------------------------------------------------------
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/include" 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/cub"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" 
    "${CUDF_INCLUDE}"
    "${NVGRAPH_INCLUDE}"
    "${CMAKE_CURRENT_BINARY_DIR}/gunrock/"
    "${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/include"
    "${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/cub"
    ${Boost_INCLUDE_DIRS}
)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
# - library targets -------------------------------------------------------------------------------
add_library(cugraph SHARED
    src/grmat.cu
    src/cugraph.cu
    src/pagerank.cu
    src/bfs.cu
    src/nvgraph_gdf.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/test_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/error_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/misc_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/src/mgpucontext.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/src/mgpuutil.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/gitsha1.c
)

option(BUILD_TESTS "Configure CMake to build tests" OFF)

add_subdirectory(../thirdparty/rmm "${CMAKE_BINARY_DIR}/rmm")
include_directories("${CMAKE_SOURCE_DIR}/../thirdparty/rmm/include")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cugraph cudart cuda cudf ${Boost_LIBRARIES} rmm)

# Command to symlink files into the build directory
add_custom_command(  # link the include directory
    OUTPUT include
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/include include
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Configure the C++ tests
###################################################################################################
# - add gtest -------------------------------------------------------------------------------------
find_package(GTest QUIET)
set_package_properties(GTest PROPERTIES TYPE OPTIONAL
    PURPOSE "Google C++ Testing Framework (Google Test)."
    URL "https://github.com/google/googletest")


###################################################################################################
# - Generate the Tests -------------------------------------------------------------------------------
if(GTEST_FOUND)
    message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
    include_directories(${GTEST_INCLUDE_DIRS})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/tests)
else()
    message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
endif()


###################################################################################################
# - install targets -------------------------------------------------------------------------------
install(TARGETS cugraph LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/cugraph)

# Print the project summary
feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES)
