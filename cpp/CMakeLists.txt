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

project(CUGRAPH VERSION 0.8.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

###################################################################################################
###   C++ ABI changes.
###
###   By default, cugraph builds with the new C++ ABI.  In order to insure that thirdparty
###   applications build with the properly setting (specifically RMM) we need to set
###   the CMAKE_CXX11_ABI flag appropriately.
###
###   If a user wants to build with the OLD ABI, then they need to define CMAKE_CXX11_ABI
###   to be OFF (typically on the cmake command line).
###
###   This block of code will configure the old ABI if the flag is set to OFF and
###   do nothing (the default behavior of the C++14 compiler).
###
    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" ON)
    if(CMAKE_CXX11_ABI)
        message(STATUS "CUGRAPH: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "CUGRAPH: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable")

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
if (CMAKE_CUDA_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -Xcompiler -rdynamic")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)
if(BUILD_TESTS)
    include(CTest)
endif(BUILD_TESTS)

###################################################################################################
# - gunrock ---------------------------------------------------------------------------------------

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
# - find boost ------------------------------------------------------------------------------------

find_package(Boost REQUIRED)
if(Boost_FOUND)
    message(STATUS "Boost found in ${Boost_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found, please check your settings.")
endif(Boost_FOUND)

###################################################################################################
# - find openmp -----------------------------------------------------------------------------------

find_package(OpenMP)
if(OpenMP_FOUND)
    # find_package(OPenMP) does not automatically add OpenMP flags to CUDA
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

###################################################################################################
# - find gtest ------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS
            "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
    else()
        message(AUTHOR_WARNING
            "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - cudf ------------------------------------------------------------------------------------------

find_path(CUDF_INCLUDE "cudf"
    HINTS
    "$ENV{CUDF_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/cudf"
    "$ENV{CONDA_PREFIX}/include")

find_library(CUDF_LIBRARY "cudf"
    HINTS
    "$ENV{CUDF_ROOT}/lib"
    "$ENV{CONDA_PREFIX}/lib")

message(STATUS "CUDF: CUDF_LIBRARY set to ${CUDF_LIBRARY}")
message(STATUS "CUDF: CUDF_INCLUDE set to ${CUDF_INCLUDE}")

add_library(cudf SHARED IMPORTED ${CUDF_LIBRARY})
if (CUDF_INCLUDE AND CUDF_LIBRARY)
    set_target_properties(cudf PROPERTIES IMPORTED_LOCATION ${CUDF_LIBRARY})
endif (CUDF_INCLUDE AND CUDF_LIBRARY)

###################################################################################################
# - RMM -------------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

find_library(RMM_LIBRARY "rmm"
    HINTS
    "$ENV{RMM_ROOT}/lib"
    "$ENV{CONDA_PREFIX}/lib")

message(STATUS "RMM: RMM_LIBRARY set to ${RMM_LIBRARY}")
message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

add_library(rmm SHARED IMPORTED ${RMM_LIBRARY})
if (RMM_INCLUDE AND RMM_LIBRARY)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${RMM_LIBRARY})
endif (RMM_INCLUDE AND RMM_LIBRARY)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

# target_link_directories is added in cmake 3.13, and cmake advises to use this instead of
# link_directoires (we should switch to target_link_directories once 3.13 becomes the minimum
# required version).
link_directories(
     # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the
     # link directories for nvcc.
    "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

add_library(cugraph SHARED
    src/utilities/grmat.cu
    src/utilities/degree.cu
    src/utilities/cusparse_helper.cu
    src/structure/cugraph.cu
    src/link_analysis/pagerank.cu
    src/traversal/bfs.cu
    src/link_prediction/jaccard.cu
    src/link_prediction/overlap.cu
    src/converters/nvgraph.cu
    src/converters/renumber.cu
    src/community/nvgraph_gdf.cu
    src/traversal/nvgraph_sssp.cu
    src/traversal/two_hop_neighbors.cu
    src/snmg/blas/spmv.cu
    src/snmg/link_analysis/pagerank.cu
    src/utilities/cusparse_helper.cu
    src/snmg/utils.cu
    src/snmg/degree/degree.cu
    src/nvgraph/arnoldi.cu
    src/nvgraph/bfs.cu
    src/nvgraph/bfs2d.cu
    src/nvgraph/bfs_kernels.cu
    src/nvgraph/convert.cu
    src/nvgraph/csrmv.cu
    src/nvgraph/csrmv_cub.cu
    src/nvgraph/csr_graph.cpp
    src/nvgraph/graph_extractor.cu
    src/nvgraph/jaccard_gpu.cu
    src/nvgraph/kmeans.cu
    src/nvgraph/lanczos.cu
    src/nvgraph/lobpcg.cu
    src/nvgraph/matrix.cu
    src/nvgraph/modularity_maximization.cu
    src/nvgraph/nvgraph.cu
    src/nvgraph/nvgraph_cusparse.cpp
    src/nvgraph/nvgraph_cublas.cpp
    src/nvgraph/nvgraph_error.cu
    src/nvgraph/nvgraph_lapack.cu
    src/nvgraph/nvgraph_vector_kernels.cu
    src/nvgraph/pagerank.cu
    src/nvgraph/pagerank_kernels.cu
    src/nvgraph/partition.cu
    src/nvgraph/size2_selector.cu
    src/nvgraph/sssp.cu
    src/nvgraph/triangles_counting.cpp
    src/nvgraph/triangles_counting_kernels.cu
    src/nvgraph/valued_csr_graph.cpp
    src/nvgraph/widest_path.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/test_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/error_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/misc_utils.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/src/mgpucontext.cu
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/src/mgpuutil.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/gunrock/gunrock/util/gitsha1.c
)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

target_include_directories(cugraph
    PRIVATE
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${Boost_INCLUDE_DIRS}"
    "${RMM_INCLUDE}"
    "${CUDF_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/cub"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/gunrock/"
    "${CMAKE_CURRENT_BINARY_DIR}/gunrock/externals/moderngpu/include"
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cugraph PRIVATE
    ${CUDF_LIBRARY} ${RMM_LIBRARY} cublas cusparse curand cusolver cudart cuda ${Boost_LIBRARIES})
if(OpenMP_CXX_FOUND)
target_link_libraries(cugraph PRIVATE
###################################################################################################
###   Use ${OpenMP_CXX_LIB_NAMES} instead of OpenMP::OpenMP_CXX to avoid the following warnings.
###
###   Cannot generate a safe runtime search path for target TARGET_NAME
###   because files in some directories may conflict with libraries in implicit
###   directories:
###   ...
###
###   libgomp.so is included in the conda base environment and copied to every new conda
###   environment. If a full file path is provided (e.g ${CUDF_LIBRARY} and ${RMM_LIBRARY}), cmake
###   extracts the directory path and adds the directory path to BUILD_RPATH (if BUILD_RPATH is not
###   disabled).
###
###   cmake maintains a system specific implicit directories (e.g. /lib, /lib/x86_64-linux-gnu,
###   /lib32, /lib32/x86_64-linux-gnu, /lib64, /lib64/x86_64-linux-gnu, /usr/lib,
###   /usr/lib/gcc/x86_64-linux-gnu/7, /usr/lib/x86_64-linux-gnu, /usr/lib32,
###   /usr/lib32/x86_64-linux-gnu, /usr/lib64, /usr/lib64/x86_64-linux-gnu,
###   /usr/local/cuda-10.0/lib64", /usr/local/cuda-10.0/lib64/stubs).
###
###   If a full path to libgomp.so is provided (which is the case with OpenMP::OpenMP_CXX), cmake
###   checks whether there is any other libgomp.so with the different full path (after resolving
###   soft links) in the search paths (implicit directoires + BUILD_RAPTH). There is one in the
###   path included in BUILD_RPATH when ${CUDF_LIBRARY} and ${RMM_LIBRARY} are added; this one can
###   potentially hide the one in the provided full path and cmake generates a warning (and RPATH
###   is searched before the directories in /etc/ld.so/conf; ld.so.conf does not coincide but
###   overlaps with implicit directories).
###
###   If we provide just the library names (gomp;pthread), cmake does not generate warnings (we
###   did not specify which libgomp.so should be loaded in runtime), and the one first found in
###   the search order is loaded (we can change the loaded library by setting LD_LIBRARY_PATH or
###   manually editing BUILD_RPATH).
###
###   Manually editing BUILD_RPATH:
###   set(TARGET_BUILD_RPATH "")
###   foreach(TMP_VAR_FULLPATH IN LISTS OpenMP_CXX_LIBRARIES)
###       get_filename_component(TMP_VAR_DIR ${TMP_VAR_FULLPATH} DIRECTORY)
###       string(APPEND TARGET_BUILD_RPATH "${TMP_VAR_DIR};")
###       get_filename_component(TMP_VAR_REALPATH ${TMP_VAR_FULLPATH} REALPATH)
###       get_filename_component(TMP_VAR_DIR ${TMP_VAR_REALPATH} DIRECTORY)
###       # cmake automatically removes duplicates, so skip checking.
###       string(APPEND TARGET_BUILD_RPATH "${TMP_VAR_DIR};")
###   endforeach()
###   string(APPEND TARGET_BUILD_RPATH "${CONDA_PREFIX}/lib")
###   message(STATUS "TARGET_BUILD_RPATH=${TARGET_BUILD_RPATH}")
###   set_target_properties(target PROPERTIES
###       BUILD_RPATH "${TARGET_BUILD_RPATH}")
    ${OpenMP_CXX_LIB_NAMES})
endif(OpenMP_CXX_FOUND)

###################################################################################################
# - generate tests --------------------------------------------------------------------------------

if(BUILD_TESTS)
    if(GTEST_FOUND)
        # target_link_directories is added in cmake 3.13, and cmake advises to use this instead of
        # link_directoires (we should switch to target_link_directories once 3.13 becomes the
        # minimum required version).
        link_directories(${GTEST_LIBRARY_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/src/tests)
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cugraph LIBRARY
    DESTINATION lib)

install(DIRECTORY include/
    DESTINATION include/cugraph)
