#=============================================================================
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.18...3.18 FATAL_ERROR)

project(CUGRAPH VERSION 0.19.0 LANGUAGES C CXX CUDA)

# Write the version header
include(cmake/Modules/Version.cmake)
write_version()

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

##############################################################################
# - User Options  ------------------------------------------------------------

option(BUILD_CUGRAPH_MG_TESTS "Build cuGraph multigpu algorithm tests" OFF)

###################################################################################################
# - user options ------------------------------------------------------------------------------

set(BLAS_LIBRARIES "" CACHE STRING
    "Location of BLAS library for FAISS build.")
option(BUILD_STATIC_FAISS "Build the FAISS library for nearest neighbors search on GPU" OFF)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-error=deprecated-declarations")
endif(CMAKE_COMPILER_IS_GNUCXX)

find_package(CUDA)

# Configure GPU arch to build
set(GUNROCK_GENCODE_SM60 "OFF")
set(GUNROCK_GENCODE_SM61 "OFF")
set(GUNROCK_GENCODE_SM70 "OFF")
set(GUNROCK_GENCODE_SM72 "OFF")
set(GUNROCK_GENCODE_SM75 "OFF")
set(GUNROCK_GENCODE_SM80 "OFF")

# ARCHS handling:
#
if("${GPU_ARCHS}" STREQUAL "")
  include(cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()


# CUDA 11 onwards cub ships with CTK
if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
  set(CUB_IS_PART_OF_CTK ON)
else()
  set(CUB_IS_PART_OF_CTK OFF)
endif()

if("${GPU_ARCHS}" STREQUAL "ALL")
  set(GPU_ARCHS "60")
  if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
    set(GPU_ARCHS "${GPU_ARCHS};70")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 10) OR (CUDA_VERSION_MAJOR GREATER 10))
    set(GPU_ARCHS "${GPU_ARCHS};75")
  endif()
  if((CUDA_VERSION_MAJOR EQUAL 11) OR (CUDA_VERSION_MAJOR GREATER 11))
    set(GPU_ARCHS "${GPU_ARCHS};80")
  endif()
endif()

message("-- Building for GPU_ARCHS = ${GPU_ARCHS}")
foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  set(GUNROCK_GENCODE_SM${arch} "ON")
  set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

list(GET GPU_ARCHS -1 ptx)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${ptx},code=compute_${ptx}")
set(FAISS_GPU_ARCHS "${FAISS_GPU_ARCHS} -gencode arch=compute_${ptx},code=compute_${ptx}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas --disable-warnings")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=-compress-all")

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

# To apply RUNPATH to transitive dependencies (this is a temporary solution)
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")

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
# - find blas -------------------------------------------------------------------------------------

if(NOT DEFINED BLAS_LIBRARIES)
  find_package( BLAS REQUIRED )
else()
  message(STATUS "Manually setting BLAS to ${BLAS_LIBRARIES}")
endif()

###################################################################################################
# - find gtest ------------------------------------------------------------------------------------

if(BUILD_TESTS)
    find_package(GTest REQUIRED)
endif(BUILD_TESTS)

###################################################################################################
# - find RMM --------------------------------------------------------------------------------------

find_path(RMM_INCLUDE "rmm"
    HINTS
    "$ENV{RMM_ROOT}/include"
    "$ENV{CONDA_PREFIX}/include/rmm"
    "$ENV{CONDA_PREFIX}/include")

message(STATUS "RMM: RMM_INCLUDE set to ${RMM_INCLUDE}")

###################################################################################################
# - find NCCL -------------------------------------------------------------------------------------

if(NOT NCCL_PATH)
    find_package(NCCL REQUIRED)
else()
    message("-- Manually set NCCL PATH to ${NCCL_PATH}")
    set(NCCL_INCLUDE_DIRS ${NCCL_PATH}/include)
    set(NCCL_LIBRARIES ${NCCL_PATH}/lib/libnccl.so)
endif(NOT NCCL_PATH)

###################################################################################################
# - find MPI - only enabled if MG tests are to be built

if(BUILD_CUGRAPH_MG_TESTS)
  find_package(MPI REQUIRED)
endif(BUILD_CUGRAPH_MG_TESTS)

###################################################################################################
# - Fetch Content ---------------------------------------------------------------------------------
include(FetchContent)

# - THRUST/CUB
message("Fetching Thrust")

FetchContent_Declare(
    thrust
    GIT_REPOSITORY https://github.com/thrust/thrust.git
    # August 28, 2020
    GIT_TAG        52a8bda46c5c2128414d1d47f546b486ff0be2f0
)

FetchContent_GetProperties(thrust)
if(NOT thrust_POPULATED)
  FetchContent_Populate(thrust)
  # We are not using the thrust CMake targets, so no need to call `add_subdirectory()`.
endif()
set(THRUST_INCLUDE_DIR "${thrust_SOURCE_DIR}")

# - cuco
message("Fetching cuco")

FetchContent_Declare(
    cuco
    GIT_REPOSITORY https://github.com/NVIDIA/cuCollections.git
    GIT_TAG        2196040f0562a0280292eebef5295d914f615e63
)

FetchContent_GetProperties(cuco)
if(NOT cuco_POPULATED)
  FetchContent_Populate(cuco)
endif()
set(CUCO_INCLUDE_DIR "${cuco_SOURCE_DIR}/include")

# - libcudacxx
#     NOTE:  This is necessary because libcudacxx is not supported in
#      debian cuda 10.2 packages.  Once 10.2 is deprecated
#      we should not need this any longer.
message("Fetching libcudacxx")

FetchContent_Declare(
    libcudacxx
    GIT_REPOSITORY https://github.com/NVIDIA/libcudacxx.git
    GIT_TAG        1.3.0
    GIT_SHALLOW    true
)

FetchContent_GetProperties(libcudacxx)
if(NOT libcudacxx_POPULATED)
  message("populating libcudacxx")
  FetchContent_Populate(libcudacxx)
endif()
set(LIBCUDACXX_INCLUDE_DIR "${libcudacxx_SOURCE_DIR}/include")
message("set LIBCUDACXX_INCLUDE_DIR to: ${LIBCUDACXX_INCLUDE_DIR}")

# - CUHORNET
FetchContent_Declare(
    cuhornet
    GIT_REPOSITORY    https://github.com/rapidsai/cuhornet.git
    GIT_TAG           9cb8e8803852bd895a9c95c0fe778ad6eeefa7ad
    GIT_SHALLOW       true
    SOURCE_SUBDIR     hornet
)

FetchContent_GetProperties(cuhornet)
if(NOT cuhornet_POPULATED)
  message("populating cuhornet")
  FetchContent_Populate(cuhornet)
  # We are not using the cuhornet CMake targets, so no need to call `add_subdirectory()`.
endif()
set(CUHORNET_INCLUDE_DIR ${cuhornet_SOURCE_DIR} CACHE STRING "Path to cuhornet includes")

# - raft - (header only)
# Only cloned if RAFT_PATH env variable is not defined
if(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable detected.")
  message(STATUS "RAFT_DIR set to $ENV{RAFT_PATH}")
  set(RAFT_DIR "$ENV{RAFT_PATH}")

else(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable NOT detected, cloning RAFT")

  FetchContent_Declare(
    raft
    GIT_REPOSITORY    https://github.com/rapidsai/raft.git
    GIT_TAG           6455e05b3889db2b495cf3189b33c2b07bfbebf2

    SOURCE_SUBDIR     raft
  )

  FetchContent_GetProperties(raft)
  if(NOT raft_POPULATED)
     message("populating raft")
     FetchContent_Populate(raft)
     # We are not using any raft CMake targets, so no need to call `add_subdirectory()`.
  endif()

  set(RAFT_DIR "${raft_SOURCE_DIR}")
endif(DEFINED ENV{RAFT_PATH})

###################################################################################################
# - External Projects -----------------------------------------------------------------------------

# https://cmake.org/cmake/help/v3.0/module/ExternalProject.html

# FIXME: gunrock is still using ExternalProject instead of
# FetchContent. Consider migrating to FetchContent soon (this may require
# updates to the gunrock cmake files to support this).

include(ExternalProject)

# - GUNROCK
set(GUNROCK_DIR ${CMAKE_CURRENT_BINARY_DIR}/gunrock CACHE STRING "Path to gunrock repo")
set(GUNROCK_INCLUDE_DIR ${GUNROCK_DIR}/src/gunrock_ext CACHE STRING "Path to gunrock includes")

ExternalProject_Add(gunrock_ext
  GIT_REPOSITORY    https://github.com/gunrock/gunrock.git
  GIT_TAG           v1.2
  PREFIX            ${GUNROCK_DIR}
  CMAKE_ARGS        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DGUNROCK_BUILD_SHARED_LIBS=OFF
                    -DGUNROCK_BUILD_TESTS=OFF
                    -DCUDA_AUTODETECT_GENCODE=OFF
                    -DGUNROCK_GENCODE_SM60=${GUNROCK_GENCODE_SM60}
                    -DGUNROCK_GENCODE_SM61=${GUNROCK_GENCODE_SM61}
                    -DGUNROCK_GENCODE_SM70=${GUNROCK_GENCODE_SM70}
                    -DGUNROCK_GENCODE_SM72=${GUNROCK_GENCODE_SM72}
                    -DGUNROCK_GENCODE_SM75=${GUNROCK_GENCODE_SM75}
                    -DGUNROCK_GENCODE_SM80=${GUNROCK_GENCODE_SM80}
                    ${GUNROCK_GENCODE}
  BUILD_BYPRODUCTS  ${GUNROCK_DIR}/src/gunrock_ext-build/lib/libgunrock.a
  INSTALL_COMMAND   ""
)

add_library(gunrock STATIC IMPORTED)
add_dependencies(gunrock gunrock_ext)
set_property(TARGET gunrock PROPERTY IMPORTED_LOCATION ${GUNROCK_DIR}/src/gunrock_ext-build/lib/libgunrock.a)

# - FAISS
# FIXME: The commit currently being fetched from faiss is using autotools which
# is more convenient to build with ExternalProjectAdd.
# Consider migrating to FetchContent once the tagged commit is changed.

if(BUILD_STATIC_FAISS)
  set(FAISS_DIR ${CMAKE_CURRENT_BINARY_DIR}/faiss CACHE STRING
    "Path to FAISS source directory")
  ExternalProject_Add(faiss
    GIT_REPOSITORY    https://github.com/facebookresearch/faiss.git
    GIT_TAG           7c2d2388a492d65fdda934c7e74ae87acaeed066
    CONFIGURE_COMMAND LIBS=-pthread
                      CPPFLAGS=-w
                      LDFLAGS=-L${CMAKE_INSTALL_PREFIX}/lib
                        cmake -B build .
                        -DCMAKE_BUILD_TYPE=Release
                        -DBUILD_TESTING=OFF
                        -DFAISS_ENABLE_PYTHON=OFF
                        -DBUILD_SHARED_LIBS=OFF
                        -DFAISS_ENABLE_GPU=ON
                        -DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR}
                        -DCUDA_ARCHITECTURES=${FAISS_GPU_ARCHS}
                        -DBLAS_LIBRARIES=${BLAS_LIBRARIES}
    PREFIX            ${FAISS_DIR}
    BUILD_COMMAND     make -C build -j${PARALLEL_LEVEL} VERBOSE=1
    BUILD_BYPRODUCTS  ${FAISS_DIR}/src/faiss/build/faiss/libfaiss.a
    BUILD_ALWAYS      1
    INSTALL_COMMAND   ""
    UPDATE_COMMAND    ""
    BUILD_IN_SOURCE   1)

  ExternalProject_Get_Property(faiss install_dir)
  add_library(FAISS::FAISS STATIC IMPORTED)
  set_property(TARGET FAISS::FAISS PROPERTY
    IMPORTED_LOCATION ${FAISS_DIR}/src/faiss/build/faiss/libfaiss.a)
  set(FAISS_INCLUDE_DIRS "${FAISS_DIR}/src/faiss")
else()
  set(FAISS_INSTALL_DIR ENV{FAISS_ROOT})
  find_package(FAISS REQUIRED)
endif(BUILD_STATIC_FAISS)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(cugraph SHARED
    src/utilities/spmv_1D.cu
    src/utilities/cython.cu
    src/structure/graph.cu
    src/linear_assignment/hungarian.cu
    src/link_analysis/gunrock_hits.cpp
    src/traversal/bfs.cu
    src/traversal/sssp.cu
    src/traversal/tsp.cu
    src/link_prediction/jaccard.cu
    src/link_prediction/overlap.cu
    src/layout/force_atlas2.cu
    src/converters/renumber.cu
    src/converters/COOtoCSR.cu
    src/community/spectral_clustering.cu
    src/community/louvain.cu
    src/community/leiden.cu
    src/community/ktruss.cu
    src/community/ecg.cu
    src/community/triangles_counting.cu
    src/community/extract_subgraph_by_vertex.cu
    src/community/egonet.cu
    src/sampling/random_walks.cu
    src/cores/core_number.cu
    src/traversal/two_hop_neighbors.cu
    src/components/connectivity.cu
    src/centrality/katz_centrality.cu
    src/centrality/betweenness_centrality.cu
    src/experimental/generate_rmat_edgelist.cu
    src/experimental/graph.cu
    src/experimental/graph_view.cu
    src/experimental/coarsen_graph.cu
    src/experimental/renumber_edgelist.cu
    src/experimental/relabel.cu
    src/experimental/induced_subgraph.cu
    src/experimental/bfs.cu
    src/experimental/sssp.cu
    src/experimental/pagerank.cu
    src/experimental/katz_centrality.cu
    src/tree/mst.cu
)

target_link_directories(cugraph
    PRIVATE
    # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the
    # link directories for nvcc.
    "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

#
# NOTE:  This dependency will force the building of cugraph to
#        wait until after cugunrock is constructed.
#
add_dependencies(cugraph gunrock_ext)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------
target_include_directories(cugraph
    PRIVATE
    "${THRUST_INCLUDE_DIR}"
    "${CUCO_INCLUDE_DIR}"
    "${LIBCUDACXX_INCLUDE_DIR}"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${Boost_INCLUDE_DIRS}"
    "${RMM_INCLUDE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
    "${CUHORNET_INCLUDE_DIR}/hornet/include"
    "${CUHORNET_INCLUDE_DIR}/hornetsnest/include"
    "${CUHORNET_INCLUDE_DIR}/xlib/include"
    "${CUHORNET_INCLUDE_DIR}/primitives"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${GUNROCK_INCLUDE_DIR}"
    "${NCCL_INCLUDE_DIRS}"
    "${RAFT_DIR}/cpp/include"
    PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(cugraph PRIVATE
  gunrock cublas cusparse curand cusolver cudart cuda FAISS::FAISS ${NCCL_LIBRARIES})

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
###   environment. If a full file path is provided (e.g ${NCCL_LIBRARIES}), cmake
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
###   path included in BUILD_RPATH when ${NCCL_LIBRARIES} are added; this one can
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

# CUDA_ARCHITECTURES=OFF implies cmake will not pass arch flags to the
# compiler. CUDA_ARCHITECTURES must be set to a non-empty value to prevent cmake
# warnings about policy CMP0104. With this setting, arch flags must be manually
# set! ("evaluate_gpu_archs(GPU_ARCHS)" is the current mechanism used in
# cpp/CMakeLists.txt for setting arch options).  Run "cmake --help-policy
# CMP0104" for policy details.
# NOTE: the CUDA_ARCHITECTURES=OFF setting may be removed after migrating to the
# findcudatoolkit features in cmake 3.17+
set_target_properties(cugraph PROPERTIES
    CUDA_ARCHITECTURES OFF)

###################################################################################################
# - generate tests --------------------------------------------------------------------------------

if(BUILD_TESTS)
    if(GTEST_FOUND)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS cugraph LIBRARY
    DESTINATION lib)

install(DIRECTORY include/
    DESTINATION include/cugraph)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph/version_config.hpp
        DESTINATION include/cugraph)

install(DIRECTORY ${RAFT_DIR}/cpp/include/raft/
    DESTINATION include/cugraph/raft)
###################################################################################################
# - make documentation ----------------------------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cugraph

# doc targets for cuGraph
add_custom_command(OUTPUT CUGRAPH_DOXYGEN
                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                   COMMAND doxygen Doxyfile
                   VERBATIM)

add_custom_target(docs_cugraph DEPENDS CUGRAPH_DOXYGEN)
