#=============================================================================
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH)

project(CUGRAPH VERSION "${RAPIDS_VERSION}" LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

# Write the version header
rapids_cmake_write_version_file(include/cugraph/version_config.hpp)

################################################################################
# - build type -----------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

################################################################################
# - User Options  --------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuGraph shared libraries" ON)
option(BUILD_CUGRAPH_MG_TESTS "Build cuGraph multigpu algorithm tests" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)
option(BUILD_TESTS "Configure CMake to build tests" ON)
option(USE_RAFT_STATIC "Build raft as a static library" OFF)
option(CUGRAPH_COMPILE_RAFT_LIB "Compile the raft library instead of using it header-only" ON)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)

message(VERBOSE "CUGRAPH: CUDA_STATIC_RUNTIME=${CUDA_STATIC_RUNTIME}")

################################################################################
# - compiler options -----------------------------------------------------------

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-exports
    INSTALL_EXPORT_SET  cugraph-exports
    )

if (BUILD_CUGRAPH_MTMG_TESTS)
    if(NOT TARGET ucx::ucp)
        find_package(ucx REQUIRED)
    endif()

    if(NOT TARGET ucxx::ucxx)
        find_package(ucxx REQUIRED)
    endif()
endif()

set(CUGRAPH_C_FLAGS "")
set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
endif(CMAKE_COMPILER_IS_GNUCXX)

list(APPEND CUGRAPH_CUDA_FLAGS -target-dir sbsa-linux)

message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND CUGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif()

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------

rapids_cpm_init()

###
# Linking to the `raft::raft` target implicitly links cugraph targets to the
# following public header-only  raft dependencies:
# * CCCL
# * RMM
# * GTest/GMock
#
# The CMakeLists.txt for each of these projects are properly configured
# to generate a build and install export-set, so reimplementing finding or
# fetching those targets in cuGraph is redundant (at best), and potentially
# error-prone if something about those targets change and our implementation
# lags behind.
###

# Need CCCL, then rmm, then cuCollections, then RAFT.
# This ensures that libraries can be overridden for testing.
include(cmake/thirdparty/get_cccl.cmake)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(BUILD_EXPORT_SET cugraph-exports)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
include(cmake/thirdparty/get_raft.cmake)

if(BUILD_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

################################################################################
# - libcugraph library target --------------------------------------------------

# NOTE: The most expensive compilations are listed first
#       since ninja will run them in parallel in this order,
#       which should give us a better parallel schedule.

set(CUGRAPH_SOURCES
    src/utilities/shuffle_vertices_mg_v64_fp.cu
    src/utilities/shuffle_vertices_mg_v64_integral.cu
    src/utilities/shuffle_vertex_pairs_mg_v64_e64.cu
    src/detail/groupby_and_count_mg_v64_e64.cu
    src/detail/collect_comm_wrapper_mg_v64_e64.cu
    src/detail/utility_wrappers_32.cu
    src/detail/utility_wrappers_64.cu
    src/structure/graph_view_mg_v64_e64.cu
    src/structure/remove_self_loops_sg_v64_e64.cu
    src/structure/remove_multi_edges_sg_v64_e64.cu
    src/generators/generate_rmat_edgelist_sg_v64_e64.cu
    src/generators/generator_tools_sg_v64_e64.cu
    src/structure/graph_mg_v64_e64.cu
    src/structure/decompress_to_edgelist_mg_v64_e64.cu
    src/structure/symmetrize_graph_mg_v64_e64.cu
    src/structure/renumber_edgelist_mg_v64_e64.cu
    src/structure/renumber_utils_mg_v64_e64.cu
    src/traversal/bfs_mg_v64_e64.cu
    src/structure/create_graph_from_edgelist_mg_v64_e64.cu
    src/structure/create_graph_from_edgelist_mg_v64_e64_t32.cu
    src/structure/create_graph_from_edgelist_mg_v64_e64_t64.cu
    src/structure/symmetrize_edgelist_mg_v64_e64.cu
)

add_library(cugraph ${CUGRAPH_SOURCES})

set_target_properties(cugraph
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               CUDA_STANDARD                       17
               CUDA_STANDARD_REQUIRED              ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                    "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

file(WRITE "${CUGRAPH_BINARY_DIR}/fatbin.ld"
[=[
SECTIONS
{
  .nvFatBinSegment : { *(.nvFatBinSegment) }
  .nv_fatbin : { *(.nv_fatbin) }
}
]=])
target_link_options(cugraph PRIVATE "${CUGRAPH_BINARY_DIR}/fatbin.ld")

add_library(cugraph::cugraph ALIAS cugraph)

################################################################################
# - include paths --------------------------------------------------------------

target_include_directories(cugraph
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

set(COMPILED_RAFT_LIB )
if(CUGRAPH_COMPILE_RAFT_LIB)
  set(COMPILED_RAFT_LIB raft::compiled)
  if(USE_RAFT_STATIC)
    set(COMPILED_RAFT_LIB raft::compiled_static)
  endif()
endif()

################################################################################
# - link libraries -------------------------------------------------------------
target_link_libraries(cugraph
    PUBLIC
        rmm::rmm
        raft::raft
        $<BUILD_LOCAL_INTERFACE:CUDA::toolkit>
    PRIVATE
        ${COMPILED_RAFT_LIB}
        cuco::cuco
    )

################################################################################
# - generate tests -------------------------------------------------------------

if(BUILD_TESTS)
  include(CTest)
  add_subdirectory(tests)
endif()

################################################################################
# - install targets ------------------------------------------------------------
rapids_cmake_install_lib_dir( lib_dir )
include(CPack)

install(TARGETS cugraph
        DESTINATION ${lib_dir}
        EXPORT cugraph-exports)

install(DIRECTORY include/cugraph/
        DESTINATION include/cugraph)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph/version_config.hpp
        DESTINATION include/cugraph)

