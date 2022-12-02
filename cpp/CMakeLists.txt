#=============================================================================
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

include(../fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH)

project(CUGRAPH VERSION 23.02.00 LANGUAGES C CXX CUDA)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()

# Remove the following archs from CMAKE_CUDA_ARCHITECTURES that
# cuhornet currently doesn't support
#
# >= 90
set(supported_archs "60" "62" "70" "72" "75" "80" "86" "89" "90")
foreach( arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE "-real" "" arch ${arch})
    if( arch IN_LIST supported_archs )
        list(APPEND usable_arch_values ${arch})
    endif()
endforeach()
# Make sure everything but the 'newest' arch
# is marked as `-real` so we only generate PTX for
# arch > 86
list(POP_BACK usable_arch_values latest_arch)
list(TRANSFORM usable_arch_values APPEND "-real")
if (usable_arch_values)
    list(APPEND usable_arch_values ${latest_arch})
else()
    list(APPEND usable_arch_values ${latest_arch}-real)
endif()

set(CMAKE_CUDA_ARCHITECTURES ${usable_arch_values})

# Write the version header
rapids_cmake_write_version_file(include/cugraph/version_config.hpp)
rapids_cmake_write_version_file(include/cugraph_c/version_config.hpp)

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
option(USE_CUGRAPH_OPS "Enable all functions that call cugraph-ops" ON)
option(USE_RAFT_STATIC "Build raft as a static library" OFF)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUGRAPH_USE_CUGRAPH_OPS_STATIC "Build and statically link the cugraph-ops library" OFF)
option(CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL "Exclude cugraph-ops targets from cuGraph's 'all' target" OFF)
option(ALLOW_CLONE_CUGRAPH_OPS "Whether to attempt to clone cugraph-ops when a local version is not available" OFF)

message(VERBOSE "CUGRAPH: CUDA_STATIC_RUNTIME=${CUDA_STATIC_RUNTIME}")
message(VERBOSE "CUGRAPH: CUGRAPH_USE_CUGRAPH_OPS_STATIC=${CUGRAPH_USE_CUGRAPH_OPS_STATIC}")
message(VERBOSE "CUGRAPH: CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL=${CUGRAPH_EXCLUDE_CUGRAPH_OPS_FROM_ALL}")

################################################################################
# - compiler options -----------------------------------------------------------

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})

rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-exports
    INSTALL_EXPORT_SET  cugraph-exports
    )

set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Werror=cross-execution-space-call -Wno-deprecated-declarations -Xptxas=--disable-warnings)
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

if(NOT USE_CUGRAPH_OPS)
    message(STATUS "Disabling functions that reference cugraph-ops")
    list(APPEND CUGRAPH_CXX_FLAGS -DNO_CUGRAPH_OPS)
    list(APPEND CUGRAPH_CUDA_FLAGS -DNO_CUGRAPH_OPS)
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------

rapids_cpm_init()

###
# Linking to the `raft::raft` target implicitly links cugraph targets to the
# following public header-only  raft dependencies:
# * RMM
# * Thrust
# * GTest/GMock
#
# The CMakeLists.txt for each of these projects are properly configured
# to generate a build and install export-set, so reimplementing finding or
# fetching those targets in cuGraph is redundant (at best), and potentially
# error-prone if something about those targets change and our implementation
# lags behind.
###

# Putting this before raft to override RAFT from pulling them in.
include(cmake/thirdparty/get_libcudacxx.cmake)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)

include(cmake/thirdparty/get_raft.cmake)

if(USE_CUGRAPH_OPS)
  include(cmake/thirdparty/get_libcugraphops.cmake)
endif()

include(cmake/thirdparty/get_nccl.cmake)
include(cmake/thirdparty/get_cuhornet.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

################################################################################
# - libcugraph library target --------------------------------------------------

# NOTE: The most expensive compilations are listed first
#       since ninja will run them in parallel in this order,
#       which should give us a better parallel schedule.

set(CUGRAPH_SOURCES
    src/detail/shuffle_vertices.cu
    src/detail/shuffle_edgelist.cu
    src/detail/collect_local_vertex_values.cu
    src/detail/groupby_and_count.cu
    src/sampling/random_walks_mg.cu
    src/community/detail/common_methods_mg.cu
    src/community/detail/common_methods_sg.cu
    src/detail/utility_wrappers.cu
    src/structure/graph_view_mg.cu
    src/utilities/cython.cu
    src/utilities/path_retrieval.cu
    src/structure/legacy/graph.cu
    src/linear_assignment/legacy/hungarian.cu
    src/traversal/legacy/bfs.cu
    src/link_prediction/legacy/jaccard.cu
    src/link_prediction/legacy/overlap.cu
    src/link_prediction/jaccard_sg.cu
    src/link_prediction/sorensen_sg.cu
    src/link_prediction/overlap_sg.cu
    src/link_prediction/jaccard_mg.cu
    src/link_prediction/sorensen_mg.cu
    src/link_prediction/overlap_mg.cu
    src/layout/legacy/force_atlas2.cu
    src/converters/legacy/COOtoCSR.cu
    src/community/legacy/spectral_clustering.cu
    src/community/louvain_sg.cu
    src/community/louvain_mg.cu
    src/community/leiden_sg.cu
    src/community/legacy/louvain.cu
    src/community/legacy/leiden.cu
    src/community/legacy/ktruss.cu
    src/community/legacy/ecg.cu
    src/community/legacy/extract_subgraph_by_vertex.cu
    src/community/egonet_sg.cu
    src/community/egonet_mg.cu
    src/sampling/random_walks.cu
    src/sampling/random_walks_sg.cu
    src/sampling/detail/sampling_utils_mg.cu
    src/sampling/detail/sampling_utils_sg.cu
    src/sampling/uniform_neighbor_sampling_mg.cpp
    src/sampling/uniform_neighbor_sampling_sg.cpp
    src/cores/legacy/core_number.cu
    src/cores/core_number_sg.cu
    src/cores/core_number_mg.cu
    src/cores/k_core_sg.cu
    src/cores/k_core_mg.cu
    src/traversal/legacy/two_hop_neighbors.cu
    src/components/legacy/connectivity.cu
    src/centrality/legacy/betweenness_centrality.cu
    src/generators/generate_rmat_edgelist.cu
    src/generators/generator_tools.cu
    src/generators/simple_generators.cu
    src/generators/erdos_renyi_generator.cu
    src/structure/graph_sg.cu
    src/structure/graph_mg.cu
    src/structure/graph_view_sg.cu
    src/structure/decompress_to_edgelist_sg.cu
    src/structure/decompress_to_edgelist_mg.cu
    src/structure/symmetrize_graph_sg.cu
    src/structure/symmetrize_graph_mg.cu
    src/structure/transpose_graph_sg.cu
    src/structure/transpose_graph_mg.cu
    src/structure/transpose_graph_storage_sg.cu
    src/structure/transpose_graph_storage_mg.cu
    src/structure/coarsen_graph_sg.cu
    src/structure/coarsen_graph_mg.cu
    src/structure/graph_weight_utils_mg.cu
    src/structure/graph_weight_utils_sg.cu
    src/structure/renumber_edgelist_sg.cu
    src/structure/renumber_edgelist_mg.cu
    src/structure/renumber_utils_sg.cu
    src/structure/renumber_utils_mg.cu
    src/structure/relabel_sg.cu
    src/structure/relabel_mg.cu
    src/structure/induced_subgraph_sg.cu
    src/structure/induced_subgraph_mg.cu
    src/traversal/extract_bfs_paths_sg.cu
    src/traversal/extract_bfs_paths_mg.cu
    src/traversal/bfs_sg.cu
    src/traversal/bfs_mg.cu
    src/traversal/sssp_sg.cu
    src/traversal/sssp_mg.cu
    src/link_analysis/hits_sg.cu
    src/link_analysis/hits_mg.cu
    src/link_analysis/pagerank_sg.cu
    src/link_analysis/pagerank_mg.cu
    src/centrality/katz_centrality_sg.cu
    src/centrality/katz_centrality_mg.cu
    src/centrality/eigenvector_centrality_sg.cu
    src/centrality/eigenvector_centrality_mg.cu
    src/centrality/betweenness_centrality_sg.cu
    src/centrality/betweenness_centrality_mg.cu
    src/tree/legacy/mst.cu
    src/components/weakly_connected_components_sg.cu
    src/components/weakly_connected_components_mg.cu
    src/structure/create_graph_from_edgelist_sg.cu
    src/structure/create_graph_from_edgelist_mg.cu
    src/structure/symmetrize_edgelist_sg.cu
    src/structure/symmetrize_edgelist_mg.cu
    src/community/triangle_count_sg.cu
    src/community/triangle_count_mg.cu
    src/traversal/k_hop_nbrs_sg.cu
    src/traversal/k_hop_nbrs_mg.cu
)

if(USE_CUGRAPH_OPS)
    list(APPEND CUGRAPH_SOURCES
        src/sampling/neighborhood.cu
    )
endif()

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

if(CUDA_STATIC_RUNTIME)
  get_target_property(_includes raft::raft INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(cugraph PUBLIC ${_includes})
  # Add CTK include paths because we make our CTK library links private below
  target_include_directories(cugraph SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

################################################################################
# - link libraries -------------------------------------------------------------
if (USE_CUGRAPH_OPS)
    target_link_libraries(cugraph
        PUBLIC
            rmm::rmm
            cugraph-ops::cugraph-ops++
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::raft>
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::distance>
        PRIVATE
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::raft>
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::distance>
            cuco::cuco
            cugraph::cuHornet
            NCCL::NCCL
    )
else()
    target_link_libraries(cugraph
        PUBLIC
            rmm::rmm
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::raft>
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::distance>
        PRIVATE
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::raft>
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::distance>
            cuco::cuco
            cugraph::cuHornet
            NCCL::NCCL
    )
endif()

################################################################################
# - C-API library --------------------------------------------------------------

add_library(cugraph_c
        src/c_api/resource_handle.cpp
        src/c_api/array.cpp
        src/c_api/error.cpp
        src/c_api/graph_sg.cpp
        src/c_api/graph_mg.cpp
        src/c_api/graph_functions.cpp
        src/c_api/pagerank.cpp
        src/c_api/katz.cpp
        src/c_api/centrality_result.cpp
        src/c_api/eigenvector_centrality.cpp
        src/c_api/betweenness_centrality.cpp
        src/c_api/core_number.cpp
        src/c_api/core_result.cpp
        src/c_api/extract_ego.cpp
        src/c_api/k_core.cpp
        src/c_api/induced_subgraph.cpp
        src/c_api/induced_subgraph_helper.cu
        src/c_api/graph_helper.cu
        src/c_api/induced_subgraph_result.cpp
        src/c_api/hits.cpp
        src/c_api/bfs.cpp
        src/c_api/sssp.cpp
        src/c_api/extract_paths.cpp
        src/c_api/random_walks.cpp
        src/c_api/similarity.cpp
        src/c_api/louvain.cpp
        src/c_api/triangle_count.cpp
        src/c_api/uniform_neighbor_sampling.cpp
        src/c_api/labeling_result.cpp
        src/c_api/weakly_connected_components.cpp
        src/c_api/strongly_connected_components.cpp
        )
add_library(cugraph::cugraph_c ALIAS cugraph_c)

# Currently presuming we aren't calling any CUDA kernels in cugraph_c

set_target_properties(cugraph_c
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

target_compile_options(cugraph_c
             PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

# Per-thread default stream option see https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html
# The per-thread default stream does not synchronize with other streams
target_compile_definitions(cugraph_c PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_link_options(cugraph_c PRIVATE "${CUGRAPH_BINARY_DIR}/fatbin.ld")

################################################################################
# - C-API include paths --------------------------------------------------------
target_include_directories(cugraph_c
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

if(CUDA_STATIC_RUNTIME)
  get_target_property(_includes raft::raft INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(cugraph_c PUBLIC ${_includes})
  # Add CTK include paths because we make our CTK library links private below
  target_include_directories(cugraph_c SYSTEM PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif()

################################################################################
# - C-API link libraries -------------------------------------------------------
target_link_libraries(cugraph_c
        PUBLIC
                CUDA::cublas
                CUDA::curand
                CUDA::cusolver
                CUDA::cusparse
                rmm::rmm
                $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::raft>
                $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::distance>
        PRIVATE
                cuco::cuco
                cugraph::cugraph
                $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::raft>
                $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::distance>
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

install(TARGETS cugraph_c
        DESTINATION ${lib_dir}
        EXPORT cugraph-exports)

install(DIRECTORY include/cugraph_c/
        DESTINATION include/cugraph_c)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cugraph_c/version_config.hpp
        DESTINATION include/cugraph_c)

################################################################################
# - install export -------------------------------------------------------------

set(doc_string
[=[
Provide targets for cuGraph.

cuGraph library is a collection of GPU accelerated graph algorithms that process data found in
[GPU DataFrames](https://github.com/rapidsai/cudf).

]=])

rapids_export(INSTALL cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph cugraph_c
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )

################################################################################
# - build export ---------------------------------------------------------------
rapids_export(BUILD cugraph
    EXPORT_SET cugraph-exports
    GLOBAL_TARGETS cugraph cugraph_c
    NAMESPACE cugraph::
    DOCUMENTATION doc_string
    )

################################################################################
# - make documentation ---------------------------------------------------------
# requires doxygen and graphviz to be installed
# from build directory, run make docs_cugraph

# doc targets for cugraph
find_package(Doxygen 1.8.11)
if(Doxygen_FOUND)
    add_custom_command(OUTPUT CUGRAPH_DOXYGEN
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/doxygen
                       COMMAND doxygen Doxyfile
                       VERBATIM)

    add_custom_target(docs_cugraph DEPENDS CUGRAPH_DOXYGEN)
endif()
