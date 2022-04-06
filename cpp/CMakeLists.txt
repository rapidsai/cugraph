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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)
file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.06/RAPIDS.cmake
    ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH)

project(CUGRAPH VERSION 22.06.00 LANGUAGES C CXX CUDA)

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
set(supported_archs "60" "62" "70" "72" "75" "80" "86")
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

################################################################################
# - compiler options -----------------------------------------------------------

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

################################################################################
# - find openmp ----------------------------------------------------------------

find_package(OpenMP)
if(OpenMP_FOUND)
    # find_package(OPenMP) does not automatically add OpenMP flags to CUDA
    list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif()

###################################################################################################
# - find CPM based dependencies  ------------------------------------------------------------------

rapids_cpm_init()

###
# Linking to the `raft::raft` target implicitly links cugraph targets to the
# following public header-only  raft dependencies:
# * RMM
# * Thrust
# * libcu++
# * GTest/GMock
# * cuCollections
#
# The CMakeLists.txt for each of these projects are properly configured
# to generate a build and install export-set, so reimplementing finding or
# fetching those targets in cuGraph is redundant (at best), and potentially
# error-prone if something about those targets change and our implementation
# lags behind.
###
include(cmake/thirdparty/get_raft.cmake)
include(cmake/thirdparty/get_libcugraphops.cmake)

include(cmake/thirdparty/get_nccl.cmake)
include(cmake/thirdparty/get_cuhornet.cmake)

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

################################################################################
# - libcugraph library target --------------------------------------------------

add_library(cugraph
    src/detail/utility_wrappers.cu
    src/detail/shuffle_wrappers.cu
    src/utilities/cython.cu
    src/utilities/path_retrieval.cu
    src/utilities/graph_bcast.cu
    src/structure/legacy/graph.cu
    src/linear_assignment/hungarian.cu
    src/traversal/legacy/bfs.cu
    src/traversal/legacy/sssp.cu
    src/link_prediction/jaccard.cu
    src/link_prediction/overlap.cu
    src/layout/force_atlas2.cu
    src/converters/COOtoCSR.cu
    src/community/legacy/spectral_clustering.cu
    src/community/louvain_sg.cu
    src/community/louvain_mg.cu
    src/community/legacy/louvain.cu
    src/community/legacy/leiden.cu
    src/community/legacy/ktruss.cu
    src/community/legacy/ecg.cu
    src/community/legacy/triangles_counting.cu
    src/community/legacy/extract_subgraph_by_vertex.cu
    src/community/legacy/egonet.cu
    src/sampling/neighborhood.cu
    src/sampling/random_walks.cu
    src/sampling/detail/gather_utils_impl.cu
    src/sampling/nbr_sampling_mg.cu
    src/cores/legacy/core_number.cu
    src/cores/core_number_sg.cu
    src/cores/core_number_mg.cu
    src/traversal/two_hop_neighbors.cu
    src/components/connectivity.cu
    src/centrality/legacy/katz_centrality.cu
    src/centrality/betweenness_centrality.cu
    src/generators/generate_rmat_edgelist.cu
    src/generators/generator_tools.cu
    src/generators/simple_generators.cu
    src/generators/erdos_renyi_generator.cu
    src/structure/graph_sg.cu
    src/structure/graph_mg.cu
    src/structure/graph_view_sg.cu
    src/structure/graph_view_mg.cu
    src/structure/coarsen_graph_sg.cu
    src/structure/coarsen_graph_mg.cu
    src/structure/renumber_edgelist_sg.cu
    src/structure/renumber_edgelist_mg.cu
    src/structure/renumber_utils_sg.cu
    src/structure/renumber_utils_mg.cu
    src/structure/relabel_sg.cu
    src/structure/relabel_mg.cu
    src/structure/induced_subgraph_sg.cu
    ## FIXME: Not currently supported
    ##src/structure/induced_subgraph_mg.cu
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
    src/serialization/serializer.cu
    src/tree/mst.cu
    src/components/weakly_connected_components_sg.cu
    src/components/weakly_connected_components_mg.cu
    src/structure/create_graph_from_edgelist_sg.cu
    src/structure/create_graph_from_edgelist_mg.cu
    src/structure/symmetrize_edgelist_sg.cu
    src/structure/symmetrize_edgelist_mg.cu
    src/visitors/graph_envelope.cpp
    src/visitors/visitors_factory.cpp
    src/visitors/bfs_visitor.cpp
    src/visitors/rw_visitor.cpp
    src/visitors/graph_make_visitor.cpp
)

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

################################################################################
# - link libraries -------------------------------------------------------------
target_link_libraries(cugraph
    PUBLIC
        cugraph-ops::cugraph-ops++
        raft::raft
    PRIVATE
        cugraph::cuHornet
        NCCL::NCCL
)

if(OpenMP_CXX_FOUND)
target_link_libraries(cugraph PRIVATE
################################################################################
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
endif()

################################################################################
# - C-API library --------------------------------------------------------------

add_library(cugraph_c
        src/c_api/cugraph_api.cpp
        src/c_api/resource_handle.cpp
        src/c_api/array.cpp
        src/c_api/error.cpp
        src/c_api/graph_sg.cpp
        src/c_api/graph_mg.cpp
        src/c_api/pagerank.cpp
        src/c_api/centrality_result.cpp
        src/c_api/eigenvector_centrality.cpp
        src/c_api/hits.cpp
        src/c_api/bfs.cpp
        src/c_api/sssp.cpp
        src/c_api/extract_paths.cpp
        src/c_api/random_walks.cpp
        src/c_api/uniform_neighbor_sampling.cpp
        )
add_library(cugraph::cugraph_c ALIAS cugraph_c)

# Currently presuming we aren't calling any CUDA kernels in cugraph_c

set_target_properties(cugraph_c
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               # set target compile options
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph_c
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
)

################################################################################
# - C-API include paths --------------------------------------------------------
target_include_directories(cugraph_c
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

################################################################################
# - C-API link libraries -------------------------------------------------------
target_link_libraries(cugraph_c
        PUBLIC
                CUDA::cublas
                CUDA::curand
                CUDA::cusolver
                CUDA::cusparse
        PRIVATE
                cugraph::cugraph
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
        DESTINATION ${lib_dir})

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
