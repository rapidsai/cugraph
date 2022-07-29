#=============================================================================
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#
#=============================================================================

###################################################################################################
# - common test utils -----------------------------------------------------------------------------

add_library(cugraphtestutil STATIC
            utilities/matrix_market_file_utilities.cu
            utilities/thrust_wrapper.cu
            utilities/misc_utilities.cpp
            components/wcc_graphs.cu
            ../../thirdparty/mmio/mmio.c)

target_compile_options(cugraphtestutil
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                   "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>>"
)

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphtestutil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUGRAPH_SOURCE_DIR}/src"
)

target_link_libraries(cugraphtestutil
    PUBLIC
        cugraph::cugraph
        NCCL::NCCL
    PRIVATE
        cuco::cuco
)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraphtestutil
            cugraph::cugraph
            GTest::gtest
            GTest::gtest_main
            NCCL::NCCL
    )

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})

    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")

    install(
        TARGETS ${CMAKE_TEST_NAME}
        COMPONENT testing
        DESTINATION bin/gtests/libcugraph
        EXCLUDE_FROM_ALL)
endfunction()

function(ConfigureTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        cugraphmgtestutil
        cugraphtestutil
        cugraph::cugraph
        GTest::gtest
        GTest::gtest_main
        NCCL::NCCL
        MPI::MPI_CXX
    )

    add_test(NAME ${CMAKE_TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE}
             "--noprefix"
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS})

    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")

        install(
            TARGETS ${CMAKE_TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/libcugraph_mg
            EXCLUDE_FROM_ALL)
endfunction()

function(ConfigureCTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraph::cugraph_c
            cugraph_c_testutil
            GTest::gtest
            GTest::gtest_main
    )

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})

    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")

        install(
            TARGETS ${CMAKE_TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/libcugraph_c
            EXCLUDE_FROM_ALL)
endfunction()

function(ConfigureCTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraph::cugraph_c
            cugraph_c_testutil
            GTest::gtest
            GTest::gtest_main
            NCCL::NCCL
            MPI::MPI_CXX
    )

    add_test(NAME ${CMAKE_TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE}
             "--noprefix"
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS})

    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../../../lib")

        install(
            TARGETS ${CMAKE_TEST_NAME}
            COMPONENT testing
            DESTINATION bin/gtests/libcugraph_c
            EXCLUDE_FROM_ALL)
endfunction()

if(NOT USE_CUGRAPH_OPS)
    list(APPEND CMAKE_C_FLAGS -DNO_CUGRAPH_OPS)
    list(APPEND CMAKE_CXX_FLAGS -DNO_CUGRAPH_OPS)
    list(APPEND CMAKE_CUDA_FLAGS -DNO_CUGRAPH_OPS)
endif()

###################################################################################################
# - set rapids dataset path ----------------------------------------------------------------------

if(RAPIDS_DATASET_ROOT_DIR)
    message(STATUS "setting default RAPIDS_DATASET_ROOT_DIR to: ${RAPIDS_DATASET_ROOT_DIR}")
    string(CONCAT CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
endif(RAPIDS_DATASET_ROOT_DIR)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - graph generator tests -------------------------------------------------------------------------
ConfigureTest(GRAPH_GENERATORS_TEST generators/generators_test.cpp)

###################################################################################################
# - erdos renyi graph generator tests -------------------------------------------------------------
ConfigureTest(ERDOS_RENYI_GENERATOR_TEST generators/erdos_renyi_test.cpp)

###################################################################################################
# - betweenness centrality tests ------------------------------------------------------------------
ConfigureTest(LEGACY_BETWEENNESS_TEST centrality/legacy/betweenness_centrality_test.cu)
ConfigureTest(LEGACY_EDGE_BETWEENNESS_TEST centrality/legacy/edge_betweenness_centrality_test.cu)

###################################################################################################
# - BFS tests -------------------------------------------------------------------------------------
ConfigureTest(LEGACY_BFS_TEST traversal/legacy/bfs_test.cu)

###################################################################################################
# - LOUVAIN tests ---------------------------------------------------------------------------------
ConfigureTest(LOUVAIN_TEST community/louvain_test.cpp)

###################################################################################################
# - LEIDEN tests ---------------------------------------------------------------------------------
ConfigureTest(LEIDEN_TEST community/leiden_test.cpp)

###################################################################################################
# - ECG tests ---------------------------------------------------------------------------------
ConfigureTest(ECG_TEST community/ecg_test.cpp)

###################################################################################################
# - Balanced cut clustering tests -----------------------------------------------------------------
ConfigureTest(BALANCED_TEST community/balanced_edge_test.cpp)

###################################################################################################
# - TRIANGLE tests --------------------------------------------------------------------------------
ConfigureTest(TRIANGLE_TEST community/triangle_test.cu)

###################################################################################################
# - EGO tests --------------------------------------------------------------------------------
ConfigureTest(EGO_TEST community/egonet_test.cu)

###################################################################################################
# - FORCE ATLAS 2  tests --------------------------------------------------------------------------
ConfigureTest(FA2_TEST layout/force_atlas2_test.cu)

###################################################################################################
# - CONNECTED COMPONENTS  tests -------------------------------------------------------------------
ConfigureTest(CONNECT_TEST components/con_comp_test.cu)

###################################################################################################
# - STRONGLY CONNECTED COMPONENTS  tests ----------------------------------------------------------
ConfigureTest(SCC_TEST components/scc_test.cu)

###################################################################################################
# - WEAKLY CONNECTED COMPONENTS  tests ----------------------------------------------------------
ConfigureTest(WCC_TEST components/wcc_test.cpp)

###################################################################################################
#-Hungarian (Linear Assignment Problem)  tests ----------------------------------------------------
ConfigureTest(HUNGARIAN_TEST linear_assignment/hungarian_test.cu)

###################################################################################################
# - MST tests -------------------------------------------------------------------------------------
ConfigureTest(MST_TEST tree/mst_test.cu)

###################################################################################################
# - Stream tests ----------------------------------------------------------------------------------
ConfigureTest(STREAM_TEST structure/streams.cu)

###################################################################################################
# - R-mat graph generation tests ------------------------------------------------------------------
ConfigureTest(GENERATE_RMAT_TEST generators/generate_rmat_test.cpp)

###################################################################################################
# - Graph tests -----------------------------------------------------------------------------------
ConfigureTest(GRAPH_TEST structure/graph_test.cpp)

###################################################################################################
# - Symmetrize tests ------------------------------------------------------------------------------
ConfigureTest(SYMMETRIZE_TEST structure/symmetrize_test.cpp)

###################################################################################################
# - Transpose tests ------------------------------------------------------------------------------
ConfigureTest(TRANSPOSE_TEST structure/transpose_test.cpp)

###################################################################################################
# - Transpose Storage tests -----------------------------------------------------------------------
ConfigureTest(TRANSPOSE_STORAGE_TEST structure/transpose_storage_test.cpp)

###################################################################################################
# - Weight-sum tests ------------------------------------------------------------------------------
ConfigureTest(WEIGHT_SUM_TEST structure/weight_sum_test.cpp)

###################################################################################################
# - Degree tests ----------------------------------------------------------------------------------
ConfigureTest(DEGREE_TEST structure/degree_test.cpp)

###################################################################################################
# - Count self-loops and multi-edges tests --------------------------------------------------------
ConfigureTest(COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST
              "structure/count_self_loops_and_multi_edges_test.cpp")

###################################################################################################
# - Coarsening tests ------------------------------------------------------------------------------
ConfigureTest(COARSEN_GRAPH_TEST structure/coarsen_graph_test.cpp)

###################################################################################################
# - Induced subgraph tests ------------------------------------------------------------------------
ConfigureTest(INDUCED_SUBGRAPH_TEST community/induced_subgraph_test.cpp)

###################################################################################################
# - BFS tests -------------------------------------------------------------------------------------
ConfigureTest(BFS_TEST traversal/bfs_test.cpp)

###################################################################################################
# - Extract BFS Paths tests ------------------------------------------------------------------------
ConfigureTest(EXTRACT_BFS_PATHS_TEST
              traversal/extract_bfs_paths_test.cu)

###################################################################################################
# - Multi-source BFS tests -----------------------------------------------------------------------
ConfigureTest(MSBFS_TEST traversal/ms_bfs_test.cu)

###################################################################################################
# - SSSP tests ------------------------------------------------------------------------------------
ConfigureTest(SSSP_TEST traversal/sssp_test.cpp)

###################################################################################################
# - HITS tests ------------------------------------------------------------------------------------
ConfigureTest(HITS_TEST link_analysis/hits_test.cpp)

###################################################################################################
# - PAGERANK tests --------------------------------------------------------------------------------
ConfigureTest(PAGERANK_TEST link_analysis/pagerank_test.cpp)

###################################################################################################
# - KATZ_CENTRALITY tests -------------------------------------------------------------------------
ConfigureTest(KATZ_CENTRALITY_TEST centrality/katz_centrality_test.cpp)

###################################################################################################
# - EIGENVECTOR_CENTRALITY tests -------------------------------------------------------------------------
ConfigureTest(EIGENVECTOR_CENTRALITY_TEST centrality/eigenvector_centrality_test.cpp)

###################################################################################################
# - WEAKLY CONNECTED COMPONENTS tests -------------------------------------------------------------
ConfigureTest(WEAKLY_CONNECTED_COMPONENTS_TEST components/weakly_connected_components_test.cpp)

###################################################################################################
# - Legacy RANDOM_WALKS tests ----------------------------------------------------------------------------
ConfigureTest(LEGACY_RANDOM_WALKS_TEST sampling/random_walks_test.cu)

###################################################################################################
# - RANDOM_WALKS tests ----------------------------------------------------------------------------
#  FIXME: Rename to random_walks_test.cu once the legacy implementation is deleted
ConfigureTest(RANDOM_WALKS_TEST sampling/sg_random_walks_test.cu)

###################################################################################################
ConfigureTest(RANDOM_WALKS_LOW_LEVEL_TEST sampling/rw_low_level_test.cu)

###########################################################################################
# - MG NBR SAMPLING tests -----------------------------------------------------------------
ConfigureTest(UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/sg_uniform_neighbor_sampling.cu)
target_link_libraries(UNIFORM_NEIGHBOR_SAMPLING_TEST PRIVATE cuco::cuco)
        
###################################################################################################
# FIXME: since this is technically not a test, consider refactoring the the
# ConfigureTest function to share common code with a new ConfigureBenchmark
# function (which would not link gtest, etc.)
ConfigureTest(RANDOM_WALKS_PROFILING sampling/random_walks_profiling.cu)

###################################################################################################
# - Serialization tests ---------------------------------------------------------------------------
ConfigureTest(SERIALIZATION_TEST serialization/un_serialize_test.cpp)

###################################################################################################
# - Renumber tests --------------------------------------------------------------------------------
set(RENUMBERING_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/structure/renumbering_test.cpp")

ConfigureTest(RENUMBERING_TEST "${RENUMBERING_TEST_SRCS}")

###################################################################################################
# - Core Number tests -----------------------------------------------------------------------------
ConfigureTest(CORE_NUMBER_TEST cores/core_number_test.cpp)

###################################################################################################
# - Triangle Count tests --------------------------------------------------------------------------
ConfigureTest(TRIANGLE_COUNT_TEST community/triangle_count_test.cpp)

###################################################################################################
# - MG tests --------------------------------------------------------------------------------------

if(BUILD_CUGRAPH_MG_TESTS)
    ###############################################################################################
    # - find MPI - only enabled if MG tests are to be built
    find_package(MPI REQUIRED COMPONENTS CXX)

    add_library(cugraphmgtestutil STATIC
                utilities/device_comm_wrapper.cu
                utilities/mg_utilities.cpp)

    set_property(TARGET cugraphmgtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_include_directories(cugraphmgtestutil
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(cugraphmgtestutil
        PUBLIC
            cugraph::cugraph
            NCCL::NCCL
            MPI::MPI_CXX
    )

    # Set the GPU count to 1.  If the caller wants to execute MG tests using
    # more than 1, override from the command line using -DGPU_COUNT=<gpucount>
    if (NOT DEFINED GPU_COUNT)
      set(GPU_COUNT "1")
    endif()

    ###########################################################################################
    # - MG SYMMETRIZE tests -------------------------------------------------------------------
    ConfigureTestMG(MG_SYMMETRIZE_TEST structure/mg_symmetrize_test.cpp)

    ###########################################################################################
    # - MG Transpose tests --------------------------------------------------------------------
    ConfigureTestMG(MG_TRANSPOSE_TEST structure/mg_transpose_test.cpp)

    ###########################################################################################
    # - MG Transpose Storage tests ------------------------------------------------------------
    ConfigureTestMG(MG_TRANSPOSE_STORAGE_TEST structure/mg_transpose_storage_test.cpp)

    ###########################################################################################
    # - MG Count self-loops and multi-edges tests ---------------------------------------------
    ConfigureTestMG(MG_COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST
          "structure/mg_count_self_loops_and_multi_edges_test.cpp")

    ###########################################################################################
    # - MG PAGERANK tests ---------------------------------------------------------------------
    ConfigureTestMG(MG_PAGERANK_TEST link_analysis/mg_pagerank_test.cpp)

    ###########################################################################################
    # - MG HITS tests -------------------------------------------------------------------------
    ConfigureTestMG(MG_HITS_TEST link_analysis/mg_hits_test.cpp)

    ###########################################################################################
    # - MG KATZ CENTRALITY tests --------------------------------------------------------------
    ConfigureTestMG(MG_KATZ_CENTRALITY_TEST centrality/mg_katz_centrality_test.cpp)

    ###########################################################################################
    # - MG EIGENVECTOR CENTRALITY tests --------------------------------------------------------------
    ConfigureTestMG(MG_EIGENVECTOR_CENTRALITY_TEST centrality/mg_eigenvector_centrality_test.cpp)

    ###########################################################################################
    # - MG BFS tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_BFS_TEST traversal/mg_bfs_test.cpp)

    ###########################################################################################
    # - Extract BFS Paths tests ---------------------------------------------------------------
    ConfigureTestMG(MG_EXTRACT_BFS_PATHS_TEST
                    traversal/mg_extract_bfs_paths_test.cu)

    ###########################################################################################
    # - MG SSSP tests -------------------------------------------------------------------------
    ConfigureTestMG(MG_SSSP_TEST traversal/mg_sssp_test.cpp)

    ###########################################################################################
    # - MG LOUVAIN tests ----------------------------------------------------------------------
    ConfigureTestMG(MG_LOUVAIN_TEST
        community/mg_louvain_helper.cu
        community/mg_louvain_test.cpp)

    ###########################################################################################
    # - MG WEAKLY CONNECTED COMPONENTS tests --------------------------------------------------
    ConfigureTestMG(MG_WEAKLY_CONNECTED_COMPONENTS_TEST
                    components/mg_weakly_connected_components_test.cpp)

    ###########################################################################################
    # - MG GRAPH BROADCAST tests --------------------------------------------------------------
    ConfigureTestMG(MG_GRAPH_BROADCAST_TEST bcast/mg_graph_bcast.cpp)

    ###########################################################################################
    # - MG Core Number tests ------------------------------------------------------------------
    ConfigureTestMG(MG_CORE_NUMBER_TEST cores/mg_core_number_test.cpp)

    ###########################################################################################
    # - MG TRIANGLE COUNT tests ---------------------------------------------------------------
    ConfigureTestMG(MG_TRIANGLE_COUNT_TEST community/mg_triangle_count_test.cpp)

    ###########################################################################################
    # - MG PRIMS COUNT_IF_V tests -------------------------------------------------------------
    ConfigureTestMG(MG_COUNT_IF_V_TEST prims/mg_count_if_v.cu)
    target_link_libraries(MG_COUNT_IF_V_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_V_FRONTIER_OUTGOING_E_BY_DST tests --------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_V_FRONTIER_OUTGOING_E_BY_DST_TEST
                    prims/mg_transform_reduce_v_frontier_outgoing_e_by_dst.cu)
    target_link_libraries(MG_TRANSFORM_REDUCE_V_FRONTIER_OUTGOING_E_BY_DST_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS REDUCE_V tests ---------------------------------------------------------------
    ConfigureTestMG(MG_REDUCE_V_TEST prims/mg_reduce_v.cu)
    target_link_libraries(MG_REDUCE_V_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_V tests -----------------------------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_V_TEST prims/mg_transform_reduce_v.cu)
    target_link_libraries(MG_TRANSFORM_REDUCE_V_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_E tests -----------------------------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_E_TEST prims/mg_transform_reduce_e.cu)
    target_link_libraries(MG_TRANSFORM_REDUCE_E_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS COUNT_IF_E tests -------------------------------------------------------------
    ConfigureTestMG(MG_COUNT_IF_E_TEST prims/mg_count_if_e.cu)
    target_link_libraries(MG_COUNT_IF_E_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS PER_V_TRANSFORM_REDUCE_INCOMING_OUTGOING_E tests -----------------------------
    ConfigureTestMG(MG_PER_V_TRANSFORM_REDUCE_INCOMING_OUTGOING_E_TEST
      prims/mg_per_v_transform_reduce_incoming_outgoing_e.cu)
    target_link_libraries(MG_PER_V_TRANSFORM_REDUCE_INCOMING_OUTGOING_E_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG PRIMS EXTRACT_IF_E tests -----------------------------------------------------------
    ConfigureTestMG(MG_EXTRACT_IF_E_TEST prims/mg_extract_if_e.cu)
    target_link_libraries(MG_EXTRACT_IF_E_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG GATHER_UTILS tests -----------------------------------------------------------------
    ConfigureTestMG(MG_GATHER_UTILS_TEST sampling/detail/mg_gather_utils.cu)
    target_link_libraries(MG_GATHER_UTILS_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG GATHER_ONE_HOP tests ---------------------------------------------------------------
    ConfigureTestMG(MG_GATHER_ONE_HOP_TEST sampling/detail/mg_gather_one_hop.cu)
    target_link_libraries(MG_GATHER_ONE_HOP_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - MG NBR SAMPLING tests -----------------------------------------------------------------
    ConfigureTestMG(MG_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/mg_uniform_neighbor_sampling.cu)
    target_link_libraries(MG_UNIFORM_NEIGHBOR_SAMPLING_TEST PRIVATE cuco::cuco)

    ###########################################################################################
    # - RANDOM_WALKS tests --------------------------------------------------------------------
    ConfigureTestMG(MG_RANDOM_WALKS_TEST sampling/mg_random_walks_test.cu)

    ###########################################################################################
    # - MG C API tests ------------------------------------------------------------------------
    ConfigureCTestMG(MG_CAPI_CREATE_GRAPH c_api/mg_create_graph_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_PAGERANK c_api/mg_pagerank_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_BFS c_api/mg_bfs_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_SSSP c_api/mg_sssp_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_WEAKLY_CONNECTED_COMPONENTS c_api/mg_weakly_connected_components_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_STRONGLY_CONNECTED_COMPONENTS c_api/mg_strongly_connected_components_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_KATZ c_api/mg_katz_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_EIGENVECTOR_CENTRALITY c_api/mg_eigenvector_centrality_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_HITS c_api/mg_hits_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_UNIFORM_NEIGHBOR_SAMPLE c_api/mg_uniform_neighbor_sample_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_RANDOM_WALKS c_api/mg_random_walks_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_TRIANGLE_COUNT c_api/mg_triangle_count_test.c c_api/mg_test_utils.cpp)
    ConfigureCTestMG(MG_CAPI_LOUVAIN c_api/mg_louvain_test.c c_api/mg_test_utils.cpp)
	ConfigureCTestMG(MG_CAPI_CORE_NUMBER c_api/mg_core_number_test.c c_api/mg_test_utils.cpp)
endif()

###################################################################################################
# - C API tests -----------------------------------------------------------------------------------

###################################################################################################
# - common C API test utils -----------------------------------------------------------------------------

add_library(cugraph_c_testutil STATIC
            c_api/test_utils.cpp)

target_compile_options(cugraph_c_testutil
            PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                   "$<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>>"
)

set_property(TARGET cugraph_c_testutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraph_c_testutil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUGRAPH_SOURCE_DIR}/src"
)

target_link_libraries(cugraph_c_testutil
    PUBLIC
        cugraph
)


# FIXME: Disabling these tests, they test an older version of the API which
#        will be deleted in this release.  Leaving them here to help me remember :-)
#ConfigureCTest(CAPI_CREATE_SG_GRAPH_ENVELOPE_TEST c_api/create_sg_graph_envelope_test.c)
#ConfigureCTest(CAPI_RANDOM_WALKS_TEST c_api/random_walks_test.c)

ConfigureCTest(CAPI_CREATE_GRAPH_TEST c_api/create_graph_test.c)
ConfigureCTest(CAPI_PAGERANK_TEST c_api/pagerank_test.c)
ConfigureCTest(CAPI_KATZ_TEST c_api/katz_test.c)
ConfigureCTest(CAPI_EIGENVECTOR_CENTRALITY_TEST c_api/eigenvector_centrality_test.c)
ConfigureCTest(CAPI_HITS_TEST c_api/hits_test.c)
ConfigureCTest(CAPI_BFS_TEST c_api/bfs_test.c)
ConfigureCTest(CAPI_SSSP_TEST c_api/sssp_test.c)
ConfigureCTest(CAPI_EXTRACT_PATHS_TEST c_api/extract_paths_test.c)
ConfigureCTest(CAPI_NODE2VEC_TEST c_api/node2vec_test.c)
ConfigureCTest(CAPI_WEAKLY_CONNECTED_COMPONENTS c_api/weakly_connected_components_test.c)
ConfigureCTest(CAPI_STRONGLY_CONNECTED_COMPONENTS c_api/strongly_connected_components_test.c)
ConfigureCTest(CAPI_UNIFORM_NEIGHBOR_SAMPLE c_api/uniform_neighbor_sample_test.c)
ConfigureCTest(CAPI_RANDOM_WALKS c_api/sg_random_walks_test.c)
ConfigureCTest(CAPI_TRIANGLE_COUNT c_api/triangle_count_test.c)
ConfigureCTest(CAPI_LOUVAIN c_api/louvain_test.c)
ConfigureCTest(CAPI_CORE_NUMBER c_api/core_number_test.c)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
