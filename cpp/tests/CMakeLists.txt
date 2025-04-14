#=============================================================================
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
enable_testing()

include(rapids-test)
rapids_test_init()

###################################################################################################
# - set rapids dataset path ----------------------------------------------------------------------
if(RAPIDS_DATASET_ROOT_DIR)
    message(STATUS "setting default RAPIDS_DATASET_ROOT_DIR to: ${RAPIDS_DATASET_ROOT_DIR}")
    string(CONCAT CMAKE_C_FLAGS ${CMAKE_C_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
endif(RAPIDS_DATASET_ROOT_DIR)

###################################################################################################
# - common C++ test utils -----------------------------------------------------------------------------

add_library(cugraphtestutil STATIC
            utilities/matrix_market_file_utilities.cu
            utilities/csv_file_utilities.cu
            utilities/property_generator_utilities_sg.cu
            utilities/thrust_wrapper.cu
            utilities/misc_utilities.cpp
            utilities/conversion_utilities_sg.cu
            utilities/debug_utilities_sg.cpp
            utilities/validation_utilities.cu
            link_prediction/similarity_compare.cpp
            centrality/betweenness_centrality_validate.cu
            community/egonet_validate.cu
            cores/k_core_validate.cu
            structure/induced_subgraph_validate.cu
            sampling/random_walks_check_sg.cu
            sampling/detail/nbr_sampling_validate.cu
            sampling/detail/sampling_post_processing_validate.cu
            ../../thirdparty/mmio/mmio.c)

target_compile_options(cugraphtestutil
    PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphtestutil
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
    PRIVATE
        "${CUGRAPH_SOURCE_DIR}/src"
)

target_link_libraries(cugraphtestutil
    PUBLIC
        cugraph::cugraph
    PRIVATE
        GTest::gtest
)


###################################################################################################
# - common C API test utils -----------------------------------------------------------------------

add_library(cugraph_c_testutil STATIC
            c_api/test_utils.cpp)

target_compile_options(cugraph_c_testutil
    PUBLIC "$<$<COMPILE_LANGUAGE:C>:${CUGRAPH_C_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
           "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
)

target_include_directories(cugraph_c_testutil
    PRIVATE
        "${CUGRAPH_SOURCE_DIR}/src"
)

set_property(TARGET cugraph_c_testutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cugraph_c_testutil
    PUBLIC
        CUDA::toolkit
    PRIVATE
        cugraph::cugraph_c
        cugraph::cugraph
)

if(BUILD_CUGRAPH_MG_TESTS)
    include(../cmake/thirdparty/get_nccl.cmake)

    ###############################################################################################
    # - find MPI - only enabled if MG tests are to be built
    find_package(MPI REQUIRED COMPONENTS CXX)


    ###################################################################################################
    # - common C++ mg test utils -----------------------------------------------------------------------
    add_library(cugraphmgtestutil STATIC
                utilities/device_comm_wrapper.cu
                utilities/mg_utilities.cpp
                utilities/conversion_utilities_mg.cu
                utilities/property_generator_utilities_mg.cu
                utilities/debug_utilities_mg.cpp
                sampling/random_walks_check_mg.cu
                c_api/mg_test_utils.cpp
                )

    set_property(TARGET cugraphmgtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

    target_compile_options(cugraphmgtestutil
                PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
                       "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
    )

    target_include_directories(cugraphmgtestutil
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CUGRAPH_SOURCE_DIR}/src"
    )

    target_link_libraries(cugraphmgtestutil
        PRIVATE
            cugraph::cugraph
            NCCL::NCCL
            MPI::MPI_CXX
            GTest::gtest
    )
endif()

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    set(options)
    set(one_value GPUS PERCENT)
    set(multi_value)
    cmake_parse_arguments(_CUGRAPH_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
    if(NOT DEFINED _CUGRAPH_TEST_GPUS AND NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_GPUS 1)
        set(_CUGRAPH_TEST_PERCENT 25)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_GPUS)
        set(_CUGRAPH_TEST_GPUS 1)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_PERCENT 100)
    endif()

    add_executable(${CMAKE_TEST_NAME} ${_CUGRAPH_TEST_UNPARSED_ARGUMENTS})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraphtestutil
            GTest::gtest
            GTest::gtest_main
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUGRAPH_BINARY_DIR}/gtests>"
                       INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${CMAKE_TEST_NAME}
        GPUS ${_CUGRAPH_TEST_GPUS}
        PERCENT ${_CUGRAPH_TEST_PERCENT}
        INSTALL_COMPONENT_SET testing
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH")
endfunction()

function(ConfigureTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})
    target_include_directories(${CMAKE_TEST_NAME} PRIVATE "${CUGRAPH_SOURCE_DIR}/src")
    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        cugraphmgtestutil
        cugraphtestutil
        GTest::gtest
        GTest::gtest_main
        NCCL::NCCL
        MPI::MPI_CXX
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUGRAPH_BINARY_DIR}/gtests>"
                       INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE}
             "--noprefix"
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS}
        GPUS ${GPU_COUNT}
        PERCENT 100
        INSTALL_COMPONENT_SET testing_mg
        INSTALL_TARGET ${CMAKE_TEST_NAME}
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH_MG")

endfunction()

function(ConfigureCTest CMAKE_TEST_NAME)
    set(options)
    set(one_value GPUS PERCENT)
    set(multi_value)
    cmake_parse_arguments(_CUGRAPH_TEST "${options}" "${one_value}" "${multi_value}" ${ARGN})
    if(NOT DEFINED _CUGRAPH_TEST_GPUS AND NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_GPUS 1)
        set(_CUGRAPH_TEST_PERCENT 25)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_GPUS)
        set(_CUGRAPH_TEST_GPUS 1)
    endif()
    if(NOT DEFINED _CUGRAPH_TEST_PERCENT)
        set(_CUGRAPH_TEST_PERCENT 100)
    endif()

    add_executable(${CMAKE_TEST_NAME} ${_CUGRAPH_TEST_UNPARSED_ARGUMENTS})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraph::cugraph_c
            cugraph_c_testutil
            GTest::gtest
            GTest::gtest_main
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUGRAPH_BINARY_DIR}/gtests>"
                       INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${CMAKE_TEST_NAME}
        GPUS ${_CUGRAPH_TEST_GPUS}
        PERCENT ${_CUGRAPH_TEST_PERCENT}
        INSTALL_COMPONENT_SET testing_c
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH_C")

endfunction()

function(ConfigureCTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraph::cugraph_c
            cugraph_c_testutil
            cugraphmgtestutil
            GTest::gtest
            GTest::gtest_main
            NCCL::NCCL
            MPI::MPI_CXX
    )
    set_target_properties(
        ${CMAKE_TEST_NAME}
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUGRAPH_BINARY_DIR}/gtests>"
                       INSTALL_RPATH "\$ORIGIN/../../../lib"
                       CXX_STANDARD                        17
                       CXX_STANDARD_REQUIRED               ON
                       CUDA_STANDARD                       17
                       CUDA_STANDARD_REQUIRED              ON)

    rapids_test_add(
        NAME ${CMAKE_TEST_NAME}
        COMMAND ${MPIEXEC_EXECUTABLE}
             "--noprefix"
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS}
        GPUS ${GPU_COUNT}
        PERCENT 100
        INSTALL_COMPONENT_SET testing_mg
        INSTALL_TARGET ${CMAKE_TEST_NAME}
    )
    set_tests_properties(${CMAKE_TEST_NAME} PROPERTIES LABELS "CUGRAPH_C_MG")


endfunction()


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
# - LOUVAIN tests ---------------------------------------------------------------------------------
ConfigureTest(LOUVAIN_TEST community/louvain_test.cpp)

###################################################################################################
# - LEIDEN tests ----------------------------------------------------------------------------------
ConfigureTest(LEIDEN_TEST community/leiden_test.cpp)

###################################################################################################
# - WEIGHTED MATCHING tests -----------------------------------------------------------------------
ConfigureTest(WEIGHTED_MATCHING_TEST community/weighted_matching_test.cpp)

###################################################################################################
# - Balanced cut clustering tests -----------------------------------------------------------------
ConfigureTest(BALANCED_TEST community/balanced_edge_test.cpp)

###################################################################################################
# - EGO tests -------------------------------------------------------------------------------------
ConfigureTest(EGONET_TEST community/egonet_test.cpp GPUS 1 PERCENT 75)

###################################################################################################
# - FORCE ATLAS 2  tests --------------------------------------------------------------------------
ConfigureTest(LEGACY_FA2_TEST layout/legacy/force_atlas2_test.cu)

###################################################################################################
# - STRONGLY CONNECTED COMPONENTS  tests ----------------------------------------------------------
ConfigureTest(SCC_TEST components/scc_test.cu)
target_include_directories(SCC_TEST PRIVATE "${CUGRAPH_SOURCE_DIR}/src")

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
# - Bipartite R-mat graph generation tests --------------------------------------------------------
ConfigureTest(GENERATE_BIPARTITE_RMAT_TEST generators/generate_bipartite_rmat_test.cpp
    GPUS 1 PERCENT 100)

###################################################################################################
# - Remove multi-edges tests ----------------------------------------------------------------------
ConfigureTest(REMOVE_MULTI_EDGES_TEST structure/remove_multi_edges_test.cpp)

###################################################################################################
# - Symmetrize tests ------------------------------------------------------------------------------
ConfigureTest(SYMMETRIZE_TEST structure/symmetrize_test.cpp)

###################################################################################################
# - Transpose tests -------------------------------------------------------------------------------
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
# - Query edge existence and multiplicity tests ---------------------------------------------------
ConfigureTest(HAS_EDGE_AND_COMPUTE_MULTIPLICITY_TEST
              "structure/has_edge_and_compute_multiplicity_test.cpp")

###################################################################################################
# - Coarsening tests ------------------------------------------------------------------------------
ConfigureTest(COARSEN_GRAPH_TEST structure/coarsen_graph_test.cpp)

###################################################################################################
# - Induced subgraph tests ------------------------------------------------------------------------
ConfigureTest(INDUCED_SUBGRAPH_TEST structure/induced_subgraph_test.cpp)

##################################################################################################
# - Temporal tests -------------------------------------------------------------------------------
ConfigureTest(TEMPORAL_GRAPH_TEST structure/temporal_graph_test.cpp)

###################################################################################################
# - BFS tests -------------------------------------------------------------------------------------
ConfigureTest(BFS_TEST traversal/bfs_test.cpp)

###################################################################################################
# - Extract BFS Paths tests -----------------------------------------------------------------------
ConfigureTest(EXTRACT_BFS_PATHS_TEST
              traversal/extract_bfs_paths_test.cu)

###################################################################################################
# - Multi-source BFS tests ------------------------------------------------------------------------
ConfigureTest(MSBFS_TEST traversal/ms_bfs_test.cu)

###################################################################################################
# - SSSP tests ------------------------------------------------------------------------------------
ConfigureTest(SSSP_TEST traversal/sssp_test.cpp)

###################################################################################################
# - OD_SHORTEST_DISTANCES tests -------------------------------------------------------------------
ConfigureTest(OD_SHORTEST_DISTANCES_TEST traversal/od_shortest_distances_test.cpp)

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
# - EIGENVECTOR_CENTRALITY tests ------------------------------------------------------------------
ConfigureTest(EIGENVECTOR_CENTRALITY_TEST centrality/eigenvector_centrality_test.cpp)

###################################################################################################
# - BETWEENNESS_CENTRALITY tests ------------------------------------------------------------------
ConfigureTest(BETWEENNESS_CENTRALITY_TEST centrality/betweenness_centrality_test.cpp)
ConfigureTest(EDGE_BETWEENNESS_CENTRALITY_TEST centrality/edge_betweenness_centrality_test.cpp)

###################################################################################################
# - WEAKLY CONNECTED COMPONENTS tests -------------------------------------------------------------
ConfigureTest(WEAKLY_CONNECTED_COMPONENTS_TEST components/weakly_connected_components_test.cpp)

###################################################################################################
# - MIS tests -------------------------------------------------------------------------------------
ConfigureTest(MIS_TEST components/mis_test.cu)
target_include_directories(MIS_TEST PRIVATE "${CUGRAPH_SOURCE_DIR}/src")

###################################################################################################
# - VERTEX COLORING tests ----------------------------------------------------------------------
ConfigureTest(VERTEX_COLORING_TEST components/vertex_coloring_test.cu)
target_include_directories(VERTEX_COLORING_TEST PRIVATE "${CUGRAPH_SOURCE_DIR}/src")

###################################################################################################
# - SIMILARITY tests ------------------------------------------------------------------------------
ConfigureTest(SIMILARITY_TEST link_prediction/similarity_test.cu)

###################################################################################################
# - WEIGHTED_SIMILARITY tests ---------------------------------------------------------------------
ConfigureTest(WEIGHTED_SIMILARITY_TEST link_prediction/weighted_similarity_test.cpp)

###################################################################################################
# - RANDOM_WALKS tests ----------------------------------------------------------------------------
#  FIXME: Rename to random_walks_test.cu once the legacy implementation is deleted
ConfigureTest(RANDOM_WALKS_TEST sampling/sg_random_walks_test.cpp)

###################################################################################################
# - UNIFORM NBR SAMPLING tests --------------------------------------------------------------------
ConfigureTest(UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/uniform_neighbor_sampling.cpp
        GPUS 1 PERCENT 75)

# - HOMOGENEOUS UNIFORM NBR SAMPLING tests --------------------------------------------------------
ConfigureTest(
    HOMOGENEOUS_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/homogeneous_uniform_neighbor_sampling.cpp
        GPUS 1 PERCENT 75)

# - HETEROGENEOUS UNIFORM NBR SAMPLING tests -----------------------------------------------------
ConfigureTest(
    HETEROGENEOUS_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/heterogeneous_uniform_neighbor_sampling.cpp
        GPUS 1 PERCENT 75)

###################################################################################################
# - BIASED NBR SAMPLING tests ---------------------------------------------------------------------
ConfigureTest(BIASED_NEIGHBOR_SAMPLING_TEST sampling/biased_neighbor_sampling.cpp GPUS 1 PERCENT 75)

###################################################################################################
# - HOMOGENEOUS BIASED NBR SAMPLING tests ---------------------------------------------------------
ConfigureTest(
    HOMOGENEOUS_BIASED_NEIGHBOR_SAMPLING_TEST sampling/homogeneous_biased_neighbor_sampling.cpp GPUS 1 PERCENT 75)

###################################################################################################
# - HETEROGENEOUS BIASED NBR SAMPLING tests -------------------------------------------------------
ConfigureTest(
    HETEROGENEOUS_BIASED_NEIGHBOR_SAMPLING_TESTT sampling/heterogeneous_biased_neighbor_sampling.cpp
        GPUS 1 PERCENT 75)

###################################################################################################
# - SAMPLING_POST_PROCESSING tests ----------------------------------------------------------------
ConfigureTest(SAMPLING_POST_PROCESSING_TEST sampling/sampling_post_processing_test.cpp)

###################################################################################################
# - SAMPLING_HETEROGENEOUS_POST_PROCESSING tests --------------------------------------------------
ConfigureTest(SAMPLING_HETEROGENEOUS_POST_PROCESSING_TEST
              sampling/sampling_heterogeneous_post_processing_test.cpp)

###################################################################################################
# - NEGATIVE SAMPLING tests --------------------------------------------------------------------
ConfigureTest(NEGATIVE_SAMPLING_TEST sampling/negative_sampling.cpp PERCENT 100)

###################################################################################################
# - Renumber tests --------------------------------------------------------------------------------
ConfigureTest(RENUMBERING_TEST structure/renumbering_test.cpp)

###################################################################################################
# - Core Number tests -----------------------------------------------------------------------------
ConfigureTest(CORE_NUMBER_TEST cores/core_number_test.cpp)

###################################################################################################
# - Core Number tests -----------------------------------------------------------------------------
ConfigureTest(K_CORE_TEST cores/k_core_test.cpp)

###################################################################################################
# - K-truss tests --------------------------------------------------------------------------
ConfigureTest(K_TRUSS_TEST community/k_truss_test.cpp)

###################################################################################################
# - Triangle Count tests --------------------------------------------------------------------------
ConfigureTest(TRIANGLE_COUNT_TEST community/triangle_count_test.cpp)

###################################################################################################
# - Edge Triangle Count tests ---------------------------------------------------------------------
ConfigureTest(EDGE_TRIANGLE_COUNT_TEST community/edge_triangle_count_test.cpp)

###################################################################################################
# - EDGE SOURCE DESTINATION LOOKUP tests ----------------------------------------------------------
ConfigureTest(LOOKUP_SRC_DST_TEST lookup/lookup_src_dst_test.cpp)

###################################################################################################
# - K-hop Neighbors tests -------------------------------------------------------------------------
ConfigureTest(K_HOP_NBRS_TEST traversal/k_hop_nbrs_test.cpp)


###################################################################################################
# - install tests ---------------------------------------------------------------------------------
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcugraph)


###################################################################################################
# - MG tests --------------------------------------------------------------------------------------

if(BUILD_CUGRAPH_MG_TESTS)
    # Set the GPU count to 1.  If the caller wants to execute MG tests using
    # more than 1, override from the command line using -DGPU_COUNT=<gpucount>
    if (NOT DEFINED GPU_COUNT)
      set(GPU_COUNT "1")
    endif()

    ###############################################################################################
    # - MG SYMMETRIZE tests -----------------------------------------------------------------------
    ConfigureTestMG(MG_SYMMETRIZE_TEST structure/mg_symmetrize_test.cpp)

    ###############################################################################################
    # - MG Temporal Graph tests -----------------------------------------------------------------------
    ConfigureTestMG(MG_TEMPORAL_GRAPH_TEST structure/mg_temporal_graph_test.cpp)

    ###############################################################################################
    # - MG Transpose tests ------------------------------------------------------------------------
    ConfigureTestMG(MG_TRANSPOSE_TEST structure/mg_transpose_test.cpp)

    ###############################################################################################
    # - MG Transpose Storage tests ----------------------------------------------------------------
    ConfigureTestMG(MG_TRANSPOSE_STORAGE_TEST structure/mg_transpose_storage_test.cpp)

    ###############################################################################################
    # - MG Count self-loops and multi-edges tests -------------------------------------------------
    ConfigureTestMG(MG_COUNT_SELF_LOOPS_AND_MULTI_EDGES_TEST
          "structure/mg_count_self_loops_and_multi_edges_test.cpp")

    ###############################################################################################
    # - MG Query edge existence and multiplicity tests --------------------------------------------
    ConfigureTestMG(MG_HAS_EDGE_AND_COMPUTE_MULTIPLICITY_TEST
          "structure/mg_has_edge_and_compute_multiplicity_test.cpp")

    ###############################################################################################
    # - MG PAGERANK tests -------------------------------------------------------------------------
    ConfigureTestMG(MG_PAGERANK_TEST link_analysis/mg_pagerank_test.cpp)

    ###############################################################################################
    # - MG HITS tests -----------------------------------------------------------------------------
    ConfigureTestMG(MG_HITS_TEST link_analysis/mg_hits_test.cpp)

    ###############################################################################################
    # - MG KATZ CENTRALITY tests ------------------------------------------------------------------
    ConfigureTestMG(MG_KATZ_CENTRALITY_TEST centrality/mg_katz_centrality_test.cpp)

    ###############################################################################################
    # - MG EIGENVECTOR CENTRALITY tests -----------------------------------------------------------
    ConfigureTestMG(MG_EIGENVECTOR_CENTRALITY_TEST centrality/mg_eigenvector_centrality_test.cpp)

    ###############################################################################################
    # - MG BETWEENNESS CENTRALITY tests -----------------------------------------------------------
    ConfigureTestMG(MG_BETWEENNESS_CENTRALITY_TEST centrality/mg_betweenness_centrality_test.cpp)
    ConfigureTestMG(MG_EDGE_BETWEENNESS_CENTRALITY_TEST
                    centrality/mg_edge_betweenness_centrality_test.cpp)

    ###############################################################################################
    # - MG BFS tests ------------------------------------------------------------------------------
    ConfigureTestMG(MG_BFS_TEST traversal/mg_bfs_test.cpp)

    ###############################################################################################
    # - Extract BFS Paths tests -------------------------------------------------------------------
    ConfigureTestMG(MG_EXTRACT_BFS_PATHS_TEST
                    traversal/mg_extract_bfs_paths_test.cu)

    ###############################################################################################
    # - MG SSSP tests -----------------------------------------------------------------------------
    ConfigureTestMG(MG_SSSP_TEST traversal/mg_sssp_test.cpp)

    ###############################################################################################
    # - MG LOUVAIN tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_LOUVAIN_TEST community/mg_louvain_test.cpp)

    ###############################################################################################
    # - MG LEIDEN tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_LEIDEN_TEST community/mg_leiden_test.cpp)

    ###############################################################################################
    # - MG WEIGHTED MATCHING tests ----------------------------------------------------------------
    ConfigureTestMG(MG_WEIGHTED_MATCHING_TEST community/mg_weighted_matching_test.cpp)

    ###############################################################################################
    # - MG ECG tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_ECG_TEST community/mg_ecg_test.cpp)

    ###############################################################################################
    # - MG SELECT RANDOM VERTICES tests -----------------------------------------------------------
    ConfigureTestMG(MG_SELECT_RANDOM_VERTICES structure/mg_select_random_vertices_test.cpp)

    ###############################################################################################
    # - MG LOUVAIN tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_EGONET_TEST community/mg_egonet_test.cu)

    ###############################################################################################
    # - MG EDGE TRIANGLE COUNT tests --------------------------------------------------------------
    ConfigureTestMG(MG_EDGE_TRIANGLE_COUNT_TEST community/mg_edge_triangle_count_test.cpp)

    ###############################################################################################
    # - MG K-TRUSS tests --------------------------------------------------------------------------
    ConfigureTestMG(MG_K_TRUSS_TEST community/mg_k_truss_test.cpp)

    ###############################################################################################
    # - MG WEAKLY CONNECTED COMPONENTS tests ------------------------------------------------------
    ConfigureTestMG(MG_WEAKLY_CONNECTED_COMPONENTS_TEST
                    components/mg_weakly_connected_components_test.cpp)

    ###############################################################################################
    # - MG EDGE SOURCE DESTINATION LOOKUP tests ---------------------------------------------------
    ConfigureTestMG(MG_LOOKUP_SRC_DST_TEST lookup/mg_lookup_src_dst_test.cpp)

    ###############################################################################################
    # - MG MIS tests ------------------------------------------------------------------------------
    ConfigureTestMG(MG_MIS_TEST components/mg_mis_test.cu)

    ###############################################################################################
    # - MG VERTEX COLORING tests -------------------------------------------------------------------
    ConfigureTestMG(MG_VERTEX_COLORING_TEST components/mg_vertex_coloring_test.cu)

    ###############################################################################################
    # - MG Core Number tests ----------------------------------------------------------------------
    ConfigureTestMG(MG_CORE_NUMBER_TEST cores/mg_core_number_test.cpp)

    ###############################################################################################
    # - MG K Core tests ---------------------------------------------------------------------------
    ConfigureTestMG(MG_K_CORE_TEST cores/mg_k_core_test.cpp)

    ###############################################################################################
    # - MG TRIANGLE COUNT tests -------------------------------------------------------------------
    ConfigureTestMG(MG_TRIANGLE_COUNT_TEST community/mg_triangle_count_test.cpp)

    ###############################################################################################
    # - MG coarsening tests -----------------------------------------------------------------------
    ConfigureTestMG(MG_COARSEN_GRAPH_TEST structure/mg_coarsen_graph_test.cpp)

    ###############################################################################################
    # - MG INDUCED SUBGRAPH tests -----------------------------------------------------------------
    ConfigureTestMG(MG_INDUCED_SUBGRAPH_TEST structure/mg_induced_subgraph_test.cu)

    ###############################################################################################
    # - MG PRIMS COUNT_IF_V tests -----------------------------------------------------------------
    ConfigureTestMG(MG_COUNT_IF_V_TEST prims/mg_count_if_v.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_V_FRONTIER_OUTGOING_E_BY_DST tests ------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_V_FRONTIER_OUTGOING_E_BY_DST_TEST
                    prims/mg_transform_reduce_v_frontier_outgoing_e_by_dst.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_IF_V_FRONTIER_OUTGOING_E_BY_DST tests ---------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_IF_V_FRONTIER_OUTGOING_E_BY_DST_TEST
                    prims/mg_transform_reduce_if_v_frontier_outgoing_e_by_dst.cu)

    ###############################################################################################
    # - MG PRIMS REDUCE_V tests -------------------------------------------------------------------
    ConfigureTestMG(MG_REDUCE_V_TEST prims/mg_reduce_v.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_V tests ---------------------------------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_V_TEST prims/mg_transform_reduce_v.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_E tests ---------------------------------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_E_TEST prims/mg_transform_reduce_e.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_E _BY_SRC_DST_KEY tests -----------------------------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_E_BY_SRC_DST_KEY_TEST
                    prims/mg_transform_reduce_e_by_src_dst_key.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_E tests ----------------------------------------------------------------
    ConfigureTestMG(MG_TRANSFORM_E_TEST prims/mg_transform_e.cu)

    ###############################################################################################
    # - MG PRIMS COUNT_IF_E tests -----------------------------------------------------------------
    ConfigureTestMG(MG_COUNT_IF_E_TEST prims/mg_count_if_e.cu)

    ###############################################################################################
    # - MG PRIMS PER_V_TRANSFORM_REDUCE_INCOMING_OUTGOING_E tests ---------------------------------
    ConfigureTestMG(MG_PER_V_TRANSFORM_REDUCE_INCOMING_OUTGOING_E_TEST
      prims/mg_per_v_transform_reduce_incoming_outgoing_e.cu)

    ###############################################################################################
    # - MG PRIMS PER_V_TRANSFORM_REDUCE_DST_KEY_AGGREGATED_OUTGOING_E tests -----------------------
    ConfigureTestMG(MG_PER_V_TRANSFORM_REDUCE_DST_KEY_AGGREGATED_OUTGOING_E_TEST
      prims/mg_per_v_transform_reduce_dst_key_aggregated_outgoing_e.cu)

    ###############################################################################################
    # - MG PRIMS EXTRACT_TRANSFORM_E tests --------------------------------------------------------
    ConfigureTestMG(MG_EXTRACT_TRANSFORM_E_TEST prims/mg_extract_transform_e.cu)

    ###############################################################################################
    # - MG PRIMS EXTRACT_TRANSFORM_IF_E tests -----------------------------------------------------
    ConfigureTestMG(MG_EXTRACT_TRANSFORM_IF_E_TEST prims/mg_extract_transform_if_e.cu)

    ###############################################################################################
    # - MG PRIMS EXTRACT_TRANSFORM_V_FRONTIER_OUTGOING_E tests ------------------------------------
    ConfigureTestMG(MG_EXTRACT_TRANSFORM_V_FRONTIER_OUTGOING_E_TEST
                    prims/mg_extract_transform_v_frontier_outgoing_e.cu)

    ###############################################################################################
    # - MG PRIMS EXTRACT_TRANSFORM_IF_V_FRONTIER_OUTGOING_E tests ---------------------------------
    ConfigureTestMG(MG_EXTRACT_TRANSFORM_IF_V_FRONTIER_OUTGOING_E_TEST
                    prims/mg_extract_transform_if_v_frontier_outgoing_e.cu)

    ###############################################################################################
    # - MG PRIMS PER_V_RANDOM_SELECT_TRANSFORM_OUTGOING_E tests -----------------------------------
    ConfigureTestMG(MG_PER_V_RANDOM_SELECT_TRANSFORM_OUTGOING_E_TEST
                    prims/mg_per_v_random_select_transform_outgoing_e.cu)

    ###############################################################################################
    # - MG PRIMS PER_V_PAIR_TRANSFORM_DST_NBR_INTERSECTION tests ----------------------------------
    ConfigureTestMG(MG_PER_V_PAIR_TRANSFORM_DST_NBR_INTERSECTION_TEST
                    prims/mg_per_v_pair_transform_dst_nbr_intersection.cu)

    ###############################################################################################
    # - MG PRIMS TRANSFORM_REDUCE_DST_NBR_INTERSECTION OF_E_ENDPOINTS_BY_V tests ------------------
    ConfigureTestMG(MG_TRANSFORM_REDUCE_DST_NBR_INTERSECTION_BY_E_ENDPOINTS_BY_V_TEST
                    prims/mg_transform_reduce_dst_nbr_intersection_of_e_endpoints_by_v.cu)

    ###############################################################################################
    # - MG PRIMS PER_V_PAIR_TRANSFORM_DST_NBR_WEIGHTED_INTERSECTION tests -------------------------
    ConfigureTestMG(MG_PER_V_PAIR_TRANSFORM_DST_NBR_WEIGHTED_INTERSECTION_TEST
                    prims/mg_per_v_pair_transform_dst_nbr_weighted_intersection.cu)

    ###############################################################################################
    # - MG UNIFORM NBR SAMPLING tests -------------------------------------------------------------
    ConfigureTestMG(MG_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/mg_uniform_neighbor_sampling.cpp)

    ###############################################################################################
    # - MG HOMOGENEOUS UNIFORM NBR SAMPLING tests -------------------------------------------------
    ConfigureTestMG(
        MG_HOMOGENEOUS_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/mg_homogeneous_uniform_neighbor_sampling.cpp)

    ###############################################################################################
    # - MG HETEROGENEOUS UNIFORM NBR SAMPLING tests -------------------------------------------------
    ConfigureTestMG(
        MG_HETEROGENEOUS_UNIFORM_NEIGHBOR_SAMPLING_TEST sampling/mg_heterogeneous_uniform_neighbor_sampling.cpp)

    ###############################################################################################
    # - MG HOMOGENEOUS BIASED NBR SAMPLING tests --------------------------------------------------
    ConfigureTestMG(
        MG_HOMOGENEOUS_BIASED_NEIGHBOR_SAMPLING_TEST sampling/mg_homogeneous_biased_neighbor_sampling.cpp)

    ###############################################################################################
    # - MG HETEROGENEOUS BIASED NBR SAMPLING tests --------------------------------------------------
    ConfigureTestMG(
        MG_HETEROGENEOUS_BIASED_NEIGHBOR_SAMPLING_TEST sampling/mg_heterogeneous_biased_neighbor_sampling.cpp)

    ###############################################################################################
    # - MG BIASED NBR SAMPLING tests --------------------------------------------------------------
    ConfigureTestMG(MG_BIASED_NEIGHBOR_SAMPLING_TEST sampling/mg_biased_neighbor_sampling.cpp)

    ###################################################################################################
    # - NEGATIVE SAMPLING tests --------------------------------------------------------------------
    ConfigureTestMG(MG_NEGATIVE_SAMPLING_TEST sampling/mg_negative_sampling.cpp)


    ###############################################################################################
    # - MG RANDOM_WALKS tests ---------------------------------------------------------------------
    ConfigureTestMG(MG_RANDOM_WALKS_TEST sampling/mg_random_walks_test.cpp)

    ###############################################################################################
    # - MG WEIGHTED_SIMILARITY tests --------------------------------------------------------------
    ConfigureTestMG(MG_WEIGHTED_SIMILARITY_TEST link_prediction/mg_weighted_similarity_test.cpp)

    ###############################################################################################
    # - MG SIMILARITY tests -----------------------------------------------------------------------
    ConfigureTestMG(MG_SIMILARITY_TEST link_prediction/mg_similarity_test.cpp)

    ###############################################################################################
    # - MG K_HOP_NBRS tests -----------------------------------------------------------------------
    ConfigureTestMG(MG_K_HOP_NBRS_TEST traversal/mg_k_hop_nbrs_test.cpp)

    ###############################################################################################
    # - MG C API tests ----------------------------------------------------------------------------
    ConfigureCTestMG(MG_CAPI_CREATE_GRAPH_TEST c_api/mg_create_graph_test.c)
    ConfigureCTestMG(MG_CAPI_GENERATE_RMAT_TEST c_api/mg_generate_rmat_test.c)
    ConfigureCTestMG(MG_CAPI_PAGERANK_TEST c_api/mg_pagerank_test.c)
    ConfigureCTestMG(MG_CAPI_BFS_TEST c_api/mg_bfs_test.c)
    ConfigureCTestMG(MG_CAPI_SSSP_TEST c_api/mg_sssp_test.c)
    ConfigureCTestMG(MG_CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST c_api/mg_weakly_connected_components_test.c)
    ConfigureCTestMG(MG_CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST c_api/mg_strongly_connected_components_test.c)
    ConfigureCTestMG(MG_CAPI_KATZ_TEST c_api/mg_katz_test.c)
    ConfigureCTestMG(MG_CAPI_EIGENVECTOR_CENTRALITY_TEST c_api/mg_eigenvector_centrality_test.c)
    ConfigureCTestMG(MG_CAPI_BETWEENNESS_CENTRALITY_TEST c_api/mg_betweenness_centrality_test.c)
    ConfigureCTestMG(MG_CAPI_EDGE_BETWEENNESS_CENTRALITY_TEST c_api/mg_edge_betweenness_centrality_test.c)
    ConfigureCTestMG(MG_CAPI_HITS_TEST c_api/mg_hits_test.c)
    ConfigureCTestMG(MG_CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST c_api/mg_uniform_neighbor_sample_test.c)
    ConfigureCTestMG(MG_CAPI_BIASED_NEIGHBOR_SAMPLE_TEST c_api/mg_biased_neighbor_sample_test.c)
    ConfigureCTestMG(MG_CAPI_NEGATIVE_SAMPLING_TEST c_api/mg_negative_sampling_test.c)
    ConfigureCTestMG(MG_CAPI_LOOKUP_SRC_DST_TEST c_api/mg_lookup_src_dst_test.c)
    ConfigureCTestMG(MG_CAPI_RANDOM_WALKS_TEST c_api/mg_random_walks_test.c)
    ConfigureCTestMG(MG_CAPI_TRIANGLE_COUNT_TEST c_api/mg_triangle_count_test.c)
    ConfigureCTestMG(MG_CAPI_LOUVAIN_TEST c_api/mg_louvain_test.c)
    ConfigureCTestMG(MG_CAPI_LEIDEN_TEST c_api/mg_leiden_test.c)
    ConfigureCTestMG(MG_CAPI_ECG_TEST c_api/mg_ecg_test.c)
    ConfigureCTestMG(MG_CAPI_CORE_NUMBER_TEST c_api/mg_core_number_test.c)
    ConfigureCTestMG(MG_CAPI_SIMILARITY_TEST c_api/mg_similarity_test.c)
    ConfigureCTestMG(MG_CAPI_K_CORE_TEST c_api/mg_k_core_test.c)
    ConfigureCTestMG(MG_CAPI_INDUCED_SUBGRAPH_TEST c_api/mg_induced_subgraph_test.c)
    ConfigureCTestMG(MG_CAPI_DEGREES c_api/mg_degrees_test.c)
    ConfigureCTestMG(MG_CAPI_COUNT_MULTI_EDGES c_api/mg_count_multi_edges_test.c)
    ConfigureCTestMG(MG_CAPI_EGONET_TEST c_api/mg_egonet_test.c)
    ConfigureCTestMG(MG_CAPI_TWO_HOP_NEIGHBORS_TEST c_api/mg_two_hop_neighbors_test.c)
    ConfigureCTestMG(MG_CAPI_K_TRUSS c_api/mg_k_truss_test.c)

    rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing_mg DESTINATION bin/gtests/libcugraph_mg)

endif()

###################################################################################################
# - C API tests -----------------------------------------------------------------------------------

ConfigureCTest(CAPI_CREATE_GRAPH_TEST c_api/create_graph_test.c)
ConfigureCTest(CAPI_GENERATE_RMAT_TEST c_api/generate_rmat_test.c)
ConfigureCTest(CAPI_PAGERANK_TEST c_api/pagerank_test.c)
ConfigureCTest(CAPI_KATZ_TEST c_api/katz_test.c)
ConfigureCTest(CAPI_EIGENVECTOR_CENTRALITY_TEST c_api/eigenvector_centrality_test.c)
ConfigureCTest(CAPI_BETWEENNESS_CENTRALITY_TEST c_api/betweenness_centrality_test.c)
ConfigureCTest(CAPI_EDGE_BETWEENNESS_CENTRALITY_TEST c_api/edge_betweenness_centrality_test.c)
ConfigureCTest(CAPI_HITS_TEST c_api/hits_test.c)
ConfigureCTest(CAPI_BFS_TEST c_api/bfs_test.c)
ConfigureCTest(CAPI_SSSP_TEST c_api/sssp_test.c)
ConfigureCTest(CAPI_EXTRACT_PATHS_TEST c_api/extract_paths_test.c)
ConfigureCTest(CAPI_NODE2VEC_TEST c_api/node2vec_test.c)
ConfigureCTest(CAPI_WEAKLY_CONNECTED_COMPONENTS_TEST c_api/weakly_connected_components_test.c)
ConfigureCTest(CAPI_STRONGLY_CONNECTED_COMPONENTS_TEST c_api/strongly_connected_components_test.c)
ConfigureCTest(CAPI_UNIFORM_NEIGHBOR_SAMPLE_TEST c_api/uniform_neighbor_sample_test.c)
ConfigureCTest(CAPI_BIASED_NEIGHBOR_SAMPLE_TEST c_api/biased_neighbor_sample_test.c)
ConfigureCTest(CAPI_NEGATIVE_SAMPLING_TEST c_api/negative_sampling_test.c)
ConfigureCTest(CAPI_RANDOM_WALKS_TEST c_api/sg_random_walks_test.c)
ConfigureCTest(CAPI_TRIANGLE_COUNT_TEST c_api/triangle_count_test.c)
ConfigureCTest(CAPI_LOUVAIN_TEST c_api/louvain_test.c)
ConfigureCTest(CAPI_LEIDEN_TEST c_api/leiden_test.c)
ConfigureCTest(CAPI_ECG_TEST c_api/ecg_test.c)
ConfigureCTest(CAPI_RENUMBER_ARBITRARY_EDGELIST_TEST c_api/renumber_arbitrary_edgelist_test.c)

#############################################################################
# Skipping due to CUDA 12.2 failure that traces back to RAFT                #
# TODO: Uncomment this once the issue is fixed.                             #
#                                                                           #
# xref: https://github.com/rapidsai/cugraph/issues/4173                     #
# xref: https://github.com/rapidsai/raft/issues/2186                        #
#############################################################################
# ConfigureCTest(CAPI_LEGACY_SPECTRAL_TEST c_api/legacy_spectral_test.c)
ConfigureCTest(CAPI_CORE_NUMBER_TEST c_api/core_number_test.c)
ConfigureCTest(CAPI_SIMILARITY_TEST c_api/similarity_test.c)
ConfigureCTest(CAPI_K_CORE_TEST c_api/k_core_test.c)
ConfigureCTest(CAPI_INDUCED_SUBGRAPH_TEST c_api/induced_subgraph_test.c)
ConfigureCTest(CAPI_DEGREES c_api/degrees_test.c)
ConfigureCTest(CAPI_COUNT_MULTI_EDGES c_api/count_multi_edges_test.c)
ConfigureCTest(CAPI_EGONET_TEST c_api/egonet_test.c)
ConfigureCTest(CAPI_TWO_HOP_NEIGHBORS_TEST c_api/two_hop_neighbors_test.c)
ConfigureCTest(CAPI_K_TRUSS_TEST c_api/k_truss_test.c)

if (BUILD_CUGRAPH_MTMG_TESTS)
    ###################################################################################################
    # - MTMG tests -------------------------------------------------------------------------
    ConfigureTest(MTMG_TEST mtmg/threaded_test.cpp)
    target_link_libraries(MTMG_TEST
                          PRIVATE
                          cugraphmgtestutil
                          ${COMPILED_RAFT_LIB}
                          ucx::ucp
                          ucx::ucs
                          ucxx::ucxx
                         )

    ConfigureTest(MTMG_LOUVAIN_TEST mtmg/threaded_test_louvain.cpp)
    target_link_libraries(MTMG_LOUVAIN_TEST
                          PRIVATE
                          cugraphmgtestutil
                          ${COMPILED_RAFT_LIB}
                          ucx::ucp
                          ucx::ucs
                          ucxx::ucxx
                         )

    ConfigureTest(MTMG_JACCARD_TEST mtmg/threaded_test_jaccard.cpp)
    target_link_libraries(MTMG_JACCARD_TEST
                          PRIVATE
                          cugraphmgtestutil
                          ${COMPILED_RAFT_LIB}
                          ucx::ucp
                          ucx::ucs
                          ucxx::ucxx
                         )

    if(BUILD_CUGRAPH_MG_TESTS)
        ###############################################################################################
        # - Multi-node MTMG tests ---------------------------------------------------------------------
        ConfigureTest(MTMG_MULTINODE_TEST mtmg/multi_node_threaded_test.cpp utilities/mg_utilities.cpp)
        target_link_libraries(MTMG_MULTINODE_TEST
                              PRIVATE
                              cugraphmgtestutil
                              ${COMPILED_RAFT_LIB}
                              ucx::ucp
                              ucx::ucs
                              ucxx::ucxx
                             )
    endif(BUILD_CUGRAPH_MG_TESTS)
endif(BUILD_CUGRAPH_MTMG_TESTS)

rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing_c DESTINATION bin/gtests/libcugraph_c)
