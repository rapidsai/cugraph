#=============================================================================
#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
            utilities/rmat_utilities.cpp
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
        cugraph
        cuco::cuco
        NCCL::NCCL
)


add_library(cugraphmgtestutil STATIC
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/device_comm_wrapper.cu")

set_property(TARGET cugraphmgtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphmgtestutil
    PRIVATE
    "${CUB_INCLUDE_DIR}"
    "${THRUST_INCLUDE_DIR}"
    "${CUCO_INCLUDE_DIR}"
    "${LIBCUDACXX_INCLUDE_DIR}"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${RMM_INCLUDE}"
    "${NCCL_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${RAFT_DIR}/cpp/include"
)

target_link_libraries(cugraphmgtestutil cugraph)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
            cugraphtestutil
            cugraph
            GTest::gmock
            GTest::gmock_main
            GTest::gtest
            GTest::gtest_main
            NCCL::NCCL
            CUDA::cublas
            CUDA::cusparse
            CUDA::cusolver
            CUDA::curand
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${CMAKE_TEST_NAME} PRIVATE
###################################################################################################
###   Use ${OpenMP_CXX_LIB_NAMES} instead of OpenMP::OpenMP_CXX to avoid the following warnings.
###
###   Cannot generate a safe runtime search path for target TARGET_NAME
###   because files in some directories may conflict with libraries in implicit
###   directories:
###   ...
###
###   libgomp.so is included in the conda base environment and copied to every new conda
###   environment. If a full file path is provided (e.g ${CUDF_LIBRARY}), cmake
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
###   path included in BUILD_RPATH when ${CUDF_LIBRARY} is added; this one can
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

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction()

function(ConfigureTestMG CMAKE_TEST_NAME)
    add_executable(${CMAKE_TEST_NAME} ${ARGN})

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        cugraphmgtestutil
        cugraphtestutil
        cugraph
        GTest::gmock
        GTest::gmock_main
        GTest::gtest
        GTest::gtest_main
        NCCL::NCCL
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
        CUDA::curand
        MPI::MPI_CXX
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${CMAKE_TEST_NAME} PRIVATE
###################################################################################################
###   Use ${OpenMP_CXX_LIB_NAMES} instead of OpenMP::OpenMP_CXX to avoid the following warnings.
###
###   Cannot generate a safe runtime search path for target TARGET_NAME
###   because files in some directories may conflict with libraries in implicit
###   directories:
###   ...
###
###   libgomp.so is included in the conda base environment and copied to every new conda
###   environment. If a full file path is provided (e.g ${CUDF_LIBRARY}), cmake
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
###   path included in BUILD_RPATH when ${CUDF_LIBRARY} is added; this one can
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

    add_test(NAME ${CMAKE_TEST_NAME}
             COMMAND ${MPIEXEC_EXECUTABLE}
             ${MPIEXEC_NUMPROC_FLAG}
             ${GPU_COUNT}
             ${MPIEXEC_PREFLAGS}
             ${CMAKE_TEST_NAME}
             ${MPIEXEC_POSTFLAGS})
endfunction()

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

set(GRAPH_GENERATORS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/generators/generators_test.cpp")

  ConfigureTest(GRAPH_GENERATORS_TEST "${GRAPH_GENERATORS_TEST_SRC}")

###################################################################################################
# - erdos renyi graph generator tests -------------------------------------------------------------

set(ERDOS_RENYI_GENERATOR_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/generators/erdos_renyi_test.cpp")

  ConfigureTest(ERDOS_RENYI_GENERATOR_TEST "${ERDOS_RENYI_GENERATOR_TEST_SRC}")

###################################################################################################
# - katz centrality tests -------------------------------------------------------------------------
ConfigureTest(LEGACY_KATZ_TEST centrality/legacy/katz_centrality_test.cu)

###################################################################################################
# - betweenness centrality tests ------------------------------------------------------------------
ConfigureTest(LEGACY_BETWEENNESS_TEST centrality/legacy/betweenness_centrality_test.cu)
ConfigureTest(LEGACY_EDGE_BETWEENNESS_TEST centrality/legacy/edge_betweenness_centrality_test.cu)

###################################################################################################
# - SSSP tests ------------------------------------------------------------------------------------
ConfigureTest(LEGACY_SSSP_TEST traversal/legacy/sssp_test.cu)

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
# - TSP  tests --------------------------------------------------------------------------
ConfigureTest(TSP_TEST traversal/tsp_test.cu)

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
# - MST tests ----------------------------------------------------------------------------
ConfigureTest(MST_TEST tree/mst_test.cu)

###################################################################################################
# - Experimental stream tests -----------------------------------------------------
ConfigureTest(EXPERIMENTAL_STREAM experimental/streams.cu)

###################################################################################################
# - Experimental R-mat graph generation tests -----------------------------------------------------
ConfigureTest(EXPERIMENTAL_GENERATE_RMAT_TEST experimental/generate_rmat_test.cpp)

###################################################################################################
# - Experimental Graph tests ----------------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_GRAPH_TEST experimental/graph_test.cpp)

###################################################################################################
# - Experimental weight-sum tests -----------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_WEIGHT_SUM_TEST experimental/weight_sum_test.cpp)

###################################################################################################
# - Experimental degree tests ---------------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_DEGREE_TEST experimental/degree_test.cpp)

###################################################################################################
# - Experimental coarsening tests -----------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_COARSEN_GRAPH_TEST experimental/coarsen_graph_test.cpp)

###################################################################################################
# - Experimental induced subgraph tests -----------------------------------------------------------
ConfigureTest(EXPERIMENTAL_INDUCED_SUBGRAPH_TEST experimental/induced_subgraph_test.cpp)

###################################################################################################
# - Experimental BFS tests ------------------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_BFS_TEST experimental/bfs_test.cpp)

###################################################################################################
# - Experimental Multi-source BFS tests -----------------------------------------------------------
ConfigureTest(EXPERIMENTAL_MSBFS_TEST experimental/ms_bfs_test.cpp)

###################################################################################################
# - Experimental SSSP tests -----------------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_SSSP_TEST experimental/sssp_test.cpp)

###################################################################################################
# - Experimental PAGERANK tests -------------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_PAGERANK_TEST experimental/pagerank_test.cpp)

###################################################################################################
# - Experimental KATZ_CENTRALITY tests ------------------------------------------------------------
ConfigureTest(EXPERIMENTAL_KATZ_CENTRALITY_TEST experimental/katz_centrality_test.cpp)

###################################################################################################
# - WEAKLY CONNECTED COMPONENTS tests -------------------------------------------------------------

set(WEAKLY_CONNECTED_COMPONENTS_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/components/weakly_connected_components_test.cpp")

ConfigureTest(WEAKLY_CONNECTED_COMPONENTS_TEST "${WEAKLY_CONNECTED_COMPONENTS_TEST_SRCS}")

###################################################################################################
# - Experimental RANDOM_WALKS tests ---------------------------------------------------------------
ConfigureTest(RANDOM_WALKS_TEST sampling/random_walks_test.cu)

###################################################################################################
ConfigureTest(RANDOM_WALKS_LOW_LEVEL_TEST sampling/rw_low_level_test.cu)

###################################################################################################
# FIXME: since this is technically not a test, consider refactoring the the
# ConfigureTest function to share common code with a new ConfigureBenchmark
# function (which would not link gtest, etc.)
ConfigureTest(RANDOM_WALKS_PROFILING sampling/random_walks_profiling.cu)

###################################################################################################
# - Serialization tests ---------------------------------------------------------------------------

set(SERIALIZATION_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/serialization/un_serialize_test.cpp")

ConfigureTest(SERIALIZATION_TEST "${SERIALIZATION_TEST_SRCS}")


###################################################################################################
# - BFS Visitor tests -----------------------------------------------------------------------------

set(BFS_VISITOR_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/visitors/bfs_test.cpp")

ConfigureTest(BFS_VISITOR_TEST "${BFS_VISITOR_TEST_SRCS}")

###################################################################################################
# - MG tests --------------------------------------------------------------------------------------

if(BUILD_CUGRAPH_MG_TESTS)

    ###############################################################################################
    # - find MPI - only enabled if MG tests are to be built
    find_package(MPI REQUIRED)

    execute_process(
      COMMAND nvidia-smi -L
      COMMAND wc -l
      OUTPUT_VARIABLE GPU_COUNT)

    string(REGEX REPLACE "\n$" "" GPU_COUNT ${GPU_COUNT})
    MESSAGE(STATUS "GPU_COUNT: " ${GPU_COUNT})

    if(MPI_CXX_FOUND)
        ###########################################################################################
        # - MG PAGERANK tests ---------------------------------------------------------------------
        ConfigureTestMG(MG_PAGERANK_TEST pagerank/mg_pagerank_test.cpp)

        ###########################################################################################
        # - MG KATZ CENTRALITY tests --------------------------------------------------------------
        ConfigureTestMG(MG_KATZ_CENTRALITY_TEST experimental/mg_katz_centrality_test.cpp)

        ###########################################################################################
        # - MG BFS tests --------------------------------------------------------------------------
        ConfigureTestMG(MG_BFS_TEST experimental/mg_bfs_test.cpp)

        ###########################################################################################
        # - MG SSSP tests -------------------------------------------------------------------------
        ConfigureTestMG(MG_SSSP_TEST experimental/mg_sssp_test.cpp)

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
        # - MG PRIMS COUNT_IF_V tests -------------------------------------------------------------
        ConfigureTestMG(MG_COUNT_IF_V_TEST prims/mg_count_if_v.cu)

        ###########################################################################################
        # - MG PRIMS REDUCE_V tests ---------------------------------------------------------------
        ConfigureTestMG(MG_REDUCE_V_TEST prims/mg_reduce_v.cu)
    else()
       message(FATAL_ERROR "OpenMPI NOT found, cannot build MG tests.")
    endif()
endif()

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
