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
            "${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_utilities.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio/mmio.c")

set_property(TARGET cugraphtestutil PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cugraphtestutil
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

target_link_libraries(cugraphtestutil cugraph)

# CUDA_ARCHITECTURES=OFF implies cmake will not pass arch flags to the
# compiler. CUDA_ARCHITECTURES must be set to a non-empty value to prevent
# cmake warnings about policy CMP0104. With this setting, arch flags must be
# manually set! ("evaluate_gpu_archs(GPU_ARCHS)" is the current mechanism
# used in cpp/CMakeLists.txt for setting arch options).
# Run "cmake --help-policy CMP0104" for policy details.
# NOTE: the CUDA_ARCHITECTURES=OFF setting may be removed after migrating to
# the findcudatoolkit features in cmake 3.17+
set_target_properties(cugraphtestutil PROPERTIES
        CUDA_ARCHITECTURES OFF)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
        ${CMAKE_TEST_SRC})

    target_include_directories(${CMAKE_TEST_NAME}
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
        "${CMAKE_CURRENT_SOURCE_DIR}/../src"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${RAFT_DIR}/cpp/include"
    )

    target_link_directories(${CMAKE_TEST_NAME}
        PRIVATE
        # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported
        # variable containing the link directories for nvcc.
        "${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        cugraphtestutil
        cugraph
        GTest::GTest
        GTest::Main
        ${CUDF_LIBRARY}
        ${NCCL_LIBRARIES}
        cudart
        cuda
        cublas
        cusparse
        cusolver
        curand)

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

    # CUDA_ARCHITECTURES=OFF implies cmake will not pass arch flags to the
    # compiler. CUDA_ARCHITECTURES must be set to a non-empty value to prevent
    # cmake warnings about policy CMP0104. With this setting, arch flags must be
    # manually set! ("evaluate_gpu_archs(GPU_ARCHS)" is the current mechanism
    # used in cpp/CMakeLists.txt for setting arch options).
    # Run "cmake --help-policy CMP0104" for policy details.
    # NOTE: the CUDA_ARCHITECTURES=OFF setting may be removed after migrating to
    # the findcudatoolkit features in cmake 3.17+
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
        CUDA_ARCHITECTURES OFF)

    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
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
# - katz centrality tests -------------------------------------------------------------------------

set(KATZ_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/centrality/katz_centrality_test.cu")

  ConfigureTest(KATZ_TEST "${KATZ_TEST_SRC}")

###################################################################################################
# - betweenness centrality tests ------------------------------------------------------------------

set(BETWEENNESS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/centrality/betweenness_centrality_test.cu")

  ConfigureTest(BETWEENNESS_TEST "${BETWEENNESS_TEST_SRC}")

set(EDGE_BETWEENNESS_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/centrality/edge_betweenness_centrality_test.cu")

  ConfigureTest(EDGE_BETWEENNESS_TEST "${EDGE_BETWEENNESS_TEST_SRC}")

###################################################################################################
# - SSSP tests ------------------------------------------------------------------------------------

set(SSSP_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/traversal/sssp_test.cu")

ConfigureTest(SSSP_TEST "${SSSP_TEST_SRCS}")

###################################################################################################
# - BFS tests -------------------------------------------------------------------------------------

set(BFS_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/traversal/bfs_test.cu")

ConfigureTest(BFS_TEST "${BFS_TEST_SRCS}")

###################################################################################################
# - LOUVAIN tests ---------------------------------------------------------------------------------

set(LOUVAIN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/louvain_test.cpp")

ConfigureTest(LOUVAIN_TEST "${LOUVAIN_TEST_SRC}")

###################################################################################################
# - LEIDEN tests ---------------------------------------------------------------------------------

set(LEIDEN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/leiden_test.cpp")

ConfigureTest(LEIDEN_TEST "${LEIDEN_TEST_SRC}")

###################################################################################################
# - ECG tests ---------------------------------------------------------------------------------

set(ECG_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/ecg_test.cpp")

ConfigureTest(ECG_TEST "${ECG_TEST_SRC}")

###################################################################################################
# - Balanced cut clustering tests -----------------------------------------------------------------

set(BALANCED_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/balanced_edge_test.cpp")

ConfigureTest(BALANCED_TEST "${BALANCED_TEST_SRC}")

###################################################################################################
# - TRIANGLE tests --------------------------------------------------------------------------------

set(TRIANGLE_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/triangle_test.cu")

ConfigureTest(TRIANGLE_TEST "${TRIANGLE_TEST_SRC}")

###################################################################################################
# - EGO tests --------------------------------------------------------------------------------

set(EGO_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/community/egonet_test.cu")

ConfigureTest(EGO_TEST "${EGO_TEST_SRC}" "")
###################################################################################################
# - RENUMBERING tests -----------------------------------------------------------------------------

set(RENUMBERING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/renumber/renumber_test.cu")

ConfigureTest(RENUMBERING_TEST "${RENUMBERING_TEST_SRC}")

###################################################################################################
# - FORCE ATLAS 2  tests --------------------------------------------------------------------------

set(FA2_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/layout/force_atlas2_test.cu")

ConfigureTest(FA2_TEST "${FA2_TEST_SRC}")

###################################################################################################
# - TSP  tests --------------------------------------------------------------------------

set(TSP_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/traversal/tsp_test.cu")

  ConfigureTest(TSP_TEST "${TSP_TEST_SRC}" "")

###################################################################################################
# - CONNECTED COMPONENTS  tests -------------------------------------------------------------------

set(CONNECT_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/components/con_comp_test.cu")

ConfigureTest(CONNECT_TEST "${CONNECT_TEST_SRC}")

###################################################################################################
# - STRONGLY CONNECTED COMPONENTS  tests ----------------------------------------------------------

set(SCC_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/components/scc_test.cu")

ConfigureTest(SCC_TEST "${SCC_TEST_SRC}")

###################################################################################################
#-Hungarian (Linear Assignment Problem)  tests ---------------------------------------------------------------------

set(HUNGARIAN_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_assignment/hungarian_test.cu")

ConfigureTest(HUNGARIAN_TEST "${HUNGARIAN_TEST_SRC}")

###################################################################################################
# - MST tests ----------------------------------------------------------------------------

set(MST_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/tree/mst_test.cu")

ConfigureTest(MST_TEST "${MST_TEST_SRC}")


###################################################################################################
# - Experimental Graph tests ----------------------------------------------------------------------

set(EXPERIMENTAL_GRAPH_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/graph_test.cpp")

ConfigureTest(EXPERIMENTAL_GRAPH_TEST "${EXPERIMENTAL_GRAPH_TEST_SRCS}")

###################################################################################################
# - Experimental weight-sum tests -----------------------------------------------------------------

set(EXPERIMENTAL_WEIGHT_SUM_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio/mmio.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_utilities.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/weight_sum_test.cpp")

ConfigureTest(EXPERIMENTAL_WEIGHT_SUM_TEST "${EXPERIMENTAL_WEIGHT_SUM_TEST_SRCS}")

###################################################################################################
# - Experimental degree tests ---------------------------------------------------------------------

set(EXPERIMENTAL_DEGREE_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/mmio/mmio.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/utilities/test_utilities.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/degree_test.cpp")

ConfigureTest(EXPERIMENTAL_DEGREE_TEST "${EXPERIMENTAL_DEGREE_TEST_SRCS}")

###################################################################################################
# - Experimental coarsening tests -----------------------------------------------------------------

set(EXPERIMENTAL_COARSEN_GRAPH_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/coarsen_graph_test.cpp")

ConfigureTest(EXPERIMENTAL_COARSEN_GRAPH_TEST "${EXPERIMENTAL_COARSEN_GRAPH_TEST_SRCS}")

###################################################################################################
# - Experimental induced subgraph tests -----------------------------------------------------------

set(EXPERIMENTAL_INDUCED_SUBGRAPH_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/induced_subgraph_test.cpp")

ConfigureTest(EXPERIMENTAL_INDUCED_SUBGRAPH_TEST "${EXPERIMENTAL_INDUCED_SUBGRAPH_TEST_SRCS}")

###################################################################################################
# - Experimental BFS tests ------------------------------------------------------------------------

set(EXPERIMENTAL_BFS_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/bfs_test.cpp")

ConfigureTest(EXPERIMENTAL_BFS_TEST "${EXPERIMENTAL_BFS_TEST_SRCS}")

###################################################################################################
# - Experimental SSSP tests -----------------------------------------------------------------------

set(EXPERIMENTAL_SSSP_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/sssp_test.cpp")

ConfigureTest(EXPERIMENTAL_SSSP_TEST "${EXPERIMENTAL_SSSP_TEST_SRCS}")

###################################################################################################
# - Experimental PAGERANK tests -------------------------------------------------------------------

set(EXPERIMENTAL_PAGERANK_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/pagerank_test.cpp")

ConfigureTest(EXPERIMENTAL_PAGERANK_TEST "${EXPERIMENTAL_PAGERANK_TEST_SRCS}")

###################################################################################################
# - Experimental LOUVAIN tests -------------------------------------------------------------------

set(EXPERIMENTAL_LOUVAIN_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/louvain_test.cu")

ConfigureTest(EXPERIMENTAL_LOUVAIN_TEST "${EXPERIMENTAL_LOUVAIN_TEST_SRCS}")

###################################################################################################
# - Experimental KATZ_CENTRALITY tests ------------------------------------------------------------

set(EXPERIMENTAL_KATZ_CENTRALITY_TEST_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/experimental/katz_centrality_test.cpp")

ConfigureTest(EXPERIMENTAL_KATZ_CENTRALITY_TEST "${EXPERIMENTAL_KATZ_CENTRALITY_TEST_SRCS}")


###################################################################################################
# - MG tests --------------------------------------------------------------------------------------

if(BUILD_CUGRAPH_MG_TESTS)
    if(MPI_CXX_FOUND)
        ###########################################################################################
        # - MG PAGERANK tests ---------------------------------------------------------------------

        set(MG_PAGERANK_TEST_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/pagerank/mg_pagerank_test.cpp")

        ConfigureTest(MG_PAGERANK_TEST "${MG_PAGERANK_TEST_SRCS}")
        target_link_libraries(MG_PAGERANK_TEST PRIVATE MPI::MPI_C MPI::MPI_CXX)

    else(MPI_CXX_FOUND)
       message(FATAL_ERROR "OpenMPI NOT found, cannot build MG tests.")
    endif(MPI_CXX_FOUND)
endif(BUILD_CUGRAPH_MG_TESTS)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
