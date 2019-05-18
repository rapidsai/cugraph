#=============================================================================
#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUGRAPH_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME}
                   ${CMAKE_TEST_SRC})
    target_link_libraries(${CMAKE_TEST_NAME} OpenMP::OpenMP_CXX gtest gmock_main gmock cugraph nvgraph cudf)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")
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
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/../thirdparty/mmio"
)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - gdfgraph tests --------------------------------------------------------------------------------

set(GDFGRAPH_TEST_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/gdf_graph/gdf_graph.cu")

ConfigureTest(GDFGRAPH_TEST "${GDFGRAPH_TEST_SRC}")

###################################################################################################
# - grmat tests -----------------------------------------------------------------------------------

set(GRMAT_TEST_SRC
	"${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/grmat/grmat_test.cu")

ConfigureTest(GRMAT_TEST "${GRMAT_TEST_SRC}")

###################################################################################################
# - pagerank tests --------------------------------------------------------------------------------

set(PAGERANK_TEST_SRC
	"${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/pagerank/pagerank_test.cu")

ConfigureTest(PAGERANK_TEST "${PAGERANK_TEST_SRC}")

###################################################################################################
# - SSSP tests ------------------------------------------------------------------------------------

set(SSSP_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_sssp.cpp")

ConfigureTest(SSSP_TEST "${SSSP_TEST_SRC}")

###################################################################################################
# - LOUVAIN tests ---------------------------------------------------------------------------------

set(LOUVAIN_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_louvain.cpp")

ConfigureTest(LOUVAIN_TEST "${LOUVAIN_TEST_SRC}")

###################################################################################################
# - JACCARD tests ---------------------------------------------------------------------------------

set(JACCARD_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_jaccard.cpp")

ConfigureTest(JACCARD_TEST "${JACCARD_TEST_SRC}")

###################################################################################################
# - RENUMBERING tests -----------------------------------------------------------------------------

set(RENUMBERING_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/renumber/renumber_test.cu")

ConfigureTest(RENUMBERING_TEST "${RENUMBERING_TEST_SRC}")

###################################################################################################
# - SNMG SpMV tests -------------------------------------------------------------------------------

set(SNMG_SPMV_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/snmg_spmv/snmg_spmv_test.cu")

ConfigureTest(SNMG_SPMV_TEST "${SNMG_SPMV_TEST_SRC}")

###################################################################################################
#-SNMG_DEGREE  tests ------------------------------------------------------------------------------

set(SNMG_DEGREE_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/snmg_degree/snmg_degree_test.cu")

configure_test(SNMG_DEGREE_TEST "${SNMG_DEGREE_TEST_SRC}")

set(SNMG_PAGERANK_TEST_SRC
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/snmg_pagerank/snmg_pagerank_test.cu")

ConfigureTest(SNMG_PAGERANK_TEST "${SNMG_PAGERANK_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
