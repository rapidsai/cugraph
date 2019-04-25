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
#pass the dependency aibraries as optional arguments using ${ARGN}

#NOTE the order of libraries matter, so try to link first with the most high level lib

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUGRAPH_TESTS LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "-shared -cudart shared")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -Xcompiler -rdynamic -lineinfo")

if(CMAKE_CXX11_ABI)
    message(STATUS "cuGraph test: Enabling the GLIBCXX11 ABI")
else()
    message(STATUS "cuGraph test: Disabling the GLIBCXX11 ABI")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
endif(CMAKE_CXX11_ABI)

if(RAPIDS_DATASET_ROOT_DIR)
    message(STATUS "setting default RAPIDS_DATASET_ROOT_DIR to: ${RAPIDS_DATASET_ROOT_DIR}")
    string(CONCAT CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
    string(CONCAT CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} " -DRAPIDS_DATASET_ROOT_DIR=" "\\\"" ${RAPIDS_DATASET_ROOT_DIR} "\\\"")
endif(RAPIDS_DATASET_ROOT_DIR)

# - compiler function -----------------------------------------------------------------------------
function(configure_test TEST_NAME Tests_SRCS)
#    message(STATUS "${TEST_NAME} will link against: gdf, cugraph")

    add_executable(${TEST_NAME} ${Tests_SRCS})
    target_link_libraries(${TEST_NAME} OpenMP::OpenMP_CXX gmock_main gmock GTest::GTest cudart cudf cugraph nvgraph)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    #register the target as CMake test so we can use ctest for this executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${OpenMP_CXX_FLAGS} --expt-extended-lambda")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Configure unit tests with Google Test
# ------------------
###################################################################################################
# - include paths ----------------------------------------------------------------------------------
message(STATUS "******** Configuring tests ********")
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/../thirdparty/mmio"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

###################################################################################################
#- gdfgraph tests -- ---------------------------------------------------------------------------------
set(GDFGRAPH_TEST_SRCS
	"${CMAKE_CURRENT_SOURCE_DIR}/gdf_graph/gdf_graph.cu")

configure_test(GDFGRAPH_TEST "${GDFGRAPH_TEST_SRCS}")

###################################################################################################
#- grmat tests -- ---------------------------------------------------------------------------------
set(GRMAT_TEST_SRCS
	"${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/grmat/grmat_test.cu")

configure_test(GRMAT_TEST "${GRMAT_TEST_SRCS}")

###################################################################################################
#- pagerank tests -- ---------------------------------------------------------------------------------
set(PAGERANK_TEST_SRCS
	"${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
	"${CMAKE_CURRENT_SOURCE_DIR}/pagerank/pagerank_test.cu")

configure_test(PAGERANK_TEST "${PAGERANK_TEST_SRCS}")

###################################################################################################
#-SSSP tests -- ---------------------------------------------------------------------------------
set(SSSP_TEST_SRCS
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_sssp.cpp")

configure_test(SSSP_TEST "${SSSP_TEST_SRCS}")

###################################################################################################
#-LOUVAIN tests -- ---------------------------------------------------------------------------------
set(LOUVAIN_TEST_SRCS
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_louvain.cpp")

configure_test(LOUVAIN_TEST "${LOUVAIN_TEST_SRCS}")

###################################################################################################
#-JACCARD  tests -- ---------------------------------------------------------------------------------
set(JACCARD_TEST_SRCS
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin/nvgraph_gdf_jaccard.cpp")

configure_test(JACCARD_TEST "${JACCARD_TEST_SRCS}")

###################################################################################################
#-RENUMBERING  tests -- ---------------------------------------------------------------------------------
set(RENUMBERING_TEST_SRCS
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/renumber/renumber_test.cu")

configure_test(RENUMBERING_TEST "${RENUMBERING_TEST_SRCS}")

set(SNMG_SPMV_TEST_SRCS
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/snmg_spmv/snmg_spmv_test.cu")

configure_test(SNMG_SPMV_TEST "${SNMG_SPMV_TEST_SRCS}")


message(STATUS "******** Tests are ready ********")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
