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

    target_include_directories(${CMAKE_TEST_NAME}
        PRIVATE
        "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
        "${GTEST_INCLUDE_DIR}"
        "${RMM_INCLUDE}"
        "${CUDF_INCLUDE}"
        "${CMAKE_SOURCE_DIR}/../thirdparty/cub"
        "${CMAKE_SOURCE_DIR}/../thirdparty/mmio"
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${CMAKE_TEST_NAME}
        PRIVATE
        gtest gmock_main gmock cugraph ${CUDF_LIBRARY} ${RMM_LIBRARY} cudart cuda)
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
###   environment. If a full file path is provided (e.g ${CUDF_LIBRARY} and ${RMM_LIBRARY}), cmake
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
###   path included in BUILD_RPATH when ${CUDF_LIBRARY} and ${RMM_LIBRARY} are added; this one can
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

ConfigureTest(SNMG_DEGREE_TEST "${SNMG_DEGREE_TEST_SRC}")

set(SNMG_PAGERANK_TEST_SRC
    "${CMAKE_SOURCE_DIR}/../thirdparty/mmio/mmio.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/snmg_pagerank/snmg_pagerank_test.cu")

ConfigureTest(SNMG_PAGERANK_TEST "${SNMG_PAGERANK_TEST_SRC}")

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()
