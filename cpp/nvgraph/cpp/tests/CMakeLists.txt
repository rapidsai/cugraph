cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(CUDF_TESTS LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread nvgraph_rapids cublas cusparse curand cusolver cudart)
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/../external"
                    "${CMAKE_SOURCE_DIR}/../external/cusp"
                    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
                   )

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
                 "${CMAKE_BINARY_DIR}/lib"
                 "${GTEST_LIBRARY_DIR}")

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - triangles tests -------------------------------------------------------------------------------------

set(TRIANGLES_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_triangles.cpp")

ConfigureTest(TRIANGLES_TEST "${TRIANGLES_TEST_SRC}")

###################################################################################################
# - nvgraph tests -------------------------------------------------------------------------------------

set(NVGRAPH_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_test.cpp"
   )

ConfigureTest(NVGRAPH_TEST "${NVGRAPH_TEST_SRC}")

###################################################################################################
# - 2d_partitioning -------------------------------------------------------------------------------

set(2DPARTITIONING_TEST_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/2d_partitioning_test.cpp"
   )

ConfigureTest(2DPARTITIONING_TEST "${2DPARTITIONING_TEST_SRC}")

###################################################################################################
# - nvgraph_benchmark -----------------------------------------------------------------------------

set(NVGRAPH_BENCHMARK_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mmio.c"
   )

ConfigureTest(NVGRAPH_BENCHMARK "${NVGRAPH_BENCHMARK_SRC}")

###################################################################################################
# - nvgraph_capi_tests_2d_bfs ---------------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_2D_BFS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_2d_bfs.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_2D_BFS "${NVGRAPH_CAPI_TESTS_2D_BFS_SRC}")

###################################################################################################
# - nvgraph_capi_tests_2d_bfs_net -----------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_2D_BFS_NET_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_2d_bfs_net.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_2D_BFS_NET "${NVGRAPH_CAPI_TESTS_2D_BFS_NET_SRC}")

###################################################################################################
# - nvgraph_capi_tests_algorithms -----------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_ALGORITHMS_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_algorithms.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_ALGORITHMS "${NVGRAPH_CAPI_TESTS_ALGORITHMS_SRC}")

###################################################################################################
# - nvgraph_capi_tests_clustering -----------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_CLUSTERING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_clustering.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mmio.c"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_CLUSTERING "${NVGRAPH_CAPI_TESTS_CLUSTERING_SRC}")

###################################################################################################
# - nvgraph_capi_tests_contraction ----------------------------------------------------------------
if(NOT NVGRAPH_LIGHT)
set(NVGRAPH_CAPI_TESTS_CONTRACTION_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_contraction.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_CONTRACTION "${NVGRAPH_CAPI_TESTS_CONTRACTION_SRC}")
endif(NOT NVGRAPH_LIGHT)

###################################################################################################
# - nvgraph_capi_test_conversion ------------------------------------------------------------------

set(NVGRAPH_CAPI_TEST_CONVERSION_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_conversion.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TEST_CONVERSION "${NVGRAPH_CAPI_TEST_CONVERSION_SRC}")

###################################################################################################
# - nvgraph_capi_tests_subgraph -------------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_SUBGRAPH_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_subgraph.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_SUBGRAPH "${NVGRAPH_CAPI_TESTS_SUBGRAPH_SRC}")

###################################################################################################
# - nvgraph_capi_tests_traversal ------------------------------------------------------------------

set(NVGRAPH_CAPI_TESTS_TRAVERSAL_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_capi_tests_traversal.cpp"
   )

ConfigureTest(NVGRAPH_CAPI_TESTS_TRAVERSAL "${NVGRAPH_CAPI_TESTS_TRAVERSAL_SRC}")


###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

