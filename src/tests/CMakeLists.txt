
#pass the dependency libraries as optional arguments using ${ARGN}
#NOTE the order of libraries matter, so try to link first with the most high level lib
function(configure_test TEST_NAME Tests_SRCS)
#    message(STATUS "${TEST_NAME} will link against: gdf, cugraph")
    cuda_add_executable(${TEST_NAME} ${Tests_SRCS})
    target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest gdf cugraph)
    if(NVG_PLUGIN)
        if(NOT TARGET nvgraph)
            add_library( nvgraph SHARED IMPORTED)
            set_target_properties( nvgraph PROPERTIES IMPORTED_LOCATION /usr/local/lib/libnvgraph_st.so)
            include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../../plugin/nvgraph")
        endif()
        include_directories(
            SYSTEM "/usr/local/include/nvgraph"
        )
        target_link_libraries(${TEST_NAME} nvgraph)
        target_link_libraries(${TEST_NAME} nvggdf)
    endif()
    #target_link_libraries(${TEST_NAME} gmock_main gmock GTest::GTest cugraph)
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests/")

    #register the target as CMake test so we can use ctest for this executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

## Main ##
enable_testing()

# Configure unit tests with Google Test
# -------------------
message(STATUS "******** Configuring tests ********")
include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CUDA_INCLUDE_DIRS}"
)

add_subdirectory(gdf_graph)
add_subdirectory(pagerank)
add_subdirectory(grmat)

if(NVG_PLUGIN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nvgraph_plugin)
endif()

message(STATUS "******** Tests are ready ********")
