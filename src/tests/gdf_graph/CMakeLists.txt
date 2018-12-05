#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(gdfgraph_test_SRCS
    gdf_graph.cu
)

configure_test(gdfgraph_test "${gdfgraph_test_SRCS}")
