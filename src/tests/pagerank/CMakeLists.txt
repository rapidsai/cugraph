#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(pagerank_test_SRCS
	../mmio.c
    pagerank_test.cu
)

configure_test(pagerank_test "${pagerank_test_SRCS}")
