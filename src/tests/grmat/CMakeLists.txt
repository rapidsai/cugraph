#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(grmat_test_SRCS
	../mmio.c
    grmat_test.cu
)

configure_test(grmat_test "${grmat_test_SRCS}")
