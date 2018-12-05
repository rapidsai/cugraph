#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(sssp_test_SRCS
	../mmio.c
	nvgraph_gdf_sssp.cpp
)
configure_test(sssp_test "${sssp_test_SRCS}")

set(louvain_test_SRCS
	../mmio.c
	nvgraph_gdf_louvain.cpp
)
configure_test(louvain_test "${louvain_test_SRCS}")

set(jaccard_test_SRCS
	../mmio.c
	nvgraph_gdf_jaccard.cpp

)
configure_test(jaccard_test "${jaccard_test_SRCS}")


	
