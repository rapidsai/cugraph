# Makefile for building NVCompute/CUDA BLAS library
SOLNDIR  := ../.

# Get the profile settings
ifdef VULCAN
include $(VULCAN_TOOLKIT_BASE)/build/getprofile.mk
include $(VULCAN_TOOLKIT_BASE)/build/config/$(PROFILE).mk
include $(VULCAN_TOOLKIT_BASE)/build/config/DetectOS.mk
else
include ../../build/getprofile.mk
include ../../build/config/$(PROFILE).mk
include ../../build/config/DetectOS.mk
endif

export I_AM_SLOPPY = 1
AGNOSTIC_PROJECTS += nvgraph_test 
AGNOSTIC_PROJECTS += nvgraph_capi_tests 
AGNOSTIC_PROJECTS += nvgraph_capi_tests_subgraph
AGNOSTIC_PROJECTS += nvgraph_capi_tests_conversion
AGNOSTIC_PROJECTS += nvgraph_benchmark
AGNOSTIC_PROJECTS += nvgraph_capi_tests_clustering
AGNOSTIC_PROJECTS += nvgraph_capi_tests_contraction
AGNOSTIC_PROJECTS += nvgraph_capi_tests_traversal
AGNOSTIC_PROJECTS += nvgraph_capi_tests_triangles
AGNOSTIC_PROJECTS += nvgraph_2d_partitioning_test
AGNOSTIC_PROJECTS += nvgraph_capi_tests_2d_bfs
AGNOSTIC_PROJECTS += nvgraph_capi_tests_2d_bfs_net

ifdef VULCAN
include $(VULCAN_TOOLKIT_BASE)/build/common.mk
else
include ../../build/common.mk
endif
