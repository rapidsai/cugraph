#=============================================================================
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
#=============================================================================
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(NV_GRAPH VERSION 0.4.0 LANGUAGES C CXX CUDA)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
endif(CMAKE_COMPILER_IS_GNUCXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")

# set warnings as errors
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call -Xcompiler -Wall,-Werror")

# set default build type
set(CMAKE_BUILD_TYPE "Release")

option(BUILD_TESTS "Configure CMake to build tests"
       ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

    option(CMAKE_CXX11_ABI "Enable the GLIBCXX11 ABI" OFF)
    if(CMAKE_CXX11_ABI)
        message(STATUS "nvGraph: Enabling the GLIBCXX11 ABI")
    else()
        message(STATUS "nvGraph: Disabling the GLIBCXX11 ABI")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0")
    endif(CMAKE_CXX11_ABI)
endif(CMAKE_COMPILER_IS_GNUCXX)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories(${GTEST_INCLUDE_DIR})
        add_subdirectory(${CMAKE_SOURCE_DIR}/tests)
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories(
                    "${CMAKE_BINARY_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/include"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cub"
                    "${CMAKE_SOURCE_DIR}/thirdparty/cnmem/include"
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
# - library targets -------------------------------------------------------------------------------
if(NVGRAPH_LIGHT MATCHES True)
    add_library(nvgraph_rapids SHARED
                thirdparty/cnmem/src/cnmem.cpp
                src/arnoldi.cu
                src/bfs.cu
                src/bfs2d.cu
                src/bfs_kernels.cu
                src/convert.cu
                src/csrmv.cu
                src/csrmv_cub.cu
                src/csr_graph.cpp
                src/graph_extractor.cu
                src/jaccard_gpu.cu
                src/kmeans.cu
                src/lanczos.cu
                src/lobpcg.cu
                src/matrix.cu
                src/modularity_maximization.cu
                src/nvgraph.cu
                src/nvgraph_cusparse.cpp
                src/nvgraph_cublas.cpp
                src/nvgraph_error.cu
                src/nvgraph_lapack.cu
                src/nvgraph_vector_kernels.cu
                src/pagerank.cu
                src/pagerank_kernels.cu
                src/partition.cu
                src/size2_selector.cu
                src/sssp.cu
                src/triangles_counting.cpp
                src/triangles_counting_kernels.cu
                src/valued_csr_graph.cpp
                src/widest_path.cu
               )
else(NVGRAPH_LIGHT MATCHES True)
        add_library(nvgraph_rapids SHARED
                thirdparty/cnmem/src/cnmem.cpp
                src/arnoldi.cu
                src/bfs.cu
                src/bfs2d.cu
                src/bfs_kernels.cu
                src/convert.cu
                src/csrmv.cu
                src/csrmv_cub.cu
                src/csr_graph.cpp
                src/graph_extractor.cu
                src/jaccard_gpu.cu
                src/kmeans.cu
                src/lanczos.cu
                src/lobpcg.cu
                src/matrix.cu
                src/modularity_maximization.cu
                src/nvgraph.cu
                src/nvgraph_cusparse.cpp
                src/nvgraph_cublas.cpp
                src/nvgraph_error.cu
                src/nvgraph_lapack.cu
                src/nvgraph_vector_kernels.cu
                src/pagerank.cu
                src/pagerank_kernels.cu
                src/partition.cu
                src/size2_selector.cu
                src/sssp.cu
                src/triangles_counting.cpp
                src/triangles_counting_kernels.cu
                src/valued_csr_graph.cpp
                src/widest_path.cu
                src/graph_contraction/contraction_csr_max.cu
                src/graph_contraction/contraction_csr_sum.cu
                src/graph_contraction/contraction_mv_double_mul.cu
                src/graph_contraction/contraction_mv_float_min.cu
                src/graph_contraction/contraction_csr_min.cu
                src/graph_contraction/contraction_mv_double_max.cu
                src/graph_contraction/contraction_mv_double_sum.cu
                src/graph_contraction/contraction_mv_float_mul.cu
                src/graph_contraction/contraction_csr_mul.cu
                src/graph_contraction/contraction_mv_double_min.cu
                src/graph_contraction/contraction_mv_float_max.cu
                src/graph_contraction/contraction_mv_float_sum.cu
               )
endif(NVGRAPH_LIGHT MATCHES True)

###################################################################################################
# - build options ---------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

if(NVGRAPH_LIGHT MATCHES True)
    add_definitions( -DNVGRAPH_LIGHT=${NVGRAPH_LIGHT} )
endif(NVGRAPH_LIGHT MATCHES True)


###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(nvgraph_rapids cublas cusparse curand cusolver cudart )

###################################################################################################
# - install targets -------------------------------------------------------------------------------

install(TARGETS nvgraph_rapids
        DESTINATION lib)

install(FILES
         ${CMAKE_CURRENT_SOURCE_DIR}/include/nvgraph.h
         ${CMAKE_CURRENT_SOURCE_DIR}/include/test_opt_utils.cuh
        DESTINATION include/nvgraph)


