# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.23.1)

project(
  helloworld
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

set(CPM_DOWNLOAD_VERSION v0.35.3)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

set(CUGRAPH_TAG branch-23.12)
CPMFindPackage(
  NAME cugraph GIT_REPOSITORY https://github.com/rapidsai/cugraph
  GIT_TAG ${CUGRAPH_TAG}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../fetch_rapids.cmake)
include(rapids-find)
include(../cmake/thirdparty/get_nccl.cmake)


add_library(csvfileutil STATIC ../tests/utilities/csv_file_utilities.cu)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "NATIVE")
endif()

target_include_directories(csvfileutil
PUBLIC
"../../thirdparty/mmio"
PRIVATE
"../include"
"../tests"
"../src"
)

set_target_properties(csvfileutil PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(csvfileutil PUBLIC cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)

# Configure your project here
add_executable(helloworld helloworld.cpp)
target_link_libraries(helloworld PRIVATE csvfileutil)
target_compile_features(helloworld PRIVATE cxx_std_17)
