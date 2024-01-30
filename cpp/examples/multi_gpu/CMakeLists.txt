# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.23.1)

project(
  mg_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

find_package(MPI REQUIRED COMPONENTS CXX)
include(../../cmake/thirdparty/get_nccl.cmake)


add_library(csvfileutil STATIC ../../tests/utilities/csv_file_utilities.cu)

target_include_directories(csvfileutil
PUBLIC
"../../../thirdparty/mmio"
"../../tests"
PRIVATE
"../../include"
"../../src"
)

set_target_properties(csvfileutil PROPERTIES CUDA_ARCHITECTURES "native")
target_link_libraries(csvfileutil PUBLIC cugraph::cugraph NCCL::NCCL MPI::MPI_CXX)

# Configure your project here
add_executable(mg_examples mg_examples.cpp)
target_link_libraries(mg_examples PRIVATE csvfileutil)
target_compile_features(mg_examples PRIVATE cxx_std_17)
