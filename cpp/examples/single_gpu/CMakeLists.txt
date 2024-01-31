# Copyright (c) 2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.23.1)

project(
  sg_examples
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(../fetch_dependencies.cmake)

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
target_link_libraries(csvfileutil PUBLIC cugraph::cugraph)

# Configure your project here
add_executable(sg_examples sg_examples.cpp)
target_link_libraries(sg_examples PRIVATE csvfileutil)
target_compile_features(sg_examples PRIVATE cxx_std_17)
