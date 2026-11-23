# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.3 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# CUDA is required because the raft dependency brings in a header-only cuco
# dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cugraph_python)

project(
  cugraph_python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

find_package(cugraph "${RAPIDS_VERSION}" REQUIRED)

# An installed libcugraph does not ship the dlpack headers used by the bindings.
include(rapids-cpm)
rapids_cpm_init()
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

find_package(CUDAToolkit REQUIRED)

# Per-version wheels use the CPython stable ABI (wheel.py-api), same as pylibcugraph.
# SABIModule is what nanobind's STABLE_ABI option links against.
find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
if(NOT TARGET Python::SABIModule)
  message(FATAL_ERROR "Python::SABIModule is required to build the per-version stable-ABI wheel")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  COMMAND_ERROR_IS_FATAL ANY
)
find_package(nanobind CONFIG REQUIRED HINTS "${nanobind_ROOT}")

add_subdirectory(src)
