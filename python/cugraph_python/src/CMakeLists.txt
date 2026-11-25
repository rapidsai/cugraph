# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

nanobind_add_module(
  _native STABLE_ABI
  bindings.cpp
  construction.cu
  traversal.cu
  link_analysis.cu
  sampling.cu
)

target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${DLPACK_INCLUDE_DIR})
target_link_libraries(_native PRIVATE cugraph::cugraph CUDA::cudart)

# Match rapids_cython_create_modules: the extension searches $ORIGIN. libcugraph itself is
# loaded by libcugraph.load_library() before this module is imported.
set_target_properties(
  _native PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON INSTALL_RPATH "$ORIGIN"
)

install(TARGETS _native LIBRARY DESTINATION cugraph_python)
