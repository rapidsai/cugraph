# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# We always need CUDA for cuML because the raft dependency brings in a
# header-only cuco dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cugraph-python)

project(
  cugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

find_package(cugraph "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(cugraph/components)
add_subdirectory(cugraph/dask/structure)
add_subdirectory(cugraph/internals)
add_subdirectory(cugraph/linear_assignment)
add_subdirectory(cugraph/structure)
add_subdirectory(cugraph/tree)
add_subdirectory(cugraph/utilities)
