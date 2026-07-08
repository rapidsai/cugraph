# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# We always need CUDA for cuML because the raft dependency brings in a
# header-only cuco dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(pylibcugraph-python)

project(
  pylibcugraph-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

find_package(cugraph "${RAPIDS_VERSION}" REQUIRED)

# an installed version of libcugraph doesn't provide the dlpack headers so we
# need to download dlpack for the DLPack interop used by utils.pyx
include(rapids-cpm)
rapids_cpm_init()
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(pylibcugraph)
