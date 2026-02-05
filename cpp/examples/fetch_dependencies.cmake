# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
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

include(../../../../cmake/rapids_config.cmake)
include(rapids-find)
