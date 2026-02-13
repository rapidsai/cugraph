# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
set(CPM_DOWNLOAD_VERSION v0.38.5)
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/${CPM_DOWNLOAD_VERSION}/get_cpm.cmake
  ${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake
)
include(${CMAKE_BINARY_DIR}/cmake/get_cpm.cmake)

# find or build it via CPM
include(rapids-cpm)
CPMFindPackage(
  NAME cugraph
  VERSION ${RAPIDS_VERSION}
  FIND_PACKAGE_ARGUMENTS "PATHS ${cugraph_ROOT} ${cugraph_ROOT}/latest" GIT_REPOSITORY
                         https://github.com/rapidsai/cugraph
  GIT_TAG ${rapids-cmake-branch}
  GIT_SHALLOW
    TRUE
    SOURCE_SUBDIR
    cpp
)
