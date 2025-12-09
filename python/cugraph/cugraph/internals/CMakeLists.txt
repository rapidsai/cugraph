# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources internals.pyx)
set(linked_libraries cugraph::cugraph)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX internals_
  ASSOCIATED_TARGETS cugraph
)

target_include_directories(internals_internals PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
