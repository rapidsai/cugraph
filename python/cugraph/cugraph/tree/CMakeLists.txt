# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources minimum_spanning_tree_wrapper.pyx)
set(linked_libraries cugraph::cugraph)
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX tree_
  ASSOCIATED_TARGETS cugraph
)
