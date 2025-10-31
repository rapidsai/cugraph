# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources
    type_utils.pyx
)
set(linked_libraries cugraph::cugraph;cugraph::cugraph_c)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  ASSOCIATED_TARGETS cugraph
)
