# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources
    sampling_result.pyx
    coo.pyx
    edge_id_lookup_result.pyx
)
set(linked_libraries cugraph::cugraph;cugraph::cugraph_c)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}"
  ASSOCIATED_TARGETS cugraph
)
