/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "community/edge_triangle_count_impl.cuh"

namespace cugraph {

// SG instantiation
template edge_property_t<int32_t, int32_t> edge_triangle_count(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  bool do_expensive_check);

}  // namespace cugraph
