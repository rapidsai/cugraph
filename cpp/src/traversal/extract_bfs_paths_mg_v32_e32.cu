/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/extract_bfs_paths_impl.cuh"

namespace cugraph {

// MG instantiation

template std::tuple<rmm::device_uvector<int32_t>, int32_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  int32_t const* distances,
  int32_t const* predecessors,
  int32_t const* destinations,
  size_t n_destinations);

}  // namespace cugraph
