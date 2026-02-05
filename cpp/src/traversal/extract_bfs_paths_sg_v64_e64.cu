/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/extract_bfs_paths_impl.cuh"

namespace cugraph {

// SG instantiation

template std::tuple<rmm::device_uvector<int64_t>, int64_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  int64_t const* distances,
  int64_t const* predecessors,
  int64_t const* destinations,
  size_t n_destinations);

}  // namespace cugraph
