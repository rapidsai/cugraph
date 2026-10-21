/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/extract_bfs_paths_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, int64_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  int64_t const* distances,
  int64_t const* predecessors,
  int64_t const* destinations,
  size_t n_destinations);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, int64_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  vertex_property_view_t<int64_t, int64_t const*> distances,
  vertex_property_view_t<int64_t, int64_t const*> predecessors,
  raft::device_span<int64_t const> destinations,
  bool do_expensive_check);

}  // namespace cugraph
