/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dag/topological_sort_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> topological_sort(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int64_t, int64_t> topological_sort_level(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  bool do_expensive_check);

}  // namespace cugraph
