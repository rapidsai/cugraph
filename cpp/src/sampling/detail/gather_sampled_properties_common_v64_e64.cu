/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/gather_sampled_properties_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace detail {

using vertex_t = int64_t;
using edge_t   = int64_t;

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,
                                   rmm::device_uvector<vertex_t>,
                                   std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<vertex_t>,
                                   rmm::device_uvector<vertex_t>,
                                   std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, true> const& graph_view,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph
