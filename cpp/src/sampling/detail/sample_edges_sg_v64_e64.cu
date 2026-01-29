/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/sample_edges.cuh"

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
sample_edges(raft::handle_t const& handle,
             raft::random::RngState& rng_state,
             graph_view_t<int64_t, int64_t, false, false> const& graph_view,
             raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views,
             std::optional<edge_arithmetic_property_view_t<int64_t>> edge_type_view,
             std::optional<edge_arithmetic_property_view_t<int64_t>> edge_bias_view,
             raft::device_span<int64_t const> active_majors,
             std::optional<raft::device_span<int32_t const>> active_major_labels,
             raft::host_span<size_t const> Ks,
             bool with_replacement);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
sample_edges_with_visited(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<int64_t>> edge_property_views,
  std::optional<edge_arithmetic_property_view_t<int64_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<int64_t>> edge_bias_view,
  raft::device_span<int64_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  raft::host_span<size_t const> Ks,
  std::optional<rmm::device_uvector<int64_t>>& visited_vertices,
  std::optional<rmm::device_uvector<int32_t>>& visited_vertex_labels,
  bool with_replacement);

}  // namespace detail
}  // namespace cugraph
