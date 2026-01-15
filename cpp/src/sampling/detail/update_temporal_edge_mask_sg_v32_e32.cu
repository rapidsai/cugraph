/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/update_temporal_edge_mask_impl.cuh"

namespace cugraph {
namespace detail {

template void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, int32_t const*> edge_start_time_view,
  raft::device_span<int32_t const> vertices,
  raft::device_span<int32_t const> vertex_times,
  edge_property_view_t<int32_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison,
  std::optional<int32_t> window_start,
  std::optional<int32_t> window_end);

template void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, int64_t const*> edge_start_time_view,
  raft::device_span<int32_t const> vertices,
  raft::device_span<int64_t const> vertex_times,
  edge_property_view_t<int32_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison,
  std::optional<int64_t> window_start,
  std::optional<int64_t> window_end);

}  // namespace detail
}  // namespace cugraph
