/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "sampling/detail/update_temporal_edge_mask_impl.cuh"

namespace cugraph {
namespace detail {

template void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, int32_t const*> edge_start_time_view,
  raft::device_span<int32_t const> vertices,
  raft::device_span<int32_t const> vertex_times,
  edge_property_view_t<int32_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison);

template void update_temporal_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, int64_t const*> edge_start_time_view,
  raft::device_span<int32_t const> vertices,
  raft::device_span<int64_t const> vertex_times,
  edge_property_view_t<int32_t, uint32_t*, bool> edge_time_mask_view,
  temporal_sampling_comparison_t temporal_sampling_comparison);

}  // namespace detail
}  // namespace cugraph
