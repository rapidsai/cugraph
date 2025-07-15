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

#include "gather_one_hop_impl.cuh"

namespace cugraph {
namespace detail {

using vertex_t = int64_t;
using edge_t   = int64_t;
constexpr bool multi_gpu{false};

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<uint8_t const>> gather_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  edge_property_view_t<edge_t, int32_t const*> edge_time_view,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<int32_t const> active_major_times,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<uint8_t const>> gather_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    std::vector<arithmetic_device_uvector_t>,
                    std::optional<rmm::device_uvector<int32_t>>>
temporal_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  edge_property_view_t<edge_t, int64_t const*> edge_time_view,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<int64_t const> active_major_times,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<uint8_t const>> gather_flags,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
