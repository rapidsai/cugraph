/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "traversal/od_shortest_distances_impl.cuh"

namespace cugraph {

// SG instantiation

template rmm::device_uvector<float> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  raft::device_span<int32_t const> origins,
  raft::device_span<int32_t const> destinations,
  float cutoff,
  bool do_expensive_check);

template rmm::device_uvector<double> od_shortest_distances(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  raft::device_span<int32_t const> origins,
  raft::device_span<int32_t const> destinations,
  double cutoff,
  bool do_expensive_check);

}  // namespace cugraph
