/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <centrality/betweenness_centrality_impl.cuh>

namespace cugraph {

// MG instantiation

template rmm::device_uvector<float> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<float> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<float> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<std::variant<int64_t, raft::device_span<int64_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<double> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<double> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<double> betweenness_centrality(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<std::variant<int64_t, raft::device_span<int64_t const>>> vertices,
  bool const normalized,
  bool const include_endpoints,
  bool do_expensive_check);

template rmm::device_uvector<float> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

template rmm::device_uvector<float> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

template rmm::device_uvector<float> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<std::variant<int64_t, raft::device_span<int64_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

template rmm::device_uvector<double> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

template rmm::device_uvector<double> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<std::variant<int32_t, raft::device_span<int32_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

template rmm::device_uvector<double> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<std::variant<int64_t, raft::device_span<int64_t const>>> vertices,
  bool const normalized,
  bool const do_expensive_check);

}  // namespace cugraph
