/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <community/leiden_impl.cuh>

namespace cugraph {

// SG instantiation

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  size_t max_level,
  float resolution,
  float theta);

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
  size_t max_level,
  double resolution,
  double theta);

template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         raft::random::RngState&,
                                         graph_view_t<int32_t, int32_t, false, true> const&,
                                         std::optional<edge_property_view_t<int32_t, float const*>>,
                                         int32_t*,
                                         size_t,
                                         float,
                                         float);
template std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  int32_t*,
  size_t,
  double,
  double);
template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         raft::random::RngState&,
                                         graph_view_t<int32_t, int64_t, false, true> const&,
                                         std::optional<edge_property_view_t<int64_t, float const*>>,
                                         int32_t*,
                                         size_t,
                                         float,
                                         float);
template std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int32_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int32_t*,
  size_t,
  double,
  double);
template std::pair<size_t, float> leiden(raft::handle_t const&,
                                         raft::random::RngState&,
                                         graph_view_t<int64_t, int64_t, false, true> const&,
                                         std::optional<edge_property_view_t<int64_t, float const*>>,
                                         int64_t*,
                                         size_t,
                                         float,
                                         float);
template std::pair<size_t, double> leiden(
  raft::handle_t const&,
  raft::random::RngState&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int64_t*,
  size_t,
  double,
  double);

}  // namespace cugraph
