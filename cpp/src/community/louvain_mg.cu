/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <community/louvain_impl.cuh>

namespace cugraph {

// Explicit template instantations

template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, float> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, float> louvain(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  size_t,
  float);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int32_t>>, double> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  size_t,
  double);
template std::pair<std::unique_ptr<Dendrogram<int64_t>>, double> louvain(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  size_t,
  double);

template std::pair<size_t, float> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, float const*>>,
  int32_t*,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, true> const&,
  std::optional<edge_property_view_t<int32_t, double const*>>,
  int32_t*,
  size_t,
  double);
template std::pair<size_t, float> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  int32_t*,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const&,
  graph_view_t<int32_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int32_t*,
  size_t,
  double);
template std::pair<size_t, float> louvain(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, float const*>>,
  int64_t*,
  size_t,
  float);
template std::pair<size_t, double> louvain(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, true> const&,
  std::optional<edge_property_view_t<int64_t, double const*>>,
  int64_t*,
  size_t,
  double);

}  // namespace cugraph
