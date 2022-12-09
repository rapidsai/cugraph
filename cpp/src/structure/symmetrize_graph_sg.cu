/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <structure/symmetrize_graph_impl.cuh>

namespace cugraph {

// SG instantiation

template std::tuple<
  graph_t<int32_t, int32_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, float>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, float>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, float>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, false>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, true, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, false>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, false, false>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, false>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check);

}  // namespace cugraph
