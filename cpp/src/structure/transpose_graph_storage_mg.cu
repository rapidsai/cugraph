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
#include <structure/transpose_graph_storage_impl.cuh>

namespace cugraph {

// MG instantiation

template std::tuple<
  graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>&& edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int32_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>&& edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int32_t, int64_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int32_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, true, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>&& edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
transpose_graph_storage(
  raft::handle_t const& handle,
  graph_t<int64_t, int64_t, false, true>&& graph,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>&&
    edge_weights,
  std::optional<rmm::device_uvector<int64_t>>&& renumber_map,
  bool do_expensive_check);

}  // namespace cugraph
