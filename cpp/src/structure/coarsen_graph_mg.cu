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
#include <structure/coarsen_graph_impl.cuh>

namespace cugraph {

// MG instantiation

template std::tuple<
  graph_t<int32_t, int32_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, float>>,
  std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int32_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int32_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int32_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int32_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int32_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int32_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int32_t, int64_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int32_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, true, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, true, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, true, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

template std::tuple<
  graph_t<int64_t, int64_t, false, true>,
  std::optional<edge_property_t<graph_view_t<int64_t, int64_t, false, true>, double>>,
  std::optional<rmm::device_uvector<int64_t>>>
coarsen_graph(raft::handle_t const& handle,
              graph_view_t<int64_t, int64_t, false, true> const& graph_view,
              std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
              int64_t const* labels,
              bool renumber,
              bool do_expensive_check);

}  // namespace cugraph
