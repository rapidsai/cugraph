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
#include <structure/create_graph_from_edgelist_impl.cuh>

namespace cugraph {

// explicit instantiations

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>,
                                         thrust::tuple<int32_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, float, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>,
                                         thrust::tuple<int32_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, float, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, false, false>,
                                         thrust::tuple<int32_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, double, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int32_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int32_t, true, false>,
                                         thrust::tuple<int32_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, double, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, float, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, float, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, false, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, double, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int32_t, int64_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int32_t, int64_t, true, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int64_t, double, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertex_span,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, float, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>, float>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, float, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<float>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, false, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, false, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, double, int32_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

template std::tuple<
  cugraph::graph_t<int64_t, int64_t, true, false>,
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>, double>>,
  std::optional<cugraph::edge_property_t<cugraph::graph_view_t<int64_t, int64_t, true, false>,
                                         thrust::tuple<int64_t, int32_t>>>,
  std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, double, int32_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertex_span,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<double>>&& edgelist_weights,
  std::optional<std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>>&&
    edgelist_id_type_pairs,
  graph_properties_t graph_properties,
  bool renumber,
  bool do_expensive_check);

}  // namespace cugraph
