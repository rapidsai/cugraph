/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#include "utilities/conversion_utilities_impl.cuh"

namespace cugraph {
namespace test {

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<float>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int32_t const>> renumber_map);

template std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::optional<std::vector<double>>>
graph_to_host_csc(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<raft::device_span<int64_t const>> renumber_map);

template std::tuple<cugraph::graph_t<int32_t, int32_t, true, false>,
                    std::optional<cugraph::edge_property_t<int32_t, float>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, false>,
                    std::optional<cugraph::edge_property_t<int64_t, float>>,
                    std::optional<cugraph::edge_property_t<int64_t, int64_t>>,
                    std::optional<cugraph::edge_property_t<int64_t, int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int64_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int32_t, int32_t, true, false>,
                    std::optional<cugraph::edge_property_t<int32_t, double>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, false>,
                    std::optional<cugraph::edge_property_t<int64_t, double>>,
                    std::optional<cugraph::edge_property_t<int64_t, int64_t>>,
                    std::optional<cugraph::edge_property_t<int64_t, int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int64_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int32_t, int32_t, false, false>,
                    std::optional<cugraph::edge_property_t<int32_t, float>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, false>,
                    std::optional<cugraph::edge_property_t<int64_t, float>>,
                    std::optional<cugraph::edge_property_t<int64_t, int64_t>>,
                    std::optional<cugraph::edge_property_t<int64_t, int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, float const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int64_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int32_t, int32_t, false, false>,
                    std::optional<cugraph::edge_property_t<int32_t, double>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<cugraph::edge_property_t<int32_t, int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int32_t const>> renumber_map,
  bool renumber);

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, false>,
                    std::optional<cugraph::edge_property_t<int64_t, double>>,
                    std::optional<cugraph::edge_property_t<int64_t, int64_t>>,
                    std::optional<cugraph::edge_property_t<int64_t, int32_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<int64_t, double const*>> edge_weight_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int64_t const*>> edge_id_view,
  std::optional<cugraph::edge_property_view_t<int64_t, int32_t const*>> edge_type_view,
  std::optional<raft::device_span<int64_t const>> renumber_map,
  bool renumber);

template std::tuple<std::optional<rmm::device_uvector<int32_t>>, rmm::device_uvector<int32_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int32_t const>> mg_renumber_map,
  std::tuple<int32_t, int32_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int32_t const>> sg_renumber_map,
  std::optional<raft::device_span<int32_t const>> mg_vertices,
  raft::device_span<int32_t const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int32_t>>, rmm::device_uvector<int64_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int32_t const>> mg_renumber_map,
  std::tuple<int32_t, int32_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int32_t const>> sg_renumber_map,
  std::optional<raft::device_span<int32_t const>> mg_vertices,
  raft::device_span<int64_t const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int32_t>>, rmm::device_uvector<float>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int32_t const>> mg_renumber_map,
  std::tuple<int32_t, int32_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int32_t const>> sg_renumber_map,
  std::optional<raft::device_span<int32_t const>> mg_vertices,
  raft::device_span<float const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int32_t>>, rmm::device_uvector<double>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int32_t const>> mg_renumber_map,
  std::tuple<int32_t, int32_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int32_t const>> sg_renumber_map,
  std::optional<raft::device_span<int32_t const>> mg_vertices,
  raft::device_span<double const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>, rmm::device_uvector<int32_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int64_t const>> mg_renumber_map,
  std::tuple<int64_t, int64_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int64_t const>> sg_renumber_map,
  std::optional<raft::device_span<int64_t const>> mg_vertices,
  raft::device_span<int32_t const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>, rmm::device_uvector<int64_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int64_t const>> mg_renumber_map,
  std::tuple<int64_t, int64_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int64_t const>> sg_renumber_map,
  std::optional<raft::device_span<int64_t const>> mg_vertices,
  raft::device_span<int64_t const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>, rmm::device_uvector<float>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int64_t const>> mg_renumber_map,
  std::tuple<int64_t, int64_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int64_t const>> sg_renumber_map,
  std::optional<raft::device_span<int64_t const>> mg_vertices,
  raft::device_span<float const> mg_values);

template std::tuple<std::optional<rmm::device_uvector<int64_t>>, rmm::device_uvector<double>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<int64_t const>> mg_renumber_map,
  std::tuple<int64_t, int64_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<int64_t const>> sg_renumber_map,
  std::optional<raft::device_span<int64_t const>> mg_vertices,
  raft::device_span<double const> mg_values);

}  // namespace test
}  // namespace cugraph
