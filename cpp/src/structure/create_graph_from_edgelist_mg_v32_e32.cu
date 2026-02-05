/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "create_graph_from_edgelist_impl.cuh"

namespace cugraph {

// explicit instantiations

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::vector<edge_arithmetic_property_t<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, false, true>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::vector<arithmetic_device_uvector_t>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::vector<edge_arithmetic_property_t<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, true, true>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertices,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::vector<arithmetic_device_uvector_t>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, false, true>,
                    std::vector<edge_arithmetic_property_t<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, false, true>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertices,
  std::vector<rmm::device_uvector<int32_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int32_t>>&& edgelist_dsts,
  std::vector<std::vector<arithmetic_device_uvector_t>>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<graph_t<int32_t, int32_t, true, true>,
                    std::vector<edge_arithmetic_property_t<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>>
create_graph_from_edgelist<int32_t, int32_t, true, true>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int32_t>>&& vertices,
  std::vector<rmm::device_uvector<int32_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int32_t>>&& edgelist_dsts,
  std::vector<std::vector<arithmetic_device_uvector_t>>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

}  // namespace cugraph
