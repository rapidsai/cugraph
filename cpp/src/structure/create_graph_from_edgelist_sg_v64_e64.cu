/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/create_graph_from_edgelist_impl.cuh"

namespace cugraph {

// explicit instantiations

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, false>,
                    std::vector<edge_arithmetic_property_t<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::vector<arithmetic_device_uvector_t>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, false>,
                    std::vector<edge_arithmetic_property_t<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertices,
  rmm::device_uvector<int64_t>&& edgelist_srcs,
  rmm::device_uvector<int64_t>&& edgelist_dsts,
  std::vector<arithmetic_device_uvector_t>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, false, false>,
                    std::vector<edge_arithmetic_property_t<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, false, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertices,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::vector<std::vector<arithmetic_device_uvector_t>>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

template std::tuple<cugraph::graph_t<int64_t, int64_t, true, false>,
                    std::vector<edge_arithmetic_property_t<int64_t>>,
                    std::optional<rmm::device_uvector<int64_t>>>
create_graph_from_edgelist<int64_t, int64_t, true, false>(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<int64_t>>&& vertices,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_srcs,
  std::vector<rmm::device_uvector<int64_t>>&& edgelist_dsts,
  std::vector<std::vector<arithmetic_device_uvector_t>>&& edgelist_edge_properties,
  graph_properties_t graph_properties,
  bool renumber,
  std::optional<large_buffer_type_t> large_vertex_buffer_type,
  std::optional<large_buffer_type_t> large_edge_buffer_type,
  bool do_expensive_check);

}  // namespace cugraph
