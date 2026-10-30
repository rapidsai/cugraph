/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT void unrenumber_local_int_vertices<int32_t>(
  raft::handle_t const& handle,
  int32_t* vertices,
  size_t num_vertices,
  int32_t const* renumber_map_labels,
  int32_t local_int_vertex_first,
  int32_t local_int_vertex_last,
  bool do_expensive_check);

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int32_t, false, false>(
  raft::handle_t const& handle,
  int32_t* edgelist_srcs /* [INOUT] */,
  int32_t* edgelist_dsts /* [INOUT] */,
  size_t num_edgelist_edges,
  int32_t const* renumber_map_labels,
  int32_t num_vertices,
  bool do_expensive_check);

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int32_t, true, false>(
  raft::handle_t const& handle,
  int32_t* edgelist_srcs /* [INOUT] */,
  int32_t* edgelist_dsts /* [INOUT] */,
  size_t num_edgelist_edges,
  int32_t const* renumber_map_labels,
  int32_t num_vertices,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
unrenumber_local_int_edges(raft::handle_t const& handle,
                           graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                           rmm::device_uvector<int32_t>&& edgelist_srcs,
                           rmm::device_uvector<int32_t>&& edgelist_dsts,
                           vertex_property_view_t<int32_t, int32_t const*> renumber_map,
                           bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
unrenumber_local_int_edges(raft::handle_t const& handle,
                           graph_view_t<int32_t, int32_t, true, false> const& graph_view,
                           rmm::device_uvector<int32_t>&& edgelist_srcs,
                           rmm::device_uvector<int32_t>&& edgelist_dsts,
                           vertex_property_view_t<int32_t, int32_t const*> renumber_map,
                           bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> renumber_ext_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> renumber_ext_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> renumber_local_ext_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> renumber_local_ext_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> unrenumber_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int32_t> unrenumber_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& vertices,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map_labels,
  bool do_expensive_check);

}  // namespace cugraph
