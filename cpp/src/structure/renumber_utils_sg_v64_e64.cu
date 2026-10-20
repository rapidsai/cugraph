/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "structure/renumber_utils_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT void unrenumber_local_int_vertices<int64_t>(
  raft::handle_t const& handle,
  int64_t* vertices,
  size_t num_vertices,
  int64_t const* renumber_map_labels,
  int64_t local_int_vertex_first,
  int64_t local_int_vertex_last,
  bool do_expensive_check);

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int64_t, false, false>(
  raft::handle_t const& handle,
  int64_t* edgelist_srcs /* [INOUT] */,
  int64_t* edgelist_dsts /* [INOUT] */,
  size_t num_edgelist_edges,
  int64_t const* renumber_map_labels,
  int64_t num_vertices,
  bool do_expensive_check);

template CUGRAPH_EXPORT void unrenumber_local_int_edges<int64_t, true, false>(
  raft::handle_t const& handle,
  int64_t* edgelist_srcs /* [INOUT] */,
  int64_t* edgelist_dsts /* [INOUT] */,
  size_t num_edgelist_edges,
  int64_t const* renumber_map_labels,
  int64_t num_vertices,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  rmm::device_uvector<int64_t>&& vertices,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<int64_t> unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  rmm::device_uvector<int64_t>&& vertices,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map_labels,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
unrenumber_local_int_edges(raft::handle_t const& handle,
                           graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                           rmm::device_uvector<int64_t>&& edgelist_srcs,
                           rmm::device_uvector<int64_t>&& edgelist_dsts,
                           vertex_property_view_t<int64_t, int64_t const*> renumber_map,
                           bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
unrenumber_local_int_edges(raft::handle_t const& handle,
                           graph_view_t<int64_t, int64_t, true, false> const& graph_view,
                           rmm::device_uvector<int64_t>&& edgelist_srcs,
                           rmm::device_uvector<int64_t>&& edgelist_dsts,
                           vertex_property_view_t<int64_t, int64_t const*> renumber_map,
                           bool do_expensive_check);

}  // namespace cugraph
