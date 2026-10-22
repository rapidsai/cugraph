/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/collect_local_vertex_values.cuh"

#include <cugraph/export.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/functional>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map,
  float default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, int32_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, int32_t, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map,
  int32_t default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map,
  double default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map,
  float default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  vertex_property_view_t<int32_t, int32_t const*> renumber_map,
  double default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
