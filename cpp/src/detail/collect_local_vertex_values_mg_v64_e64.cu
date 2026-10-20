/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/collect_local_vertex_values.cuh"

#include <cugraph/export.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/functional>

namespace cugraph {
namespace detail {

template CUGRAPH_EXPORT vertex_property_t<int64_t, float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  float default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int64_t, double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  double default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int64_t, int64_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, int64_t, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  int64_t default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int64_t, float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  float default_value,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int64_t, double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  vertex_property_view_t<int64_t, int64_t const*> renumber_map,
  double default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
