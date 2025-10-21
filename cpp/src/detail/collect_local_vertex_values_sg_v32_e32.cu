/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/collect_local_vertex_values.cuh"

#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/functional>

namespace cugraph {
namespace detail {

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, float, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<int32_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  int32_t default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, double, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
