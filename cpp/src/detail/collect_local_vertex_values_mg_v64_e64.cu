/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/collect_local_vertex_values.cuh"
#include "detail/graph_partition_utils.cuh"

#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/functional>

namespace cugraph {
namespace detail {

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

template rmm::device_uvector<int64_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  int64_t default_value,
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
