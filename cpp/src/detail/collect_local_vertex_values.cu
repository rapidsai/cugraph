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

#include <detail/graph_utils.cuh>

#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename value_t, bool multi_gpu>
rmm::device_uvector<value_t> collect_local_vertex_values_from_ext_vertex_value_pairs(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_vertices,
  rmm::device_uvector<value_t>&& d_values,
  rmm::device_uvector<vertex_t> const& number_map,
  vertex_t local_vertex_first,
  vertex_t local_vertex_last,
  value_t default_value,
  bool do_expensive_check)
{
  rmm::device_uvector<value_t> d_local_values(0, handle.get_stream());

  if constexpr (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    std::tie(d_vertices, d_values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
      comm,
      d_vertices.begin(),
      d_vertices.end(),
      d_values.begin(),
      cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size},
      handle.get_stream());
  }

  // Now I can renumber locally
  renumber_local_ext_vertices<vertex_t, multi_gpu>(handle,
                                                   d_vertices.data(),
                                                   d_vertices.size(),
                                                   number_map.data(),
                                                   local_vertex_first,
                                                   local_vertex_last,
                                                   do_expensive_check);

  auto vertex_iterator = thrust::make_transform_iterator(
    d_vertices.begin(),
    [local_vertex_first] __device__(vertex_t v) { return v - local_vertex_first; });

  d_local_values.resize(local_vertex_last - local_vertex_first, handle.get_stream());
  thrust::fill(
    handle.get_thrust_policy(), d_local_values.begin(), d_local_values.end(), default_value);

  thrust::scatter(handle.get_thrust_policy(),
                  d_values.begin(),
                  d_values.end(),
                  vertex_iterator,
                  d_local_values.begin());

  return d_local_values;
}

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

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, float, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  float default_value,
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

template rmm::device_uvector<double>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, double, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  double default_value,
  bool do_expensive_check);

template rmm::device_uvector<float>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, float, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<float>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  float default_value,
  bool do_expensive_check);

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
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, double, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<double>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  double default_value,
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

template rmm::device_uvector<int64_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int64_t, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  int64_t default_value,
  bool do_expensive_check);

template rmm::device_uvector<int64_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  rmm::device_uvector<int64_t> const& number_map,
  int64_t local_vertex_first,
  int64_t local_vertex_last,
  int64_t default_value,
  bool do_expensive_check);

template rmm::device_uvector<int32_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int32_t, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  int32_t default_value,
  bool do_expensive_check);

template rmm::device_uvector<int64_t>
collect_local_vertex_values_from_ext_vertex_value_pairs<int32_t, int64_t, true>(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int64_t>&& d_values,
  rmm::device_uvector<int32_t> const& number_map,
  int32_t local_vertex_first,
  int32_t local_vertex_last,
  int64_t default_value,
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
