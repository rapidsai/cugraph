/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t, typename func_t>
rmm::device_uvector<vertex_t> shuffle_vertices_by_gpu_id_impl(
  raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& d_vertices, func_t func)
{
  rmm::device_uvector<vertex_t> d_rx_vertices(0, handle.get_stream());
  std::tie(d_rx_vertices, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_values(
    handle.get_comms(),
    d_vertices.begin(),
    d_vertices.end(),
    [key_func = func] __device__(auto val) { return key_func(val); },
    handle.get_stream());

  return d_rx_vertices;
}

template <typename vertex_t, typename value_t, typename func_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_vertices_and_values_by_gpu_id_impl(raft::handle_t const& handle,
                                           rmm::device_uvector<vertex_t>&& d_vertices,
                                           rmm::device_uvector<value_t>&& d_values,
                                           func_t func)
{
  std::tie(d_vertices, d_values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
    handle.get_comms(),
    d_vertices.begin(),
    d_vertices.end(),
    d_values.begin(),
    [key_func = func] __device__(auto val) { return key_func(val); },
    handle.get_stream());

  return std::make_tuple(std::move(d_vertices), std::move(d_values));
}

}  // namespace

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& vertices)
{
  auto const comm_size = handle.get_comms().get_size();

  return shuffle_vertices_by_gpu_id_impl(
    handle,
    std::move(vertices),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size});
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values)
{
  auto const comm_size = handle.get_comms().get_size();

  return shuffle_vertices_and_values_by_gpu_id_impl(
    handle,
    std::move(vertices),
    std::move(values),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size});
}

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  std::vector<vertex_t> const& vertex_partition_range_lasts)
{
  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  auto return_value = shuffle_vertices_by_gpu_id_impl(
    handle,
    std::move(vertices),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      {d_vertex_partition_range_lasts.data(), d_vertex_partition_range_lasts.size()}});

  handle.sync_stream();

  return return_value;
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values,
  std::vector<vertex_t> const& vertex_partition_range_lasts)
{
  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  auto return_value = shuffle_vertices_and_values_by_gpu_id_impl(
    handle,
    std::move(vertices),
    std::move(values),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      {d_vertex_partition_range_lasts.data(), d_vertex_partition_range_lasts.size()}});

  return return_value;
}

template rmm::device_uvector<int32_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template rmm::device_uvector<int64_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  std::vector<int64_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int32_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& d_vertices,
  rmm::device_uvector<int32_t>&& d_values,
  std::vector<int64_t> const& vertex_partition_range_lasts);

template rmm::device_uvector<int32_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle, rmm::device_uvector<int32_t>&& d_vertices);

template rmm::device_uvector<int64_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle, rmm::device_uvector<int64_t>&& d_vertices);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<int32_t>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<float>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<double>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& vertices,
  rmm::device_uvector<double>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<int64_t>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<float>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<float>&& values);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<double>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& vertices,
  rmm::device_uvector<double>&& values);

}  // namespace detail
}  // namespace cugraph
