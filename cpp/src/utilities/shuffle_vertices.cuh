/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#pragma once

#include "detail/graph_partition_utils.cuh"

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

template <typename vertex_t, typename value0_t, typename value1_t, typename func_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value0_t>, std::optional<rmm::device_uvector<value1_t>>>
shuffle_vertices_and_values_by_gpu_id_impl(raft::handle_t const& handle,
                                           rmm::device_uvector<vertex_t>&& d_vertices,
                                           rmm::device_uvector<value0_t>&& d_values_0,
                                           std::optional<rmm::device_uvector<value1_t>>&& d_values_1,
                                           func_t func)
{

 if (d_values_1) {
  auto [d_shuffled_vertices, d_values, counts] = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
    handle.get_comms(),
    d_vertices.begin(),
    d_vertices.end(),
    thrust::make_zip_iterator(d_values_0.begin(), (*d_values_1).begin()),
    [key_func = func] __device__(auto val) { return key_func(val); },
    handle.get_stream());
  
  return std::make_tuple(
    std::move(d_shuffled_vertices), std::move(std::get<0>(d_values)), std::make_optional(std::move(std::get<1>(d_values))));
 } else {
  auto [d_shuffled_vertices, d_values, counts] = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
    handle.get_comms(),
    d_vertices.begin(),
    d_vertices.end(),
    d_values_0.begin(),
    [key_func = func] __device__(auto val) { return key_func(val); },
    handle.get_stream());

    auto d_values_1 = std::optional<rmm::device_uvector<int32_t>>{std::nullopt};

  return std::make_tuple(
    std::move(d_shuffled_vertices), std::move(d_values), std::move(d_values_1));
 }
  
}

}  // namespace

namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle, rmm::device_uvector<vertex_t>&& vertices)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  return shuffle_vertices_by_gpu_id_impl(
    handle,
    std::move(vertices),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertices(0, handle.get_stream());
  rmm::device_uvector<value_t> d_values(0, handle.get_stream());

  std::tie(d_vertices, d_values, std::ignore) = shuffle_vertices_and_values_by_gpu_id_impl(
    handle,
    std::move(vertices),
    std::move(values),
    std::optional<rmm::device_uvector<int32_t>>{std::nullopt},
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});

  return std::make_tuple(std::move(d_vertices), std::move(d_values));
}

template <typename vertex_t, typename value0_t, typename value1_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value0_t>, rmm::device_uvector<value1_t>>
shuffle_ext_vertex_values_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value0_t>&& values_0,
  rmm::device_uvector<value1_t>&& values_1)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertices(0, handle.get_stream());
  rmm::device_uvector<value0_t> d_values_0(0, handle.get_stream());
  std::optional<rmm::device_uvector<value1_t>> d_values_1(std::nullopt);

  std::tie(d_vertices, d_values_0, d_values_1) = shuffle_vertices_and_values_by_gpu_id_impl(
    handle,
    std::move(vertices),
    std::move(values_0),
    std::make_optional(std::move(values_1)),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});

  return std::make_tuple(std::move(d_vertices), std::move(d_values_0), std::move(*d_values_1));
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
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto return_value = shuffle_vertices_by_gpu_id_impl(
    handle,
    std::move(vertices),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size});

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
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertices(0, handle.get_stream());
  rmm::device_uvector<value_t> d_values(0, handle.get_stream());

  std::tie(d_vertices, d_values, std::ignore) = shuffle_vertices_and_values_by_gpu_id_impl(
    handle,
    std::move(vertices),
    std::move(values),
    std::optional<rmm::device_uvector<int32_t>>{std::nullopt},
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size});

  return std::make_tuple(std::move(d_vertices), std::move(d_values));
}

}  // namespace detail

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_external_vertex_value_pairs(raft::handle_t const& handle,
                                    rmm::device_uvector<vertex_t>&& vertices,
                                    rmm::device_uvector<value_t>&& values)
{
  return detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
    handle, std::move(vertices), std::move(values));
}

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_external_vertices(raft::handle_t const& handle,
                                                        rmm::device_uvector<vertex_t>&& vertices)
{
  return detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(handle,
                                                                          std::move(vertices));
}

}  // namespace cugraph
