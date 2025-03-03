/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

  std::tie(vertices, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_values(
    handle.get_comms(),
    vertices.begin(),
    vertices.end(),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size},
    handle.get_stream());

  return std::move(vertices);
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

  std::tie(vertices, values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
    handle.get_comms(),
    vertices.begin(),
    vertices.end(),
    get_dataframe_buffer_begin(values),
    cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size},
    handle.get_stream());

  return std::make_tuple(std::move(vertices), std::move(values));
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

  std::tie(vertices, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_values(
    handle.get_comms(),
    vertices.begin(),
    vertices.end(),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    handle.get_stream());

  return std::move(vertices);
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

  std::tie(vertices, values, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_kv_pairs(
    handle.get_comms(),
    vertices.begin(),
    vertices.end(),
    get_dataframe_buffer_begin(values),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    handle.get_stream());

  return std::make_tuple(std::move(vertices), std::move(values));
}

}  // namespace detail

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices(raft::handle_t const& handle,
                                                   rmm::device_uvector<vertex_t>&& vertices)
{
  return detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(handle,
                                                                          std::move(vertices));
}

// deprecated
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_external_vertices(raft::handle_t const& handle,
                                                        rmm::device_uvector<vertex_t>&& vertices)
{
  return detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(handle,
                                                                          std::move(vertices));
}

// deprecated
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_external_vertex_value_pairs(raft::handle_t const& handle,
                                    rmm::device_uvector<vertex_t>&& vertices,
                                    rmm::device_uvector<value_t>&& values)
{
  return detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
    handle, std::move(vertices), std::move(values));
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<vertex_t>&& vertices,
                               rmm::device_uvector<value_t>&& values)
{
  return detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
    handle, std::move(vertices), std::move(values));
}

}  // namespace cugraph
