/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t>
std::vector<size_t> compute_local_edge_major_tx_counts(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sorted_edge_majors,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_rank = minor_comm.get_rank();
  auto const minor_comm_size = minor_comm.get_size();

  std::vector<vertex_t> h_major_range_lasts(minor_comm_size);
  for (int i = 0; i < minor_comm_size; ++i) {
    auto vertex_partition_id =
      detail::compute_local_edge_partition_major_range_vertex_partition_id_t{
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
    h_major_range_lasts[i] = vertex_partition_range_lasts[vertex_partition_id];
  }

  rmm::device_uvector<vertex_t> d_major_range_lasts(minor_comm_size, handle.get_stream());
  raft::update_device(d_major_range_lasts.data(),
                      h_major_range_lasts.data(),
                      h_major_range_lasts.size(),
                      handle.get_stream());

  rmm::device_uvector<size_t> d_tx_lasts(minor_comm_size, handle.get_stream());
  thrust::lower_bound(handle.get_thrust_policy(),
                      sorted_edge_majors.begin(),
                      sorted_edge_majors.end(),
                      d_major_range_lasts.begin(),
                      d_major_range_lasts.end(),
                      d_tx_lasts.begin());

  std::vector<size_t> h_tx_lasts(minor_comm_size);
  raft::update_host(h_tx_lasts.data(), d_tx_lasts.data(), d_tx_lasts.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<size_t> tx_counts(minor_comm_size);
  std::adjacent_difference(h_tx_lasts.begin(), h_tx_lasts.end(), tx_counts.begin());

  return tx_counts;
}

template <typename vertex_t>
std::vector<size_t> compute_local_edge_minor_tx_counts(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> sorted_edge_minors,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_rank = minor_comm.get_rank();
  auto const minor_comm_size = minor_comm.get_size();

  std::vector<vertex_t> h_minor_range_lasts(major_comm_size);
  for (int i = 0; i < major_comm_size; ++i) {
    auto vertex_partition_id =
      detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
    h_minor_range_lasts[i] = vertex_partition_range_lasts[vertex_partition_id];
  }

  rmm::device_uvector<vertex_t> d_minor_range_lasts(major_comm_size, handle.get_stream());
  raft::update_device(d_minor_range_lasts.data(),
                      h_minor_range_lasts.data(),
                      h_minor_range_lasts.size(),
                      handle.get_stream());

  rmm::device_uvector<size_t> d_tx_lasts(major_comm_size, handle.get_stream());
  thrust::lower_bound(handle.get_thrust_policy(),
                      sorted_edge_minors.begin(),
                      sorted_edge_minors.end(),
                      d_minor_range_lasts.begin(),
                      d_minor_range_lasts.end(),
                      d_tx_lasts.begin());

  std::vector<size_t> h_tx_lasts(major_comm_size);
  raft::update_host(h_tx_lasts.data(), d_tx_lasts.data(), d_tx_lasts.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<size_t> tx_counts(major_comm_size);
  std::adjacent_difference(h_tx_lasts.begin(), h_tx_lasts.end(), tx_counts.begin());

  return tx_counts;
}

}  // namespace

namespace detail {

// edges are already partitioned by edge partitioning; re-shuffle edge majors by vertex
// partitioning; and this can be done using just minor communicator
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_majors_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_majors,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

  thrust::sort(handle.get_thrust_policy(), edge_majors.begin(), edge_majors.end());

  auto tx_counts = compute_local_edge_major_tx_counts(
    handle,
    raft::device_span<vertex_t const>{edge_majors.data(), edge_majors.size()},
    vertex_partition_range_lasts);

  std::tie(edge_majors, std::ignore) =
    shuffle_values(minor_comm,
                   edge_majors.begin(),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());

  return std::move(edge_majors);
}

// edges are already partitioned by edge partitioning; re-shuffle edge majors by vertex
// partitioning; and this can be done using just minor communicator
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_major_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_majors,
  dataframe_buffer_type_t<value_t>&& edge_values,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

  thrust::sort_by_key(handle.get_thrust_policy(),
                      edge_majors.begin(),
                      edge_majors.end(),
                      get_dataframe_buffer_begin(edge_values));

  auto tx_counts = compute_local_edge_major_tx_counts(
    handle,
    raft::device_span<vertex_t const>{edge_majors.data(), edge_majors.size()},
    vertex_partition_range_lasts);

  std::tie(edge_majors, std::ignore) =
    shuffle_values(minor_comm,
                   edge_majors.begin(),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());
  std::tie(edge_values, std::ignore) =
    shuffle_values(minor_comm,
                   get_dataframe_buffer_begin(edge_values),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());

  return std::make_tuple(std::move(edge_majors), std::move(edge_values));
}

// edges are already partitioned by edge partitioning; re-shuffle edge minors by vertex
// partitioning; and this can be done using just major communicator
template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_minors_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_minors,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());

  thrust::sort(handle.get_thrust_policy(), edge_minors.begin(), edge_minors.end());

  auto tx_counts = compute_local_edge_minor_tx_counts(
    handle,
    raft::device_span<vertex_t const>{edge_minors.data(), edge_minors.size()},
    vertex_partition_range_lasts);

  std::tie(edge_minors, std::ignore) =
    shuffle_values(major_comm,
                   edge_minors.begin(),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());

  return std::move(edge_minors);
}

// edges are already partitioned by edge partitioning; re-shuffle edge minors by vertex
// partitioning; and this can be done using just major communicator
template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_minor_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_minors,
  dataframe_buffer_type_t<value_t>&& edge_values,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
{
  auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());

  thrust::sort_by_key(handle.get_thrust_policy(),
                      edge_minors.begin(),
                      edge_minors.end(),
                      get_dataframe_buffer_begin(edge_values));

  auto tx_counts = compute_local_edge_minor_tx_counts(
    handle,
    raft::device_span<vertex_t const>{edge_minors.data(), edge_minors.size()},
    vertex_partition_range_lasts);

  std::tie(edge_minors, std::ignore) =
    shuffle_values(major_comm,
                   edge_minors.begin(),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());
  std::tie(edge_values, std::ignore) =
    shuffle_values(major_comm,
                   get_dataframe_buffer_begin(edge_values),
                   raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                   handle.get_stream());

  return std::make_tuple(std::move(edge_minors), std::move(edge_values));
}

}  // namespace detail

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_srcs(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_srcs,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  bool store_transposed)
{
  if (store_transposed) {
    return detail::shuffle_local_edge_minors_to_local_gpu_by_vertex_partitioning(
      handle, std::move(edge_srcs), vertex_partition_range_lasts);
  } else {
    return detail::shuffle_local_edge_majors_to_local_gpu_by_vertex_partitioning(
      handle, std::move(edge_srcs), vertex_partition_range_lasts);
  }
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_src_value_pairs(raft::handle_t const& handle,
                                   rmm::device_uvector<vertex_t>&& edge_srcs,
                                   dataframe_buffer_type_t<value_t>&& edge_values,
                                   raft::host_span<vertex_t const> vertex_partition_range_lasts,
                                   bool store_transposed)
{
  if (store_transposed) {
    return detail::
      shuffle_local_edge_minor_value_pairs_to_local_gpu_by_vertex_partitioning<vertex_t, value_t>(
        handle, std::move(edge_srcs), std::move(edge_values), vertex_partition_range_lasts);
  } else {
    return detail::
      shuffle_local_edge_major_value_pairs_to_local_gpu_by_vertex_partitioning<vertex_t, value_t>(
        handle, std::move(edge_srcs), std::move(edge_values), vertex_partition_range_lasts);
  }
}

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_local_edge_dsts(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edge_dsts,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  bool store_transposed)
{
  if (store_transposed) {
    return detail::shuffle_local_edge_majors_to_local_gpu_by_vertex_partitioning(
      handle, std::move(edge_dsts), vertex_partition_range_lasts);
  } else {
    return detail::shuffle_local_edge_minors_to_local_gpu_by_vertex_partitioning(
      handle, std::move(edge_dsts), vertex_partition_range_lasts);
  }
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, dataframe_buffer_type_t<value_t>>
shuffle_local_edge_dst_value_pairs(raft::handle_t const& handle,
                                   rmm::device_uvector<vertex_t>&& edge_dsts,
                                   dataframe_buffer_type_t<value_t>&& edge_values,
                                   raft::host_span<vertex_t const> vertex_partition_range_lasts,
                                   bool store_transposed)
{
  if (store_transposed) {
    return detail::
      shuffle_local_edge_major_value_pairs_to_local_gpu_by_vertex_partitioning<vertex_t, value_t>(
        handle, std::move(edge_dsts), std::move(edge_values), vertex_partition_range_lasts);
  } else {
    return detail::
      shuffle_local_edge_minor_value_pairs_to_local_gpu_by_vertex_partitioning<vertex_t, value_t>(
        handle, std::move(edge_dsts), std::move(edge_values), vertex_partition_range_lasts);
  }
}

}  // namespace cugraph
