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
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename func_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::vector<size_t>>
shuffle_vertex_pairs_with_values_by_gpu_id_impl(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  func_t func,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  int edge_property_count = 0;
  size_t element_size     = sizeof(vertex_t) * 2;

  if (weights) {
    ++edge_property_count;
    element_size += sizeof(weight_t);
  }

  if (edge_ids) {
    ++edge_property_count;
    element_size += sizeof(edge_t);
  }
  if (edge_types) {
    ++edge_property_count;
    element_size += sizeof(edge_type_t);
  }
  if (edge_start_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }
  if (edge_end_times) {
    ++edge_property_count;
    element_size += sizeof(edge_time_t);
  }

  if (edge_property_count > 1) { element_size = sizeof(vertex_t) * 2 + sizeof(size_t); }

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio =
    0.05;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
           // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
           // group-by by default, and thrust::sort requires temporary buffer comparable to the
           // input data size)
  auto mem_frugal_threshold =
    static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

  auto mem_frugal_flag =
    host_scalar_allreduce(comm,
                          majors.size() > mem_frugal_threshold ? int{1} : int{0},
                          raft::comms::op_t::MAX,
                          handle.get_stream());

  // invoke groupby_and_count and shuffle values to pass mem_frugal_threshold instead of directly
  // calling groupby_gpu_id_and_shuffle_values there is no benefit in reducing peak memory as we
  // need to allocate a receive buffer anyways) but this reduces the maximum memory allocation size
  // by half or more (thrust::sort used inside the groupby_and_count allocates the entire temporary
  // buffer in a single chunk, and the pool allocator  often cannot handle a large single allocation
  // (due to fragmentation) even when the remaining free memory in aggregate is significantly larger
  // than the requested size).

  rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());

  if (edge_property_count == 0) {
    d_tx_value_counts = cugraph::groupby_and_count(
      thrust::make_zip_iterator(majors.begin(), minors.begin()),
      thrust::make_zip_iterator(majors.end(), minors.end()),
      [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
      comm_size,
      mem_frugal_threshold,
      handle.get_stream(),
      large_buffer_type);
  } else if (edge_property_count == 1) {
    if (weights) {
      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), weights->end()),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_ids) {
      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), edge_ids->end()),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_types) {
      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), edge_types->end()),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_start_times) {
      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_start_times->begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), edge_start_times->end()),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_end_times) {
      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_end_times->begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), edge_end_times->end()),
        [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);
    }
  } else {
    auto property_position =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<edge_t>(majors.size(), handle.get_stream())
        : rmm::device_uvector<edge_t>(majors.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), edge_t{0});

    d_tx_value_counts = cugraph::groupby_and_count(
      thrust::make_zip_iterator(majors.begin(), minors.begin(), property_position.begin()),
      thrust::make_zip_iterator(majors.end(), minors.end(), property_position.end()),
      [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
      comm_size,
      mem_frugal_threshold,
      handle.get_stream(),
      large_buffer_type);

    if (weights) {
      auto tmp = large_buffer_type
                   ? large_buffer_manager::allocate_memory_buffer<weight_t>(
                       property_position.size(), handle.get_stream())
                   : rmm::device_uvector<weight_t>(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     weights->begin(),
                     tmp.begin());

      weights = std::move(tmp);
    }

    if (edge_ids) {
      auto tmp = large_buffer_type
                   ? large_buffer_manager::allocate_memory_buffer<edge_t>(property_position.size(),
                                                                          handle.get_stream())
                   : rmm::device_uvector<edge_t>(property_position.size(), handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edge_ids->begin(),
                     tmp.begin());

      edge_ids = std::move(tmp);
    }

    if (edge_types) {
      auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<edge_type_t>(
                                       property_position.size(), handle.get_stream())
                                   : rmm::device_uvector<edge_type_t>(property_position.size(),
                                                                      handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edge_types->begin(),
                     tmp.begin());

      edge_types = std::move(tmp);
    }

    if (edge_start_times) {
      auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<edge_time_t>(
                                       property_position.size(), handle.get_stream())
                                   : rmm::device_uvector<edge_time_t>(property_position.size(),
                                                                      handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edge_start_times->begin(),
                     tmp.begin());

      edge_start_times = std::move(tmp);
    }

    if (edge_end_times) {
      auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<edge_time_t>(
                                       property_position.size(), handle.get_stream())
                                   : rmm::device_uvector<edge_time_t>(property_position.size(),
                                                                      handle.get_stream());

      thrust::gather(handle.get_thrust_policy(),
                     property_position.begin(),
                     property_position.end(),
                     edge_end_times->begin(),
                     tmp.begin());

      edge_end_times = std::move(tmp);
    }
  }

  std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
  raft::update_host(h_tx_value_counts.data(),
                    d_tx_value_counts.data(),
                    d_tx_value_counts.size(),
                    handle.get_stream());
  handle.sync_stream();

  std::vector<size_t> rx_counts{};

  if (mem_frugal_flag ||
      (edge_property_count > 1)) {  // trade-off potential parallelism to lower peak memory
    std::tie(majors, rx_counts) = shuffle_values(
      comm,
      majors.begin(),
      raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
      handle.get_stream(),
      large_buffer_type);

    std::tie(minors, rx_counts) = shuffle_values(
      comm,
      minors.begin(),
      raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
      handle.get_stream(),
      large_buffer_type);

    if (weights) {
      std::tie(weights, rx_counts) = shuffle_values(
        comm,
        (*weights).begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }

    if (edge_ids) {
      std::tie(edge_ids, rx_counts) = shuffle_values(
        comm,
        (*edge_ids).begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }

    if (edge_types) {
      std::tie(edge_types, rx_counts) = shuffle_values(
        comm,
        (*edge_types).begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }

    if (edge_start_times) {
      std::tie(edge_start_times, rx_counts) = shuffle_values(
        comm,
        (*edge_start_times).begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }

    if (edge_end_times) {
      std::tie(edge_end_times, rx_counts) = shuffle_values(
        comm,
        (*edge_end_times).begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }
  } else {
    // There is at most one edge property set
    if (weights) {
      std::forward_as_tuple(std::tie(majors, minors, weights), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_ids) {
      std::forward_as_tuple(std::tie(majors, minors, edge_ids), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_types) {
      std::forward_as_tuple(std::tie(majors, minors, edge_types), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_start_times) {
      std::forward_as_tuple(std::tie(majors, minors, edge_start_times), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_start_times->begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else if (edge_end_times) {
      std::forward_as_tuple(std::tie(majors, minors, edge_end_times), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_end_times->begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    } else {
      std::forward_as_tuple(std::tie(majors, minors), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(rx_counts));
}

}  // namespace

namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  return shuffle_vertex_pairs_with_values_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::move(edge_start_times),
    std::move(edge_start_times),
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size},
    large_buffer_type);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(vertex_partition_range_lasts.size(),
                                                               handle.get_stream());
  raft::update_device(d_vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.data(),
                      vertex_partition_range_lasts.size(),
                      handle.get_stream());

  return shuffle_vertex_pairs_with_values_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(weights),
    std::move(edge_ids),
    std::move(edge_types),
    std::move(edge_start_times),
    std::move(edge_start_times),
    cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      comm_size,
      major_comm_size,
      minor_comm_size},
    large_buffer_type);
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::optional<rmm::device_uvector<weight_t>>&& edge_weights,
                  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
                  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edge_start_times,
                  std::optional<rmm::device_uvector<edge_time_t>>&& edge_end_times,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  auto majors = store_transposed ? std::move(edge_dsts) : std::move(edge_srcs);
  auto minors = store_transposed ? std::move(edge_srcs) : std::move(edge_dsts);
  std::vector<size_t> rx_counts{};
  std::tie(majors,
           minors,
           edge_weights,
           edge_ids,
           edge_types,
           edge_start_times,
           edge_end_times,
           rx_counts) =
    detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
      handle,
      std::move(majors),
      std::move(minors),
      std::move(edge_weights),
      std::move(edge_ids),
      std::move(edge_types),
      std::move(edge_start_times),
      std::move(edge_end_times),
      large_buffer_type);
  edge_srcs = store_transposed ? std::move(minors) : std::move(majors);
  edge_dsts = store_transposed ? std::move(majors) : std::move(minors);

  return std::make_tuple(std::move(edge_srcs),
                         std::move(edge_dsts),
                         std::move(edge_weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(rx_counts));
}

}  // namespace cugraph
