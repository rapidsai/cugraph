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
#include "detail/graph_partition_utils.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename func_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::vector<size_t>>
shuffle_vertex_pairs_with_values_by_gpu_id_impl(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  func_t func)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  auto total_global_mem = handle.get_device_properties().totalGlobalMem;
  auto element_size     = sizeof(vertex_t) * 2 + (weights ? sizeof(weight_t) : size_t{0}) +
                      (edge_ids ? sizeof(edge_t) : size_t{0}) +
                      (edge_types ? sizeof(edge_type_t) : size_t{0});
  auto constexpr mem_frugal_ratio =
    0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
          // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
          // group-by by default, and thrust::sort requires temporary buffer comparable to the input
          // data size)
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

  // FIXME: Consider a generic function that takes a value tuple of optionals
  //        to eliminate this complexity
  rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());

  if (weights) {
    if (edge_ids) {
      if (edge_types) {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(),
                                    minors.begin(),
                                    weights->begin(),
                                    edge_ids->begin(),
                                    edge_types->begin()),
          thrust::make_zip_iterator(
            majors.end(), minors.end(), weights->end(), edge_ids->end(), edge_types->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      } else {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(
            majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), weights->end(), edge_ids->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      }
    } else {
      if (edge_types) {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(
            majors.begin(), minors.begin(), weights->begin(), edge_types->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), weights->end(), edge_types->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      } else {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), weights->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      }
    }
  } else {
    if (edge_ids) {
      if (edge_types) {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(
            majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), edge_ids->end(), edge_types->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      } else {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), edge_ids->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      }
    } else {
      if (edge_types) {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), edge_types->end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      } else {
        d_tx_value_counts = cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(), minors.begin()),
          thrust::make_zip_iterator(majors.end(), minors.end()),
          [func] __device__(auto val) { return func(thrust::get<0>(val), thrust::get<1>(val)); },
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      }
    }
  }

  std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
  raft::update_host(h_tx_value_counts.data(),
                    d_tx_value_counts.data(),
                    d_tx_value_counts.size(),
                    handle.get_stream());
  handle.sync_stream();

  std::vector<size_t> rx_counts{};

  if (mem_frugal_flag) {  // trade-off potential parallelism to lower peak memory
    std::tie(majors, rx_counts) =
      shuffle_values(comm, majors.begin(), h_tx_value_counts, handle.get_stream());

    std::tie(minors, rx_counts) =
      shuffle_values(comm, minors.begin(), h_tx_value_counts, handle.get_stream());

    if (weights) {
      std::tie(weights, rx_counts) =
        shuffle_values(comm, (*weights).begin(), h_tx_value_counts, handle.get_stream());
    }

    if (edge_ids) {
      std::tie(edge_ids, rx_counts) =
        shuffle_values(comm, (*edge_ids).begin(), h_tx_value_counts, handle.get_stream());
    }

    if (edge_types) {
      std::tie(edge_types, rx_counts) =
        shuffle_values(comm, (*edge_types).begin(), h_tx_value_counts, handle.get_stream());
    }
  } else {
    if (weights) {
      if (edge_ids) {
        if (edge_types) {
          std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, edge_types),
                                rx_counts) =
            shuffle_values(comm,
                           thrust::make_zip_iterator(majors.begin(),
                                                     minors.begin(),
                                                     weights->begin(),
                                                     edge_ids->begin(),
                                                     edge_types->begin()),
                           h_tx_value_counts,
                           handle.get_stream());
        } else {
          std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids), rx_counts) =
            shuffle_values(comm,
                           thrust::make_zip_iterator(
                             majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()),
                           h_tx_value_counts,
                           handle.get_stream());
        }
      } else {
        if (edge_types) {
          std::forward_as_tuple(std::tie(majors, minors, weights, edge_types), rx_counts) =
            shuffle_values(comm,
                           thrust::make_zip_iterator(
                             majors.begin(), minors.begin(), weights->begin(), edge_types->begin()),
                           h_tx_value_counts,
                           handle.get_stream());
        } else {
          std::forward_as_tuple(std::tie(majors, minors, weights), rx_counts) = shuffle_values(
            comm,
            thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()),
            h_tx_value_counts,
            handle.get_stream());
        }
      }
    } else {
      if (edge_ids) {
        if (edge_types) {
          std::forward_as_tuple(std::tie(majors, minors, edge_ids, edge_types), rx_counts) =
            shuffle_values(
              comm,
              thrust::make_zip_iterator(
                majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()),
              h_tx_value_counts,
              handle.get_stream());
        } else {
          std::forward_as_tuple(std::tie(majors, minors, edge_ids), rx_counts) = shuffle_values(
            comm,
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()),
            h_tx_value_counts,
            handle.get_stream());
        }
      } else {
        if (edge_types) {
          std::forward_as_tuple(std::tie(majors, minors, edge_types), rx_counts) = shuffle_values(
            comm,
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()),
            h_tx_value_counts,
            handle.get_stream());
        } else {
          std::forward_as_tuple(std::tie(majors, minors), rx_counts) =
            shuffle_values(comm,
                           thrust::make_zip_iterator(majors.begin(), minors.begin()),
                           h_tx_value_counts,
                           handle.get_stream());
        }
      }
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(rx_counts));
}

}  // namespace

namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types)
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
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});
}

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::vector<vertex_t> const& vertex_partition_range_lasts)
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
    cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      comm_size,
      major_comm_size,
      minor_comm_size});
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int32_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int64_t> const& vertex_partition_range_lasts);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::vector<int64_t> const& vertex_partition_range_lasts);

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<vertex_t>&& edge_srcs,
                       rmm::device_uvector<vertex_t>&& edge_dsts,
                       std::optional<rmm::device_uvector<weight_t>>&& edge_weights,
                       std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<edge_type_t>>&& edge_types)
{
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  return detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
    handle,
    std::move(edge_srcs),
    std::move(edge_dsts),
    std::move(edge_weights),
    std::move(edge_ids),
    std::move(edge_types));
}

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int32_t>&& majors,
                       rmm::device_uvector<int32_t>&& minors,
                       std::optional<rmm::device_uvector<float>>&& weights,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int32_t>&& majors,
                       rmm::device_uvector<int32_t>&& minors,
                       std::optional<rmm::device_uvector<double>>&& weights,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int32_t>&& majors,
                       rmm::device_uvector<int32_t>&& minors,
                       std::optional<rmm::device_uvector<float>>&& weights,
                       std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int32_t>&& majors,
                       rmm::device_uvector<int32_t>&& minors,
                       std::optional<rmm::device_uvector<double>>&& weights,
                       std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int64_t>&& majors,
                       rmm::device_uvector<int64_t>&& minors,
                       std::optional<rmm::device_uvector<float>>&& weights,
                       std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::vector<size_t>>
shuffle_external_edges(raft::handle_t const& handle,
                       rmm::device_uvector<int64_t>&& majors,
                       rmm::device_uvector<int64_t>&& minors,
                       std::optional<rmm::device_uvector<double>>&& weights,
                       std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
                       std::optional<rmm::device_uvector<int32_t>>&& edge_types);

}  // namespace cugraph
