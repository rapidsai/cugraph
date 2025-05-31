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
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/variant/edge_properties.hpp>

#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t, typename func_t>
struct vertex_pair_groupby_functor_t {
  func_t func_;

  template <typename TupleType>
  auto __device__ operator()(TupleType tup) const
  {
    return func_(thrust::get<0>(tup), thrust::get<1>(tup));
  }
};

template <typename vertex_t, typename func_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::numeric_device_uvector_t>,
           std::vector<size_t>>
shuffle_vertex_pairs_with_values_by_gpu_id_impl(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::vector<cugraph::numeric_device_uvector_t>&& edge_properties,
  func_t func)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  size_t element_size = sizeof(vertex_t) * 2;

  if (edge_properties.size() == 1) {
    element_size += cugraph::variant_type_dispatch(edge_properties[0], cugraph::variant_size{});
  } else if (edge_properties.size() > 1) {
    element_size += sizeof(size_t);
  }

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
  vertex_pair_groupby_functor_t<vertex_t, func_t> groupby_functor{func};

  if (edge_properties.size() == 0) {
    d_tx_value_counts =
      cugraph::groupby_and_count(thrust::make_zip_iterator(majors.begin(), minors.begin()),
                                 thrust::make_zip_iterator(majors.end(), minors.end()),
                                 groupby_functor,
                                 comm_size,
                                 mem_frugal_threshold,
                                 handle.get_stream());
  } else if (edge_properties.size() == 1) {
    d_tx_value_counts = cugraph::variant_type_dispatch(
      edge_properties[0],
      [&handle, &majors, &minors, &groupby_functor, comm_size, mem_frugal_threshold](auto& prop) {
        return cugraph::groupby_and_count(
          thrust::make_zip_iterator(majors.begin(), minors.begin(), prop.begin()),
          thrust::make_zip_iterator(majors.end(), minors.end(), prop.end()),
          groupby_functor,
          comm_size,
          mem_frugal_threshold,
          handle.get_stream());
      });
  } else {
    rmm::device_uvector<size_t> property_position(majors.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

    d_tx_value_counts = cugraph::groupby_and_count(
      thrust::make_zip_iterator(majors.begin(), minors.begin(), property_position.begin()),
      thrust::make_zip_iterator(majors.end(), minors.end(), property_position.end()),
      groupby_functor,
      comm_size,
      mem_frugal_threshold,
      handle.get_stream());

    std::for_each(edge_properties.begin(),
                  edge_properties.end(),
                  [&property_position, &handle](auto& property) {
                    cugraph::variant_type_dispatch(
                      property, [&handle, &property_position](auto& prop) {
                        using T = typename std::remove_reference<decltype(prop)>::type;
                        T tmp(prop.size(), handle.get_stream());

                        thrust::gather(handle.get_thrust_policy(),
                                       property_position.begin(),
                                       property_position.end(),
                                       prop.begin(),
                                       tmp.begin());

                        prop = std::move(tmp);
                      });
                  });
  }

  std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
  raft::update_host(h_tx_value_counts.data(),
                    d_tx_value_counts.data(),
                    d_tx_value_counts.size(),
                    handle.get_stream());
  handle.sync_stream();

  std::vector<size_t> rx_counts{};

  if (mem_frugal_flag ||
      (edge_properties.size() > 1)) {  // trade-off potential parallelism to lower peak memory
    std::tie(majors, rx_counts) = shuffle_values(
      comm,
      majors.begin(),
      raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
      handle.get_stream());

    std::tie(minors, rx_counts) = shuffle_values(
      comm,
      minors.begin(),
      raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
      handle.get_stream());

    std::for_each(
      edge_properties.begin(),
      edge_properties.end(),
      [&handle, &h_tx_value_counts, &comm](auto& property) {
        cugraph::variant_type_dispatch(property, [&handle, &h_tx_value_counts, &comm](auto& prop) {
          std::tie(prop, std::ignore) = shuffle_values(
            comm,
            prop.begin(),
            raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
            handle.get_stream());
        });
      });
  } else {
    if (edge_properties.size() == 0) {
      std::forward_as_tuple(std::tie(majors, minors), rx_counts) = shuffle_values(
        comm,
        thrust::make_zip_iterator(majors.begin(), minors.begin()),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream());
    } else {
      cugraph::variant_type_dispatch(
        edge_properties[0], [&handle, &majors, &minors, &comm, &h_tx_value_counts](auto& prop) {
          std::forward_as_tuple(std::tie(majors, minors, prop), std::ignore) = shuffle_values(
            comm,
            thrust::make_zip_iterator(majors.begin(), minors.begin(), prop.begin()),
            raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
            handle.get_stream());
        });
    }
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(rx_counts));
}

}  // namespace

namespace detail {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::numeric_device_uvector_t>,
           std::vector<size_t>>
shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::vector<cugraph::numeric_device_uvector_t>&& edge_properties)
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
    std::move(edge_properties),
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::numeric_device_uvector_t>,
           std::vector<size_t>>
shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::vector<cugraph::numeric_device_uvector_t>&& edge_properties,
  raft::host_span<vertex_t const> vertex_partition_range_lasts)
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
    std::move(edge_properties),
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size});
}

}  // namespace detail

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::numeric_device_uvector_t>,
           std::vector<size_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::vector<cugraph::numeric_device_uvector_t>&& edge_properties,
                  bool store_transposed)
{
  auto& comm           = handle.get_comms();
  auto const comm_size = comm.get_size();

  auto majors = store_transposed ? std::move(edge_dsts) : std::move(edge_srcs);
  auto minors = store_transposed ? std::move(edge_srcs) : std::move(edge_dsts);

#if 1

  std::vector<size_t> rx_counts{};
  std::tie(majors, minors, edge_properties, rx_counts) =
    detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
      handle, std::move(majors), std::move(minors), std::move(edge_properties));

  edge_srcs = store_transposed ? std::move(minors) : std::move(majors);
  edge_dsts = store_transposed ? std::move(majors) : std::move(minors);

#else

  bool has_weights{false};
  bool has_edge_ids{false};
  bool has_edge_types{false};
  bool has_edge_start_times{false};
  bool has_edge_end_times{false};

  std::vector<cugraph::numeric_device_uvector_t> edge_properties{};

  if (edge_weights) {
    has_weights = true;
    edge_properties.push_back(std::move(*edge_weights));
  }
  if (edge_ids) {
    has_edge_ids = true;
    edge_properties.push_back(std::move(*edge_ids));
  }

  if (edge_types) {
    has_edge_types = true;
    edge_properties.push_back(std::move(*edge_types));
  }

  if (edge_start_times) {
    has_edge_start_times = true;
    edge_properties.push_back(std::move(*edge_start_times));
  }

  if (edge_end_times) {
    has_edge_end_times = true;
    edge_properties.push_back(std::move(*edge_end_times));
  }

  std::vector<size_t> rx_counts{};
  std::tie(majors, minors, edge_properties, rx_counts) =
    detail::shuffle_ext_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning(
      handle, std::move(majors), std::move(minors), std::move(edge_properties));

  size_t pos = 0;

  edge_weights = has_weights ? std::make_optional(std::get<rmm::device_uvector<weight_t>>(
                                 std::move(edge_properties[pos++])))
                             : std::nullopt;
  edge_ids =
    has_edge_ids
      ? std::make_optional(std::get<rmm::device_uvector<edge_t>>(std::move(edge_properties[pos++])))
      : std::nullopt;
  edge_types       = has_edge_types ? std::make_optional(std::get<rmm::device_uvector<edge_type_t>>(
                                  std::move(edge_properties[pos++])))
                                    : std::nullopt;
  edge_start_times = has_edge_start_times
                       ? std::make_optional(std::get<rmm::device_uvector<edge_time_t>>(
                           std::move(edge_properties[pos++])))
                       : std::nullopt;
  edge_end_times   = has_edge_end_times
                       ? std::make_optional(std::get<rmm::device_uvector<edge_time_t>>(
                         std::move(edge_properties[pos++])))
                       : std::nullopt;

  edge_srcs = store_transposed ? std::move(minors) : std::move(majors);
  edge_dsts = store_transposed ? std::move(majors) : std::move(minors);

#endif

  return std::make_tuple(
    std::move(edge_srcs), std::move(edge_dsts), std::move(edge_properties), std::move(rx_counts));
}

}  // namespace cugraph
