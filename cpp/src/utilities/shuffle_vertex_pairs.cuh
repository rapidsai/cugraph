/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "detail/shuffle_wrappers.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/std/tuple>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>

#include <tuple>

namespace cugraph {

namespace {

enum class vertex_pair_shuffle_mode_t { global, two_level_major, two_level_minor };

template <typename vertex_t, typename func_t>
struct vertex_pair_groupby_functor_t {
  func_t func_;
  int major_comm_size{};
  int minor_comm_size{};
  vertex_pair_shuffle_mode_t shuffle_mode{};

  template <typename TupleType>
  auto __device__ operator()(TupleType tup) const
  {
    auto comm_rank = func_(cuda::std::get<0>(tup), cuda::std::get<1>(tup));
    if (shuffle_mode == vertex_pair_shuffle_mode_t::global) {
      return comm_rank;
    } else if (shuffle_mode == vertex_pair_shuffle_mode_t::two_level_major) {
      return partition_manager::compute_major_comm_rank_from_global_comm_rank(
        major_comm_size, minor_comm_size, comm_rank);
    } else {
      assert(shuffle_mdoe == vertex_pair_shuffle_mode_t::two_level_minor);
      return partition_manager::compute_minor_comm_rank_from_global_comm_rank(
        major_comm_size, minor_comm_size, comm_rank);
    }
  }
};

template <typename vertex_t, typename func_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_vertex_pairs_with_values_by_gpu_id_impl(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
  func_t func,
  std::optional<large_buffer_type_t> large_buffer_type = std::nullopt)
{
  CUGRAPH_EXPECTS(!large_buffer_type || large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory buffer is not initialized.");

  constexpr int max_global_comm_all2all_size =
    512;  // NCCL currently has limitations in supportingg All-to-All between a large number of
          // ranks, as a workaround, use a two step approach using major_comm and minor_comm

  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  bool two_level_shuffle = false;
  if (comm_size >
      std::max(max_global_comm_all2all_size, std::max(major_comm_size, minor_comm_size))) {
    two_level_shuffle = true;
  }

  size_t element_size = sizeof(vertex_t) * 2;

  if (edge_properties.size() == 1) {
    element_size +=
      cugraph::variant_type_dispatch(edge_properties[0], cugraph::sizeof_arithmetic_element{});
  } else if (edge_properties.size() > 1) {
    element_size += sizeof(size_t);
  }

  auto mem_frugal_threshold = std::numeric_limits<size_t>::max();
  if (!large_buffer_type) {
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr mem_frugal_ratio =
      0.05;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
             // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
             // group-by by default, and thrust::sort requires temporary buffer comparable to the
             // input data size)
    mem_frugal_threshold =
      static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);
  }

  bool mem_frugal_flag{};
  {
    auto flag = majors.size() > mem_frugal_threshold ? int{1} : int{0};
#if 1  // FIXME: we should add host_allreduce to raft
    flag = host_scalar_allreduce(comm, flag, raft::comms::op_t::MAX, handle.get_stream());
#else
    comm.host_allreduce(
      std::addressof(flag), std::addressof(flag), size_t{1}, raft::comms::op_t::MAX);
#endif
    mem_frugal_flag = (flag > 0) ? true : false;
  }

  for (size_t step = 0; step < (two_level_shuffle ? size_t{2} : size_t{1}); ++step) {
    // invoke groupby_and_count and shuffle values to pass mem_frugal_threshold instead of directly
    // calling groupby_gpu_id_and_shuffle_values there is no benefit in reducing peak memory as we
    // need to allocate a receive buffer anyways) but this reduces the maximum memory allocation
    // size by half or more (thrust::sort used inside the groupby_and_count allocates the entire
    // temporary buffer in a single chunk, and the pool allocator  often cannot handle a large
    // single allocation (due to fragmentation) even when the remaining free memory in aggregate is
    // significantly larger than the requested size).

    vertex_pair_shuffle_mode_t shuffle_mode =
      two_level_shuffle ? (step == 0 ? vertex_pair_shuffle_mode_t::two_level_major
                                     : vertex_pair_shuffle_mode_t::two_level_minor)
                        : vertex_pair_shuffle_mode_t::global;
    auto& this_step_comm    = two_level_shuffle ? (step == 0 ? major_comm : minor_comm) : comm;
    int this_step_comm_size = this_step_comm.get_size();

    rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());
    vertex_pair_groupby_functor_t<vertex_t, func_t> groupby_functor{
      func, major_comm_size, minor_comm_size, shuffle_mode};

    if (edge_properties.size() == 0) {
      d_tx_value_counts =
        cugraph::groupby_and_count(thrust::make_zip_iterator(majors.begin(), minors.begin()),
                                   thrust::make_zip_iterator(majors.end(), minors.end()),
                                   groupby_functor,
                                   this_step_comm_size,
                                   mem_frugal_threshold,
                                   handle.get_stream(),
                                   large_buffer_type);
    } else if (edge_properties.size() == 1) {
      d_tx_value_counts = cugraph::variant_type_dispatch(
        edge_properties[0],
        [&handle,
         &majors,
         &minors,
         &groupby_functor,
         &large_buffer_type,
         this_step_comm_size,
         mem_frugal_threshold](auto& prop) {
          return cugraph::groupby_and_count(
            thrust::make_zip_iterator(majors.begin(), minors.begin(), prop.begin()),
            thrust::make_zip_iterator(majors.end(), minors.end(), prop.end()),
            groupby_functor,
            this_step_comm_size,
            mem_frugal_threshold,
            handle.get_stream(),
            large_buffer_type);
        });
    } else {
      rmm::device_uvector<size_t> property_position(majors.size(), handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

      d_tx_value_counts = cugraph::groupby_and_count(
        thrust::make_zip_iterator(majors.begin(), minors.begin(), property_position.begin()),
        thrust::make_zip_iterator(majors.end(), minors.end(), property_position.end()),
        groupby_functor,
        this_step_comm_size,
        mem_frugal_threshold,
        handle.get_stream(),
        large_buffer_type);

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
        this_step_comm,
        majors.begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);

      std::tie(minors, rx_counts) = shuffle_values(
        this_step_comm,
        minors.begin(),
        raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
        handle.get_stream(),
        large_buffer_type);

      std::for_each(
        edge_properties.begin(),
        edge_properties.end(),
        [&handle, &h_tx_value_counts, &large_buffer_type, &this_step_comm](auto& property) {
          cugraph::variant_type_dispatch(
            property,
            [&handle, &h_tx_value_counts, &large_buffer_type, &this_step_comm](auto& prop) {
              std::tie(prop, std::ignore) = shuffle_values(
                this_step_comm,
                prop.begin(),
                raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
                handle.get_stream(),
                large_buffer_type);
            });
        });
    } else {
      if (edge_properties.size() == 0) {
        std::forward_as_tuple(std::tie(majors, minors), rx_counts) = shuffle_values(
          this_step_comm,
          thrust::make_zip_iterator(majors.begin(), minors.begin()),
          raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
          handle.get_stream(),
          large_buffer_type);
      } else {
        cugraph::variant_type_dispatch(
          edge_properties[0],
          [&handle, &majors, &minors, &this_step_comm, &h_tx_value_counts, &large_buffer_type](
            auto& prop) {
            std::forward_as_tuple(std::tie(majors, minors, prop), std::ignore) = shuffle_values(
              this_step_comm,
              thrust::make_zip_iterator(majors.begin(), minors.begin(), prop.begin()),
              raft::host_span<size_t const>(h_tx_value_counts.data(), h_tx_value_counts.size()),
              handle.get_stream(),
              large_buffer_type);
          });
      }
    }
  }

  return std::make_tuple(std::move(majors), std::move(minors), std::move(edge_properties));
}

}  // namespace

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_ext_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
                  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto& comm                 = handle.get_comms();
  auto const comm_size       = comm.get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto majors = store_transposed ? std::move(edge_dsts) : std::move(edge_srcs);
  auto minors = store_transposed ? std::move(edge_srcs) : std::move(edge_dsts);

  std::vector<size_t> rx_counts{};
  std::tie(majors, minors, edge_properties) = shuffle_vertex_pairs_with_values_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(edge_properties),
    cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size},
    large_buffer_type);

  edge_srcs = store_transposed ? std::move(minors) : std::move(majors);
  edge_dsts = store_transposed ? std::move(majors) : std::move(minors);

  return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts), std::move(edge_properties));
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<cugraph::arithmetic_device_uvector_t>>
shuffle_int_edges(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>&& edge_srcs,
                  rmm::device_uvector<vertex_t>&& edge_dsts,
                  std::vector<cugraph::arithmetic_device_uvector_t>&& edge_properties,
                  bool store_transposed,
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

  auto majors = store_transposed ? std::move(edge_dsts) : std::move(edge_srcs);
  auto minors = store_transposed ? std::move(edge_srcs) : std::move(edge_dsts);

  std::vector<size_t> rx_counts{};
  std::tie(majors, minors, edge_properties) = shuffle_vertex_pairs_with_values_by_gpu_id_impl(
    handle,
    std::move(majors),
    std::move(minors),
    std::move(edge_properties),
    cugraph::detail::compute_gpu_id_from_int_edge_endpoints_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      comm_size,
      major_comm_size,
      minor_comm_size},
    large_buffer_type);

  edge_srcs = store_transposed ? std::move(minors) : std::move(majors);
  edge_dsts = store_transposed ? std::move(majors) : std::move(minors);

  return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts), std::move(edge_properties));
}

}  // namespace cugraph
