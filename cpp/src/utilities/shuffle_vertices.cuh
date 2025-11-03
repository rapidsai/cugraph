/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/graph_partition_utils.cuh"
#include "detail/shuffle_wrappers.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <cuda/std/tuple>
#include <thrust/gather.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename v_to_gpu_id_op_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_vertices(raft::handle_t const& handle,
                 rmm::device_uvector<vertex_t>&& vertices,
                 std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                 v_to_gpu_id_op_t v_to_gpu_id_op,
                 std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size = handle.get_comms().get_size();

  auto element_size = sizeof(vertex_t);
  if (vertex_properties.size() == 1) {
    element_size +=
      cugraph::variant_type_dispatch(vertex_properties[0], cugraph::sizeof_arithmetic_element{});
  } else if (vertex_properties.size() > 1) {
    element_size += sizeof(size_t);
  }

  auto mem_frugal_threshold = std::numeric_limits<size_t>::max();
  if (!large_buffer_type) {
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr mem_frugal_ratio =
      0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
            // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
            // group-by by default, and thrust::sort requires temporary buffer comparable to the
            // input data size)
    mem_frugal_threshold =
      static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);
  }

  if (vertex_properties.size() == 0) {
    auto d_tx_value_counts          = cugraph::groupby_and_count(vertices.begin(),
                                                        vertices.end(),
                                                        v_to_gpu_id_op,
                                                        comm_size,
                                                        mem_frugal_threshold,
                                                        handle.get_stream(),
                                                        large_buffer_type);
    std::tie(vertices, std::ignore) = shuffle_values(
      handle.get_comms(),
      vertices.begin(),
      raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
      handle.get_stream(),
      large_buffer_type);
  } else if (vertex_properties.size() == 1) {
    rmm::device_uvector<size_t> d_tx_value_counts(0, handle.get_stream());
    cugraph::variant_type_dispatch(
      vertex_properties[0],
      [&handle,
       &vertices,
       &d_tx_value_counts,
       v_to_gpu_id_op,
       large_buffer_type,
       comm_size,
       mem_frugal_threshold](auto& prop) {
        d_tx_value_counts           = cugraph::groupby_and_count(vertices.begin(),
                                                       vertices.end(),
                                                       prop.begin(),
                                                       v_to_gpu_id_op,
                                                       comm_size,
                                                       mem_frugal_threshold,
                                                       handle.get_stream(),
                                                       large_buffer_type);
        std::tie(prop, std::ignore) = shuffle_values(
          handle.get_comms(),
          prop.begin(),
          raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
          handle.get_stream(),
          large_buffer_type);
      });
    std::tie(vertices, std::ignore) = shuffle_values(
      handle.get_comms(),
      vertices.begin(),
      raft::device_span<size_t const>(d_tx_value_counts.data(), d_tx_value_counts.size()),
      handle.get_stream(),
      large_buffer_type);
  } else {
    auto property_positions =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<size_t>(vertices.size(), handle.get_stream())
        : rmm::device_uvector<size_t>(vertices.size(), handle.get_stream());
    thrust::sequence(
      handle.get_thrust_policy(), property_positions.begin(), property_positions.end(), size_t{0});

    auto d_tx_value_counts = cugraph::groupby_and_count(vertices.begin(),
                                                        vertices.end(),
                                                        property_positions.begin(),
                                                        v_to_gpu_id_op,
                                                        comm_size,
                                                        mem_frugal_threshold,
                                                        handle.get_stream(),
                                                        large_buffer_type);

    raft::device_span<size_t const> d_tx_value_counts_span(d_tx_value_counts.data(),
                                                           d_tx_value_counts.size());

    std::tie(vertices, std::ignore) = shuffle_values(handle.get_comms(),
                                                     vertices.begin(),
                                                     d_tx_value_counts_span,
                                                     handle.get_stream(),
                                                     large_buffer_type);

    std::for_each(
      vertex_properties.begin(),
      vertex_properties.end(),
      [&handle, &property_positions, d_tx_value_counts_span, large_buffer_type](auto& property) {
        cugraph::variant_type_dispatch(
          property,
          [&handle, &property_positions, d_tx_value_counts_span, large_buffer_type](auto& prop) {
            using T  = typename std::remove_reference<decltype(prop)>::type::value_type;
            auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<T>(
                                             prop.size(), handle.get_stream())
                                         : rmm::device_uvector<T>(prop.size(), handle.get_stream());
            thrust::gather(handle.get_thrust_policy(),
                           property_positions.begin(),
                           property_positions.end(),
                           prop.begin(),
                           tmp.begin());
            std::tie(prop, std::ignore) = shuffle_values(handle.get_comms(),
                                                         tmp.begin(),
                                                         d_tx_value_counts_span,
                                                         handle.get_stream(),
                                                         large_buffer_type);
          });
      });
  }

  return std::make_tuple(std::move(vertices), std::move(vertex_properties));
}

}  // namespace detail

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_ext_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                     std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size       = handle.get_comms().get_size();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  return detail::shuffle_vertices(handle,
                                  std::move(vertices),
                                  std::move(vertex_properties),
                                  cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
                                    comm_size, major_comm_size, minor_comm_size},
                                  large_buffer_type);
}

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_int_vertices(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>&& vertices,
                     std::vector<arithmetic_device_uvector_t>&& vertex_properties,
                     raft::host_span<vertex_t const> vertex_partition_range_lasts,
                     std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size       = handle.get_comms().get_size();
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

  return detail::shuffle_vertices(
    handle,
    std::move(vertices),
    std::move(vertex_properties),
    cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
      raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                        d_vertex_partition_range_lasts.size()),
      major_comm_size,
      minor_comm_size},
    large_buffer_type);
}

}  // namespace cugraph
