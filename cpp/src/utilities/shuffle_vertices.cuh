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

#include "cugraph/arithmetic_variant_types.hpp"
#include "detail/graph_partition_utils.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <thrust/gather.h>
#include <thrust/tuple.h>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  std::optional<large_buffer_type_t> large_buffer_type)
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
    handle.get_stream(),
    large_buffer_type);

  return std::move(vertices);
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values,
  std::optional<large_buffer_type_t> large_buffer_type)
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
    handle.get_stream(),
    large_buffer_type);

  return std::make_tuple(std::move(vertices), std::move(values));
}

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type)
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
    handle.get_stream(),
    large_buffer_type);

  return std::move(vertices);
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  rmm::device_uvector<value_t>&& values,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  std::optional<large_buffer_type_t> large_buffer_type)
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
    handle.get_stream(),
    large_buffer_type);

  return std::make_tuple(std::move(vertices), std::move(values));
}

}  // namespace detail

template <typename vertex_t>
rmm::device_uvector<vertex_t> shuffle_ext_vertices(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& vertices,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  return detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
    handle, std::move(vertices), large_buffer_type);
}

template <typename vertex_t, typename value_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<value_t>>
shuffle_ext_vertex_value_pairs(raft::handle_t const& handle,
                               rmm::device_uvector<vertex_t>&& vertices,
                               rmm::device_uvector<value_t>&& values,
                               std::optional<large_buffer_type_t> large_buffer_type)
{
  return detail::shuffle_ext_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
    handle, std::move(vertices), std::move(values), large_buffer_type);
}

template <typename key_t, typename key_to_gpu_op_t>
std::tuple<rmm::device_uvector<key_t>, std::vector<arithmetic_device_uvector_t>>
shuffle_keys_with_properties(raft::handle_t const& handle,
                             rmm::device_uvector<key_t>&& keys,
                             std::vector<arithmetic_device_uvector_t>&& properties,
                             key_to_gpu_op_t key_to_gpu_op,
                             std::optional<large_buffer_type_t> large_buffer_type)
{
  if (properties.size() == 0) {
    std::tie(keys, std::ignore) = cugraph::groupby_gpu_id_and_shuffle_values(handle.get_comms(),
                                                                             keys.begin(),
                                                                             keys.end(),
                                                                             key_to_gpu_op,
                                                                             handle.get_stream(),
                                                                             large_buffer_type);
  } else if (properties.size() == 1) {
    std::tie(keys, properties[0]) = cugraph::variant_type_dispatch(
      properties[0], [&handle, &keys, &key_to_gpu_op, large_buffer_type](auto& property) {
        std::tie(keys, property, std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_kv_pairs(handle.get_comms(),
                                                       keys.begin(),
                                                       keys.end(),
                                                       property.begin(),
                                                       key_to_gpu_op,
                                                       handle.get_stream(),
                                                       large_buffer_type);
        arithmetic_device_uvector_t property_variant = std::move(property);
        return std::make_tuple(std::move(keys), std::move(property_variant));
      });
  } else {
    auto comm_size = handle.get_comms().get_size();
    size_t element_size{sizeof(key_t) + sizeof(size_t)};
    auto total_global_mem = handle.get_device_properties().totalGlobalMem;
    auto constexpr mem_frugal_ratio =
      0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
            // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
            // group-by by default, and thrust::sort requires temporary buffer comparable to the
            // input data size)
    auto mem_frugal_threshold =
      static_cast<size_t>(static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

    auto property_position =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<size_t>(keys.size(), handle.get_stream())
        : rmm::device_uvector<size_t>(keys.size(), handle.get_stream());
    detail::sequence_fill(
      handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

    auto d_tx_value_counts = cugraph::groupby_and_count(keys.begin(),
                                                        keys.end(),
                                                        property_position.begin(),
                                                        key_to_gpu_op,
                                                        comm_size,
                                                        mem_frugal_threshold,
                                                        handle.get_stream(),
                                                        large_buffer_type);

    raft::device_span<size_t const> d_tx_value_counts_span{d_tx_value_counts.data(),
                                                           d_tx_value_counts.size()};

    std::tie(keys, std::ignore) = shuffle_values(handle.get_comms(),
                                                 keys.begin(),
                                                 d_tx_value_counts_span,
                                                 handle.get_stream(),
                                                 large_buffer_type);

    std::for_each(
      properties.begin(),
      properties.end(),
      [&handle, &property_position, d_tx_value_counts_span, large_buffer_type](auto& property) {
        cugraph::variant_type_dispatch(
          property,
          [&handle, &property_position, d_tx_value_counts_span, large_buffer_type](auto& prop) {
            using T  = typename std::remove_reference<decltype(prop)>::type::value_type;
            auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<T>(
                                             prop.size(), handle.get_stream())
                                         : rmm::device_uvector<T>(prop.size(), handle.get_stream());

            thrust::gather(handle.get_thrust_policy(),
                           property_position.begin(),
                           property_position.end(),
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

  return std::make_tuple(std::move(keys), std::move(properties));
}

}  // namespace cugraph
