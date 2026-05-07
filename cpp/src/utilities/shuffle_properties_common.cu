/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/shuffle_wrappers.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <optional>
#include <vector>

namespace cugraph {

std::vector<arithmetic_device_uvector_t> shuffle_properties(
  raft::handle_t const& handle,
  rmm::device_uvector<int>&& gpus,
  std::vector<arithmetic_device_uvector_t>&& properties,
  std::optional<large_buffer_type_t> large_buffer_type)
{
  auto const comm_size = handle.get_comms().get_size();

  if (properties.size() == 0) {
    gpus.resize(0, handle.get_stream());
    gpus.shrink_to_fit(handle.get_stream());
  } else if (properties.size() == 1) {
    cugraph::variant_type_dispatch(properties[0], [&handle, &gpus](auto& prop) {
      thrust::sort_by_key(handle.get_thrust_policy(), gpus.begin(), gpus.end(), prop.begin());
    });

    rmm::device_uvector<size_t> tx_counts(comm_size, handle.get_stream());
    {
      rmm::device_uvector<size_t> lasts(comm_size, handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          gpus.begin(),
                          gpus.end(),
                          thrust::make_counting_iterator(int{0}),
                          thrust::make_counting_iterator(comm_size),
                          lasts.begin());
      gpus.resize(0, handle.get_stream());
      gpus.shrink_to_fit(handle.get_stream());
      thrust::adjacent_difference(
        handle.get_thrust_policy(), lasts.begin(), lasts.end(), tx_counts.begin());
    }

    cugraph::variant_type_dispatch(
      properties[0], [&handle, &tx_counts, large_buffer_type](auto& prop) {
        std::tie(prop, std::ignore) =
          shuffle_values(handle.get_comms(),
                         prop.begin(),
                         raft::device_span<size_t const>(tx_counts.data(), tx_counts.size()),
                         handle.get_stream(),
                         large_buffer_type);
      });
  } else {
    rmm::device_uvector<size_t> property_positions =
      large_buffer_type
        ? large_buffer_manager::allocate_memory_buffer<size_t>(gpus.size(), handle.get_stream())
        : rmm::device_uvector<size_t>(gpus.size(), handle.get_stream());
    thrust::sequence(
      handle.get_thrust_policy(), property_positions.begin(), property_positions.end(), size_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), gpus.begin(), gpus.end(), property_positions.begin());

    rmm::device_uvector<size_t> tx_counts(comm_size, handle.get_stream());
    {
      rmm::device_uvector<size_t> lasts(comm_size, handle.get_stream());
      thrust::upper_bound(handle.get_thrust_policy(),
                          gpus.begin(),
                          gpus.end(),
                          thrust::make_counting_iterator(int{0}),
                          thrust::make_counting_iterator(comm_size),
                          lasts.begin());
      gpus.resize(0, handle.get_stream());
      gpus.shrink_to_fit(handle.get_stream());
      thrust::adjacent_difference(
        handle.get_thrust_policy(), lasts.begin(), lasts.end(), tx_counts.begin());
    }

    std::for_each(
      properties.begin(),
      properties.end(),
      [&handle, &property_positions, &tx_counts, large_buffer_type](auto& property) {
        cugraph::variant_type_dispatch(
          property, [&handle, &property_positions, &tx_counts, large_buffer_type](auto& prop) {
            using T  = typename std::remove_reference<decltype(prop)>::type::value_type;
            auto tmp = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<T>(
                                             prop.size(), handle.get_stream())
                                         : rmm::device_uvector<T>(prop.size(), handle.get_stream());
            thrust::gather(handle.get_thrust_policy(),
                           property_positions.begin(),
                           property_positions.end(),
                           prop.begin(),
                           tmp.begin());
            std::tie(prop, std::ignore) =
              shuffle_values(handle.get_comms(),
                             tmp.begin(),
                             raft::device_span<size_t const>(tx_counts.data(), tx_counts.size()),
                             handle.get_stream(),
                             large_buffer_type);
          });
      });
  }

  return std::move(properties);
}

}  // namespace cugraph
