/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/graph_partition_utils.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include <optional>

namespace cugraph {
namespace detail {

std::tuple<std::vector<cugraph::arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  std::vector<cugraph::arithmetic_device_uvector_t>&& property_edges,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<int32_t> input_hops,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    if (label_to_output_comm_rank) {
      indirection_t<int32_t, int32_t const*> key_to_gpu_op{label_to_output_comm_rank->begin()};

      auto comm_size = handle.get_comms().get_size();
      size_t element_size{sizeof(int32_t) + sizeof(size_t)};
      auto total_global_mem = handle.get_device_properties().totalGlobalMem;
      auto constexpr mem_frugal_ratio =
        0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
              // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
              // group-by by default, and thrust::sort requires temporary buffer comparable to the
              // input data size)
      auto mem_frugal_threshold = static_cast<size_t>(
        static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

      rmm::device_uvector<size_t> property_position(labels->size(), handle.get_stream());
      detail::sequence_fill(
        handle.get_stream(), property_position.data(), property_position.size(), size_t{0});

      auto d_tx_value_counts = cugraph::groupby_and_count(labels->begin(),
                                                          labels->end(),
                                                          property_position.begin(),
                                                          key_to_gpu_op,
                                                          comm_size,
                                                          mem_frugal_threshold,
                                                          handle.get_stream());

      raft::device_span<size_t const> d_tx_value_counts_span{d_tx_value_counts.data(),
                                                             d_tx_value_counts.size()};

      std::tie(labels, std::ignore) = shuffle_values(
        handle.get_comms(), labels->begin(), d_tx_value_counts_span, handle.get_stream());

      std::for_each(
        property_edges.begin(),
        property_edges.end(),
        [&handle, &property_position, &d_tx_value_counts_span](auto& property) {
          cugraph::variant_type_dispatch(
            property, [&handle, &property_position, d_tx_value_counts_span](auto& prop) {
              using T = typename std::remove_reference<decltype(prop)>::type::value_type;
              rmm::device_uvector<T> tmp(prop.size(), handle.get_stream());

              thrust::gather(handle.get_thrust_policy(),
                             property_position.begin(),
                             property_position.end(),
                             prop.begin(),
                             tmp.begin());

              std::tie(prop, std::ignore) = shuffle_values(
                handle.get_comms(), tmp.begin(), d_tx_value_counts_span, handle.get_stream());
            });
        });

      if (hops) {
        rmm::device_uvector<int32_t> tmp(hops->size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       property_position.begin(),
                       property_position.end(),
                       hops->begin(),
                       tmp.begin());

        std::tie(*hops, std::ignore) = shuffle_values(
          handle.get_comms(), tmp.begin(), d_tx_value_counts_span, handle.get_stream());
      }
    }

    // Sort the tuples by hop/label
    rmm::device_uvector<size_t> indices(labels->size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
    if (hops) {
      thrust::sort_by_key(handle.get_thrust_policy(),
                          thrust::make_zip_iterator(labels->begin(), hops->begin()),
                          thrust::make_zip_iterator(labels->end(), hops->end()),
                          indices.begin());
    } else {
      thrust::sort_by_key(
        handle.get_thrust_policy(), labels->begin(), labels->end(), indices.begin());
    }

    std::for_each(
      property_edges.begin(), property_edges.end(), [&handle, &indices](auto& property) {
        cugraph::variant_type_dispatch(property, [&handle, &indices](auto& edge_vector) {
          using T = typename std::remove_reference<decltype(edge_vector)>::type::value_type;
          rmm::device_uvector<T> tmp(indices.size(), handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         indices.begin(),
                         indices.end(),
                         edge_vector.begin(),
                         tmp.begin());

          edge_vector = std::move(tmp);
        });
      });

    // Need to generate offsets for each unique label (not each seed) on each GPU
    rmm::device_uvector<int32_t> unique_labels(labels->size(), handle.get_stream());
    raft::copy(unique_labels.data(), labels->data(), labels->size(), handle.get_stream());
    thrust::sort(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
    auto unique_end =
      thrust::unique(handle.get_thrust_policy(), unique_labels.begin(), unique_labels.end());
    size_t num_unique_labels =
      static_cast<size_t>(thrust::distance(unique_labels.begin(), unique_end));

    unique_labels.resize(num_unique_labels, handle.get_stream());

    offsets = rmm::device_uvector<size_t>(unique_labels.size() + 1, handle.get_stream());

    thrust::lower_bound(handle.get_thrust_policy(),
                        labels->begin(),
                        labels->end(),
                        unique_labels.begin(),
                        unique_labels.end(),
                        offsets->begin());

    size_t last_offset = labels->size();
    offsets->set_element_async(unique_labels.size(), last_offset, handle.get_stream());
    handle.sync_stream();
  }

  return std::make_tuple(
    std::move(property_edges), std::move(labels), std::move(hops), std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph
