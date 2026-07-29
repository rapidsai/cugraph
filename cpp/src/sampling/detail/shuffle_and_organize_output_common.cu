/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/export.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <optional>

namespace cugraph {
namespace detail {

CUGRAPH_EXPORT std::tuple<std::vector<cugraph::arithmetic_device_uvector_t>,
                          std::optional<rmm::device_uvector<int32_t>>,
                          std::optional<rmm::device_uvector<int32_t>>,
                          std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  std::vector<cugraph::arithmetic_device_uvector_t>&& property_edges,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<int32_t> input_hops,
  std::optional<raft::device_span<int32_t const>> output_labels,
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
      cugraph::sequence(rmm::exec_policy(handle.get_stream()),
                        property_position.data(),
                        property_position.data() + property_position.size(),
                        size_t{0});

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

              cugraph::gather(handle.get_thrust_policy(),
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
        cugraph::gather(handle.get_thrust_policy(),
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
    cugraph::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
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
          cugraph::gather(handle.get_thrust_policy(),
                          indices.begin(),
                          indices.end(),
                          edge_vector.begin(),
                          tmp.begin());

          edge_vector = std::move(tmp);
        });
      });

    CUGRAPH_EXPECTS(
      output_labels.has_value(),
      "Invalid input arguments: output_labels is required whenever labels are specified, "
      "since the offsets array delineates one range per label");

    // Searching the labels this GPU is responsible for rather than the labels present in the output
    // gives a label that sampled no edges a zero-width range instead of dropping it and shifting
    // every later label down.  A label can legitimately end up empty (for temporal sampling, when
    // no edge incident on its seeds satisfies the time window), and an entirely empty result still
    // has to produce output_labels->size() + 1 entries, all zero.
    offsets = rmm::device_uvector<size_t>(output_labels->size() + 1, handle.get_stream());

    thrust::lower_bound(handle.get_thrust_policy(),
                        labels->begin(),
                        labels->end(),
                        output_labels->begin(),
                        output_labels->end(),
                        offsets->begin());

    size_t last_offset = labels->size();
    offsets->set_element_async(output_labels->size(), last_offset, handle.get_stream());
    handle.sync_stream();
  }

  return std::make_tuple(
    std::move(property_edges), std::move(labels), std::move(hops), std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph
