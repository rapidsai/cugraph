/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/utilities/device_functors.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sort.h>

#include <optional>

namespace cugraph {
namespace detail {

std::tuple<std::vector<cugraph::arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  std::vector<cugraph::arithmetic_device_uvector_t>&& edges_with_properties,
  std::optional<size_t> hop_index,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    if (label_to_output_comm_rank) {
      std::tie(*labels, edges_with_properties) = cugraph::shuffle_keys_with_properties(
        handle,
        std::move(*labels),
        std::move(edges_with_properties),
        shuffle_to_output_comm_rank_t<int32_t>{*label_to_output_comm_rank});
    }

    // Sort the tuples by hop/label
    rmm::device_uvector<size_t> indices(labels->size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
    rmm::device_uvector<int32_t> tmp_labels(indices.size(), handle.get_stream());

    if (hop_index) {
      auto& hop_v = std::get<rmm::device_uvector<int32_t>>(edges_with_properties[*hop_index]);
      rmm::device_uvector<int32_t> tmp_hops(indices.size(), handle.get_stream());

      thrust::sort(
        handle.get_thrust_policy(),
        indices.begin(),
        indices.end(),
        [labels = raft::device_span<int32_t const>(labels->data(), labels->size()),
         hops   = raft::device_span<int32_t const>(hop_v.data(), hop_v.size())] __device__(size_t l,
                                                                                         size_t r) {
          return thrust::make_tuple(labels[l], hops[l]) < thrust::make_tuple(labels[r], hops[r]);
        });
      thrust::gather(handle.get_thrust_policy(),
                     indices.begin(),
                     indices.end(),
                     thrust::make_zip_iterator(labels->begin(), hop_v.begin()),
                     thrust::make_zip_iterator(tmp_labels.begin(), tmp_hops.begin()));
      edges_with_properties[*hop_index] = std::move(tmp_hops);
    } else {
      thrust::sort(
        handle.get_thrust_policy(),
        indices.begin(),
        indices.end(),
        [labels = raft::device_span<int32_t const>(labels->data(), labels->size())] __device__(
          size_t l, size_t r) { return labels[l] < labels[r]; });
      thrust::gather(handle.get_thrust_policy(),
                     indices.begin(),
                     indices.end(),
                     labels->begin(),
                     tmp_labels.begin());
    }
    *labels = std::move(tmp_labels);

    for (size_t i = 0; i < edges_with_properties.size(); ++i) {
      if ((!hop_index) || (*hop_index != i)) {
        cugraph::variant_type_dispatch(
          edges_with_properties[i], [&handle, &indices](auto& edge_vector) {
            using T = typename std::remove_reference<decltype(edge_vector)>::type::value_type;
            rmm::device_uvector<T> tmp(indices.size(), handle.get_stream());
            thrust::gather(handle.get_thrust_policy(),
                           indices.begin(),
                           indices.end(),
                           edge_vector.begin(),
                           tmp.begin());

            edge_vector = std::move(tmp);
          });
      }
    }

    size_t num_unique_labels =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>(labels->size()),
                       is_first_in_run_t<int32_t const*>{labels->data()});

    rmm::device_uvector<int32_t> unique_labels(num_unique_labels, handle.get_stream());
    offsets = rmm::device_uvector<size_t>(num_unique_labels + 1, handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          labels->begin(),
                          labels->end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_labels.begin(),
                          offsets->begin());

    thrust::exclusive_scan(
      handle.get_thrust_policy(), offsets->begin(), offsets->end(), offsets->begin());
    labels = std::move(unique_labels);
  }

  return std::make_tuple(std::move(edges_with_properties), std::move(labels), std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph
