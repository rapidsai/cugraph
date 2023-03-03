/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <prims/kv_store.cuh>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {

// FIXME: think about requiring old_new_label_pairs to be pre-shuffled
template <typename vertex_t, bool multi_gpu>
void relabel(raft::handle_t const& handle,
             std::tuple<vertex_t const*, vertex_t const*> old_new_label_pairs,
             vertex_t num_label_pairs,
             vertex_t* labels /* [INOUT] */,
             vertex_t num_labels,
             bool skip_missing_labels,
             bool do_expensive_check)
{
  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    auto key_func = detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size};

    // find unique old labels (to be relabeled)

    rmm::device_uvector<vertex_t> unique_old_labels(num_labels, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), labels, labels + num_labels, unique_old_labels.data());
    thrust::sort(handle.get_thrust_policy(), unique_old_labels.begin(), unique_old_labels.end());
    unique_old_labels.resize(thrust::distance(unique_old_labels.begin(),
                                              thrust::unique(handle.get_thrust_policy(),
                                                             unique_old_labels.begin(),
                                                             unique_old_labels.end())),
                             handle.get_stream());
    unique_old_labels.shrink_to_fit(handle.get_stream());

    // collect new labels for the unique old labels

    rmm::device_uvector<vertex_t> new_labels_for_unique_old_labels(0, handle.get_stream());
    {
      // shuffle the old_new_label_pairs based on applying the compute_gpu_id_from_ext_vertex_t
      // functor to the old labels

      rmm::device_uvector<vertex_t> rx_label_pair_old_labels(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_label_pair_new_labels(0, handle.get_stream());
      {
        rmm::device_uvector<vertex_t> label_pair_old_labels(num_label_pairs, handle.get_stream());
        rmm::device_uvector<vertex_t> label_pair_new_labels(num_label_pairs, handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     std::get<0>(old_new_label_pairs),
                     std::get<0>(old_new_label_pairs) + num_label_pairs,
                     label_pair_old_labels.begin());
        thrust::copy(handle.get_thrust_policy(),
                     std::get<1>(old_new_label_pairs),
                     std::get<1>(old_new_label_pairs) + num_label_pairs,
                     label_pair_new_labels.begin());
        auto pair_first = thrust::make_zip_iterator(
          thrust::make_tuple(label_pair_old_labels.begin(), label_pair_new_labels.begin()));
        std::forward_as_tuple(std::tie(rx_label_pair_old_labels, rx_label_pair_new_labels),
                              std::ignore) =
          groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            pair_first,
            pair_first + num_label_pairs,
            [key_func] __device__(auto val) { return key_func(thrust::get<0>(val)); },
            handle.get_stream());
      }

      // update intermediate relabel map

      kv_store_t<vertex_t, vertex_t, false> relabel_map(
        rx_label_pair_old_labels.begin(),
        rx_label_pair_old_labels.begin() + rx_label_pair_old_labels.size(),
        rx_label_pair_new_labels.begin(),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto relabel_map_view = relabel_map.view();

      rx_label_pair_old_labels.resize(0, handle.get_stream());
      rx_label_pair_new_labels.resize(0, handle.get_stream());
      rx_label_pair_old_labels.shrink_to_fit(handle.get_stream());
      rx_label_pair_new_labels.shrink_to_fit(handle.get_stream());

      // shuffle unique_old_labels, relabel using the intermediate relabel map, and shuffle back

      {
        rmm::device_uvector<vertex_t> rx_unique_old_labels(0, handle.get_stream());
        std::vector<size_t> rx_value_counts{};
        std::tie(rx_unique_old_labels, rx_value_counts) = groupby_gpu_id_and_shuffle_values(
          handle.get_comms(),
          unique_old_labels.begin(),
          unique_old_labels.end(),
          [key_func] __device__(auto val) { return key_func(val); },
          handle.get_stream());

        if (skip_missing_labels) {
          auto device_view = detail::kv_cuco_store_device_view_t(relabel_map_view);
          thrust::transform(
            handle.get_thrust_policy(),
            rx_unique_old_labels.begin(),
            rx_unique_old_labels.end(),
            rx_unique_old_labels.begin(),
            [device_view,
             invalid_value = invalid_vertex_id<vertex_t>::value] __device__(auto old_label) {
              auto val = device_view.find(old_label);
              return val != invalid_value ? val : old_label;
            });
        } else {
          relabel_map_view.find(rx_unique_old_labels.begin(),
                                rx_unique_old_labels.end(),
                                rx_unique_old_labels.begin(),
                                handle.get_stream());  // now rx_unique_old_lables holds new labels
                                                       // for the corresponding old labels
        }

        std::tie(new_labels_for_unique_old_labels, std::ignore) = shuffle_values(
          handle.get_comms(), rx_unique_old_labels.begin(), rx_value_counts, handle.get_stream());
      }
    }

    {
      kv_store_t<vertex_t, vertex_t, false> relabel_map(
        unique_old_labels.begin(),
        unique_old_labels.begin() + unique_old_labels.size(),
        new_labels_for_unique_old_labels.begin(),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value,
        handle.get_stream());
      auto relabel_map_view = relabel_map.view();
      relabel_map_view.find(labels, labels + num_labels, labels, handle.get_stream());
    }
  } else {
    kv_store_t<vertex_t, vertex_t, false> relabel_map(
      std::get<0>(old_new_label_pairs),
      std::get<0>(old_new_label_pairs) + num_label_pairs,
      std::get<1>(old_new_label_pairs),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value,
      handle.get_stream());
    auto relabel_map_view = relabel_map.view();
    if (skip_missing_labels) {
      auto device_view = detail::kv_cuco_store_device_view_t(relabel_map_view);
      thrust::transform(
        handle.get_thrust_policy(),
        labels,
        labels + num_labels,
        labels,
        [device_view,
         invalid_value = invalid_vertex_id<vertex_t>::value] __device__(auto old_label) {
          auto val = device_view.find(old_label);
          return val != invalid_value ? val : old_label;
        });
    } else {
      relabel_map_view.find(labels, labels + num_labels, labels, handle.get_stream());
    }
  }

  if (do_expensive_check && !skip_missing_labels) {
    CUGRAPH_EXPECTS(
      thrust::count(handle.get_thrust_policy(),
                    labels,
                    labels + num_labels,
                    invalid_vertex_id<vertex_t>::value) == 0,
      "Invalid input argument: labels include old label values missing in old_new_label_pairs.");
  }

  return;
}

}  // namespace cugraph
