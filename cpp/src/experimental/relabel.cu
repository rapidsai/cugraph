/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <cuco/static_map.cuh>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace experimental {

template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> relabel(
  raft::handle_t const &handle,
  rmm::device_uvector<vertex_t> const &old_labels,
  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> const
    &old_new_label_pairs)
{
  double constexpr load_factor = 0.7;

  rmm::device_uvector<vertex_t> new_labels(0, handle.get_stream());

  if (multi_gpu) {
    auto &comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    auto key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size};

    // find unique old labels (to be relabeled)

    rmm::device_uvector<vertex_t> unique_old_labels(old_labels, handle.get_stream());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 unique_old_labels.begin(),
                 unique_old_labels.end());
    auto it = thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                             unique_old_labels.begin(),
                             unique_old_labels.end());
    unique_old_labels.resize(thrust::distance(unique_old_labels.begin(), it), handle.get_stream());
    unique_old_labels.shrink_to_fit(handle.get_stream());

    // collect new labels for the unique old labels

    rmm::device_uvector<vertex_t> new_labels_for_unique_old_labels(0, handle.get_stream());
    {
      // shuffle the old_new_label_pairs based on applying the compute_gpu_id_from_vertex_t functor
      // to the old labels

      rmm::device_uvector<vertex_t> rx_label_pair_old_labels(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_label_pair_new_labels(0, handle.get_stream());
      {
        rmm::device_uvector<vertex_t> label_pair_old_labels(std::get<0>(old_new_label_pairs),
                                                            handle.get_stream());
        rmm::device_uvector<vertex_t> label_pair_new_labels(std::get<1>(old_new_label_pairs),
                                                            handle.get_stream());
        auto pair_first = thrust::make_zip_iterator(
          thrust::make_tuple(label_pair_old_labels.begin(), label_pair_new_labels.begin()));
        thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     pair_first,
                     pair_first + std::get<0>(old_new_label_pairs).size(),
                     [key_func] __device__(auto lhs, auto rhs) {
                       return key_func(thrust::get<0>(lhs)) < key_func(thrust::get<0>(rhs));
                     });
        auto key_first = thrust::make_transform_iterator(
          label_pair_old_labels.begin(), [key_func] __device__(auto val) { return key_func(val); });
        rmm::device_uvector<size_t> tx_value_counts(comm_size, handle.get_stream());
        thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                              key_first,
                              key_first + label_pair_old_labels.size(),
                              thrust::make_constant_iterator(size_t{1}),
                              thrust::make_discard_iterator(),
                              tx_value_counts.begin());

        std::tie(rx_label_pair_old_labels, rx_label_pair_new_labels, std::ignore) =
          cugraph::experimental::detail::shuffle_values(
            handle.get_comms(), pair_first, tx_value_counts, handle.get_stream());

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // label_pair_old_labels and label_pair_new_labels will become
                                  // out-of-scope
      }

      // update intermediate relabel map

      cuco::static_map<vertex_t, vertex_t> relabel_map{
        static_cast<size_t>(static_cast<double>(rx_label_pair_old_labels.size()) / load_factor),
        invalid_vertex_id<vertex_t>::value,
        invalid_vertex_id<vertex_t>::value};

      auto pair_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
          thrust::make_tuple(rx_label_pair_old_labels.begin(), rx_label_pair_new_labels.begin())),
        [] __device__(auto val) {
          return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
        });
      relabel_map.insert(pair_first, pair_first + rx_label_pair_old_labels.size());

      rx_label_pair_old_labels.resize(0, handle.get_stream());
      rx_label_pair_new_labels.resize(0, handle.get_stream());
      rx_label_pair_old_labels.shrink_to_fit(handle.get_stream());
      rx_label_pair_new_labels.shrink_to_fit(handle.get_stream());

      // shuffle unique_old_labels, relabel using the intermediate relabel map, and shuffle back

      {
        thrust::sort(
          rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
          unique_old_labels.begin(),
          unique_old_labels.end(),
          [key_func] __device__(auto lhs, auto rhs) { return key_func(lhs) < key_func(rhs); });

        auto key_first = thrust::make_transform_iterator(
          unique_old_labels.begin(), [key_func] __device__(auto val) { return key_func(val); });
        rmm::device_uvector<size_t> tx_value_counts(comm_size, handle.get_stream());
        thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                              key_first,
                              key_first + unique_old_labels.size(),
                              thrust::make_constant_iterator(size_t{1}),
                              thrust::make_discard_iterator(),
                              tx_value_counts.begin());

        rmm::device_uvector<vertex_t> rx_unique_old_labels(0, handle.get_stream());
        rmm::device_uvector<size_t> rx_value_counts(0, handle.get_stream());

        std::tie(rx_unique_old_labels, rx_value_counts) =
          cugraph::experimental::detail::shuffle_values(
            handle.get_comms(), unique_old_labels.begin(), tx_value_counts, handle.get_stream());

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // cuco::static_map currently does not take stream

        relabel_map.find(
          rx_unique_old_labels.begin(),
          rx_unique_old_labels.end(),
          rx_unique_old_labels
            .begin());  // now rx_unique_old_lables hold new labels for the corresponding old labels

        std::tie(new_labels_for_unique_old_labels, std::ignore) =
          cugraph::experimental::detail::shuffle_values(
            handle.get_comms(), rx_unique_old_labels.begin(), rx_value_counts, handle.get_stream());

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // tx_value_counts & rx_value_counts will become out-of-scope
      }
    }

    cuco::static_map<vertex_t, vertex_t> relabel_map(
      static_cast<size_t>(static_cast<double>(unique_old_labels.size()) / load_factor),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(unique_old_labels.begin(), new_labels_for_unique_old_labels.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    relabel_map.insert(pair_first, pair_first + unique_old_labels.size());
    new_labels.resize(old_labels.size(), handle.get_stream());
    relabel_map.find(old_labels.begin(), old_labels.end(), new_labels.begin());
  } else {
    cuco::static_map<vertex_t, vertex_t> relabel_map(
      static_cast<size_t>(static_cast<double>(std::get<0>(old_new_label_pairs).size()) /
                          load_factor),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(old_new_label_pairs).begin(),
                                                   std::get<1>(old_new_label_pairs).begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    relabel_map.insert(pair_first, pair_first + std::get<0>(old_new_label_pairs).size());
    new_labels.resize(old_labels.size(), handle.get_stream());
    relabel_map.find(old_labels.begin(), old_labels.end(), new_labels.begin());
  }

  return std::move(new_labels);
}

// explicit instantiation

template rmm::device_uvector<int32_t> relabel<int32_t, true>(
  raft::handle_t const &handle,
  rmm::device_uvector<int32_t> const &old_labels,
  std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> const
    &old_new_label_pairs);

template rmm::device_uvector<int32_t> relabel<int32_t, false>(
  raft::handle_t const &handle,
  rmm::device_uvector<int32_t> const &old_labels,
  std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>> const
    &old_new_label_pairs);

}  // namespace experimental
}  // namespace cugraph
