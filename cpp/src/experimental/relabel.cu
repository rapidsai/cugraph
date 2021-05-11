/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <experimental/include_cuco_static_map.cuh>

#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph.hpp>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <utilities/error.hpp>
#include <utilities/shuffle_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <iterator>
#include <numeric>
#include <tuple>
#include <utility>

namespace cugraph {
namespace experimental {

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
  double constexpr load_factor = 0.7;

  // FIXME: remove this check once we drop Pascal support
  CUGRAPH_EXPECTS(handle.get_device_properties().major >= 7,
                  "Relabel not supported on Pascal and older architectures.");

#ifdef CUCO_STATIC_MAP_DEFINED
  if (multi_gpu) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    auto key_func = detail::compute_gpu_id_from_vertex_t<vertex_t>{comm_size};

    // find unique old labels (to be relabeled)

    rmm::device_uvector<vertex_t> unique_old_labels(num_labels, handle.get_stream());
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 labels,
                 labels + num_labels,
                 unique_old_labels.data());
    thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 unique_old_labels.begin(),
                 unique_old_labels.end());
    unique_old_labels.resize(
      thrust::distance(
        unique_old_labels.begin(),
        thrust::unique(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                       unique_old_labels.begin(),
                       unique_old_labels.end())),
      handle.get_stream());
    unique_old_labels.shrink_to_fit(handle.get_stream());

    // collect new labels for the unique old labels

    rmm::device_uvector<vertex_t> new_labels_for_unique_old_labels(0, handle.get_stream());
    {
      // shuffle the old_new_label_pairs based on applying the compute_gpu_id_from_vertex_t functor
      // to the old labels

      rmm::device_uvector<vertex_t> rx_label_pair_old_labels(0, handle.get_stream());
      rmm::device_uvector<vertex_t> rx_label_pair_new_labels(0, handle.get_stream());
      {
        rmm::device_uvector<vertex_t> label_pair_old_labels(num_label_pairs, handle.get_stream());
        rmm::device_uvector<vertex_t> label_pair_new_labels(num_label_pairs, handle.get_stream());
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     std::get<0>(old_new_label_pairs),
                     std::get<0>(old_new_label_pairs) + num_label_pairs,
                     label_pair_old_labels.begin());
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     std::get<1>(old_new_label_pairs),
                     std::get<1>(old_new_label_pairs) + num_label_pairs,
                     label_pair_new_labels.begin());
        auto pair_first = thrust::make_zip_iterator(
          thrust::make_tuple(label_pair_old_labels.begin(), label_pair_new_labels.begin()));
        std::forward_as_tuple(std::tie(rx_label_pair_old_labels, rx_label_pair_new_labels),
                              std::ignore) =
          groupby_gpuid_and_shuffle_values(
            handle.get_comms(),
            pair_first,
            pair_first + num_label_pairs,
            [key_func] __device__(auto val) { return key_func(thrust::get<0>(val)); },
            handle.get_stream());
      }

      // update intermediate relabel map

      CUDA_TRY(cudaStreamSynchronize(
        handle.get_stream()));  // cuco::static_map currently does not take stream

      cuco::static_map<vertex_t, vertex_t> relabel_map{
        // cuco::static_map requires at least one empty slot
        std::max(
          static_cast<size_t>(static_cast<double>(rx_label_pair_old_labels.size()) / load_factor),
          rx_label_pair_old_labels.size() + 1),
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
        rmm::device_uvector<vertex_t> rx_unique_old_labels(0, handle.get_stream());
        std::vector<size_t> rx_value_counts{};
        std::tie(rx_unique_old_labels, rx_value_counts) = groupby_gpuid_and_shuffle_values(
          handle.get_comms(),
          unique_old_labels.begin(),
          unique_old_labels.end(),
          [key_func] __device__(auto val) { return key_func(val); },
          handle.get_stream());

        CUDA_TRY(cudaStreamSynchronize(
          handle.get_stream()));  // cuco::static_map currently does not take stream

        if (skip_missing_labels) {
          thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                            rx_unique_old_labels.begin(),
                            rx_unique_old_labels.end(),
                            rx_unique_old_labels.begin(),
                            [view = relabel_map.get_device_view()] __device__(auto old_label) {
                              auto found = view.find(old_label);
                              return found != view.end() ? view.find(old_label)->second.load(
                                                             cuda::std::memory_order_relaxed)
                                                         : old_label;
                            });
        } else {
          relabel_map.find(
            rx_unique_old_labels.begin(),
            rx_unique_old_labels.end(),
            rx_unique_old_labels.begin());  // now rx_unique_old_lables hold new labels for the
                                            // corresponding old labels
        }

        std::tie(new_labels_for_unique_old_labels, std::ignore) = shuffle_values(
          handle.get_comms(), rx_unique_old_labels.begin(), rx_value_counts, handle.get_stream());
      }
    }

    handle.get_stream_view().synchronize();  // cuco::static_map currently does not take stream

    cuco::static_map<vertex_t, vertex_t> relabel_map(
      // cuco::static_map requires at least one empty slot
      std::max(static_cast<size_t>(static_cast<double>(unique_old_labels.size()) / load_factor),
               unique_old_labels.size() + 1),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(unique_old_labels.begin(), new_labels_for_unique_old_labels.begin())),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    relabel_map.insert(pair_first, pair_first + unique_old_labels.size());
    relabel_map.find(labels, labels + num_labels, labels);
  } else {
    cuco::static_map<vertex_t, vertex_t> relabel_map(
      // cuco::static_map requires at least one empty slot
      std::max(static_cast<size_t>(static_cast<double>(num_label_pairs) / load_factor),
               static_cast<size_t>(num_label_pairs) + 1),
      invalid_vertex_id<vertex_t>::value,
      invalid_vertex_id<vertex_t>::value);

    auto pair_first = thrust::make_transform_iterator(
      thrust::make_zip_iterator(
        thrust::make_tuple(std::get<0>(old_new_label_pairs), std::get<1>(old_new_label_pairs))),
      [] __device__(auto val) {
        return thrust::make_pair(thrust::get<0>(val), thrust::get<1>(val));
      });

    relabel_map.insert(pair_first, pair_first + num_label_pairs);
    if (skip_missing_labels) {
      thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        labels,
                        labels + num_labels,
                        labels,
                        [view = relabel_map.get_device_view()] __device__(auto old_label) {
                          auto found = view.find(old_label);
                          return found != view.end() ? view.find(old_label)->second.load(
                                                         cuda::std::memory_order_relaxed)
                                                     : old_label;
                        });
    } else {
      relabel_map.find(labels, labels + num_labels, labels);
    }
  }

  if (do_expensive_check && !skip_missing_labels) {
    CUGRAPH_EXPECTS(
      thrust::count(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                    labels,
                    labels + num_labels,
                    invalid_vertex_id<vertex_t>::value) == 0,
      "Invalid input argument: labels include old label values missing in old_new_label_pairs.");
  }
#endif

  return;
}

// explicit instantiation

template void relabel<int32_t, true>(raft::handle_t const& handle,
                                     std::tuple<int32_t const*, int32_t const*> old_new_label_pairs,
                                     int32_t num_label_pairs,
                                     int32_t* labels,
                                     int32_t num_labels,
                                     bool skip_missing_labels,
                                     bool do_expensive_check);

template void relabel<int32_t, false>(
  raft::handle_t const& handle,
  std::tuple<int32_t const*, int32_t const*> old_new_label_pairs,
  int32_t num_label_pairs,
  int32_t* labels,
  int32_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

template void relabel<int64_t, true>(raft::handle_t const& handle,
                                     std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
                                     int64_t num_label_pairs,
                                     int64_t* labels,
                                     int64_t num_labels,
                                     bool skip_missing_labels,
                                     bool do_expensive_check);

template void relabel<int64_t, false>(
  raft::handle_t const& handle,
  std::tuple<int64_t const*, int64_t const*> old_new_label_pairs,
  int64_t num_label_pairs,
  int64_t* labels,
  int64_t num_labels,
  bool skip_missing_labels,
  bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
