/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/random/rng.cuh>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <tuple>

namespace cugraph {

namespace detail {

template <typename ValueIterator>
std::tuple<rmm::device_uvector<size_t> /* indices */,
           std::vector<size_t> /* offsets (size = value_offsets.size()) */>
partition_v_frontier(raft::handle_t const& handle,
                     ValueIterator frontier_value_first,
                     ValueIterator frontier_value_last,
                     std::vector<typename thrust::iterator_traits<ValueIterator>::value_type> const&
                       thresholds /* size = # partitions - 1, thresholds[i] marks the end
                                     (exclusive) of the i'th partition value range */
)
{
  rmm::device_uvector<size_t> indices(
    cuda::std::distance(frontier_value_first, frontier_value_last), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});

  auto num_partitions = thresholds.size() + 1;
  std::vector<size_t> v_frontier_partition_offsets(num_partitions + 1);
  v_frontier_partition_offsets[0] = size_t{0};
  v_frontier_partition_offsets.back() =
    static_cast<size_t>(cuda::std::distance(frontier_value_first, frontier_value_last));

  auto index_first = indices.begin();
  auto index_last  = indices.end();
  for (size_t i = 0; i < thresholds.size(); ++i) {
    auto false_first =
      thrust::partition(handle.get_thrust_policy(),
                        index_first,
                        index_last,
                        [frontier_value_first, threshold = thresholds[i]] __device__(size_t idx) {
                          return *(frontier_value_first + idx) < threshold;
                        });
    v_frontier_partition_offsets[1 + i] =
      v_frontier_partition_offsets[i] + cuda::std::distance(index_first, false_first);
    index_first = false_first;
  }

  return std::make_tuple(std::move(indices), std::move(v_frontier_partition_offsets));
}

// a key in the frontier has @p num_values_per_key values, the frontier is separately partitioned
// @p num_values_per_key times based on the i'th value; i = [0, @p num_values_per_key).
template <typename value_idx_t, typename ValueIterator>
std::tuple<rmm::device_uvector<size_t> /* indices */,
           rmm::device_uvector<value_idx_t>,
           std::vector<size_t> /* offsets (size = value_offsets.size()) */>
partition_v_frontier_per_value_idx(
  raft::handle_t const& handle,
  ValueIterator frontier_value_first,
  ValueIterator frontier_value_last,
  raft::host_span<typename thrust::iterator_traits<ValueIterator>::value_type const>
    thresholds /* size = num_values_per_key * (# partitions - 1), thresholds[i] marks the end
                  (exclusive) of the (i % num_values_per_key)'th partition value range for the (i /
                  num_values_per_key)'th value of each key */
  ,
  size_t num_values_per_key)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  assert((cuda::std::distance(frontier_value_first, frontier_value_last) % num_values_per_key) ==
         0);
  rmm::device_uvector<size_t> key_indices(
    cuda::std::distance(frontier_value_first, frontier_value_last), handle.get_stream());
  rmm::device_uvector<value_idx_t> value_indices(key_indices.size(), handle.get_stream());
  auto index_pair_first = thrust::make_zip_iterator(key_indices.begin(), value_indices.begin());
  auto index_pair_last  = thrust::make_zip_iterator(key_indices.end(), value_indices.end());
  thrust::tabulate(handle.get_thrust_policy(),
                   index_pair_first,
                   index_pair_last,
                   [num_values_per_key] __device__(size_t i) {
                     return thrust::make_tuple(i / num_values_per_key,
                                               static_cast<value_idx_t>(i % num_values_per_key));
                   });

  auto num_partitions = thresholds.size() / num_values_per_key + 1;
  std::vector<size_t> v_frontier_partition_offsets(num_partitions + 1);
  v_frontier_partition_offsets[0] = size_t{0};
  v_frontier_partition_offsets.back() =
    static_cast<size_t>(cuda::std::distance(frontier_value_first, frontier_value_last));

  rmm::device_uvector<value_t> d_thresholds(thresholds.size(), handle.get_stream());
  raft::update_device(
    d_thresholds.data(), thresholds.data(), thresholds.size(), handle.get_stream());
  for (size_t i = 0; i < num_partitions - 1; ++i) {
    auto false_first = thrust::partition(
      handle.get_thrust_policy(),
      index_pair_first,
      index_pair_last,
      [frontier_value_first,
       thresholds = raft::device_span<value_t>(d_thresholds.data(), d_thresholds.size()),
       num_values_per_key,
       num_partitions,
       true_partition_idx = i] __device__(auto pair) {
        auto key_idx   = thrust::get<0>(pair);
        auto value_idx = thrust::get<1>(pair);
        return *(frontier_value_first + key_idx * num_values_per_key + value_idx) <
               thresholds[value_idx * (num_partitions - 1) + true_partition_idx];
      });
    v_frontier_partition_offsets[1 + i] =
      v_frontier_partition_offsets[i] + cuda::std::distance(index_pair_first, false_first);
    index_pair_first = false_first;
  }

  return std::make_tuple(
    std::move(key_indices), std::move(value_indices), std::move(v_frontier_partition_offsets));
}

}  // namespace detail

}  // namespace cugraph
