/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

template <typename ValueIterator>
std::tuple<rmm::device_uvector<size_t> /* indices */, std::vector<size_t> /* offsets (size = value_offsets.size()) */>
partition_v_frontier(raft::handle_t const& handle,
  ValueIterator frontier_value_first,
  ValueIterator frontier_value_last,
  std::vector<typename thrust::iterator_traits<ValueIterator>::value_type> const& thresholds /* size = # partitions - 1 */
) {
  rmm::device_uvector<size_t> indices(thrust::distance(frontier_value_first, frontier_value_last), handle.get_stream());
  thrust::sequence(
    handle.get_thrust_policy(), indices.begin(), indices.end(), size_t{0});
  std::vector<size_t> v_frontier_partition_offsets(thresholds.size() + 2);
  v_frontier_partition_offsets[0] = size_t{0};
  v_frontier_partition_offsets.back() = static_cast<size_t>(thrust::distance(frontier_value_first, frontier_value_last));

  auto index_first = indices.begin();
  auto index_last = indices.end();
  for (size_t i = 0; i < thresholds.size(); ++i) {
    auto false_first = thrust::partition(
        handle.get_thrust_policy(),
        index_first,
        index_last,
        [frontier_value_first, threshold = thresholds[i]]__device__(size_t idx) {
          return *(frontier_value_first + idx) < threshold;
        });
    v_frontier_partition_offsets[1 + i] = v_frontier_partition_offsets[i] + thrust::distance(index_first, false_first);
    index_first = false_first;
  }

  return std::make_tuple(std::move(indices), std::move(v_frontier_partition_offsets));
}

}  // namespace detail

}  // namespace cugraph
