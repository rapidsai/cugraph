/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/util/integer_utils.hpp>

#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

inline std::vector<size_t> init_stream_pool_indices(raft::handle_t const& handle,
                                             size_t max_tmp_buffer_size,
                                             size_t approx_tmp_buffer_size_per_edge_partition,
                                             size_t num_local_edge_partitions,
                                             size_t num_streams_per_edge_partition)
{
  size_t num_streams =
    std::min(num_local_edge_partitions * num_streams_per_edge_partition,
             raft::round_down_safe(handle.get_stream_pool_size(), num_streams_per_edge_partition));

  auto num_concurrent_loops =
    (approx_tmp_buffer_size_per_edge_partition > 0)
      ? std::max(max_tmp_buffer_size / approx_tmp_buffer_size_per_edge_partition, size_t{1})
      : num_local_edge_partitions;
  num_streams = std::min(num_concurrent_loops * num_streams_per_edge_partition, num_streams);

  std::vector<size_t> stream_pool_indices(num_streams);
  std::iota(stream_pool_indices.begin(), stream_pool_indices.end(), size_t{0});

  return stream_pool_indices;
}

}  // namespace detail

}  // namespace cugraph
