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

#include <raft/core/handle.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda/std/iterator>

#include <numeric>
#include <vector>

namespace cugraph {

namespace detail {

inline std::vector<size_t> init_stream_pool_indices(size_t max_tmp_buffer_size,
                                                    size_t approx_tmp_buffer_size_per_loop,
                                                    size_t loop_count,
                                                    size_t num_streams_per_loop,
                                                    size_t max_streams)
{
  size_t num_streams = std::min(loop_count * num_streams_per_loop,
                                raft::round_down_safe(max_streams, num_streams_per_loop));

  auto num_concurrent_loops =
    (approx_tmp_buffer_size_per_loop > 0)
      ? std::max(max_tmp_buffer_size / approx_tmp_buffer_size_per_loop, size_t{1})
      : loop_count;
  num_streams = std::min(num_concurrent_loops * num_streams_per_loop, num_streams);

  std::vector<size_t> stream_pool_indices(num_streams);
  std::iota(stream_pool_indices.begin(), stream_pool_indices.end(), size_t{0});

  return stream_pool_indices;
}

// this assumes that the caller already knows how many items will be copied.
template <typename InputIterator, typename FlagIterator, typename OutputIterator>
void copy_if_nosync(InputIterator input_first,
                    InputIterator input_last,
                    FlagIterator flag_first,
                    OutputIterator output_first,
                    raft::device_span<size_t> count /* size = 1 */,
                    rmm::cuda_stream_view stream_view)
{
  CUGRAPH_EXPECTS(
    static_cast<size_t>(cuda::std::distance(input_first, input_last)) <=
      static_cast<size_t>(std::numeric_limits<int>::max()),
    "cugraph::detail::copy_if_nosync relies on cub::DeviceSelect::Flagged which uses int for input "
    "size, but cuda::std::distance(input_first, input_last) exceeds "
    "std::numeric_limits<int>::max().");

  size_t tmp_storage_bytes{0};
  size_t input_size = static_cast<int>(cuda::std::distance(input_first, input_last));

  cub::DeviceSelect::Flagged(static_cast<void*>(nullptr),
                             tmp_storage_bytes,
                             input_first,
                             flag_first,
                             output_first,
                             count.data(),
                             input_size,
                             stream_view);

  auto d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, stream_view);

  cub::DeviceSelect::Flagged(d_tmp_storage.data(),
                             tmp_storage_bytes,
                             input_first,
                             flag_first,
                             output_first,
                             count.data(),
                             input_size,
                             stream_view);
}

template <typename InputIterator>
void count_nosync(InputIterator input_first,
                  InputIterator input_last,
                  raft::device_span<size_t> count /* size = 1 */,
                  typename thrust::iterator_traits<InputIterator>::value_type value,
                  rmm::cuda_stream_view stream_view)
{
  CUGRAPH_EXPECTS(
    static_cast<size_t>(cuda::std::distance(input_first, input_last)) <=
      static_cast<size_t>(std::numeric_limits<int>::max()),
    "cugraph::detail::count_nosync relies on cub::DeviceReduce::Sum which uses int for input size, "
    "but cuda::std::distance(input_first, input_last) exceeds std::numeric_limits<int>::max().");

  size_t tmp_storage_bytes{0};
  size_t input_size = static_cast<int>(cuda::std::distance(input_first, input_last));

  cub::DeviceReduce::Sum(static_cast<void*>(nullptr),
                         tmp_storage_bytes,
                         input_first,
                         count.data(),
                         input_size,
                         stream_view);

  auto d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, stream_view);

  cub::DeviceReduce::Sum(
    d_tmp_storage.data(), tmp_storage_bytes, input_first, count.data(), input_size, stream_view);
}

template <typename InputIterator>
void sum_nosync(
  InputIterator input_first,
  InputIterator input_last,
  raft::device_span<typename thrust::iterator_traits<InputIterator>::value_type> sum /* size = 1 */,
  rmm::cuda_stream_view stream_view)
{
  CUGRAPH_EXPECTS(
    static_cast<size_t>(cuda::std::distance(input_first, input_last)) <=
      static_cast<size_t>(std::numeric_limits<int>::max()),
    "cugraph::detail::count_nosync relies on cub::DeviceReduce::Sum which uses int for input size, "
    "but cuda::std::distance(input_first, input_last) exceeds std::numeric_limits<int>::max().");

  size_t tmp_storage_bytes{0};
  size_t input_size = static_cast<int>(cuda::std::distance(input_first, input_last));

  cub::DeviceReduce::Sum(static_cast<void*>(nullptr),
                         tmp_storage_bytes,
                         input_first,
                         sum.data(),
                         input_size,
                         stream_view);

  auto d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, stream_view);

  cub::DeviceReduce::Sum(
    d_tmp_storage.data(), tmp_storage_bytes, input_first, sum.data(), input_size, stream_view);
}

}  // namespace detail

}  // namespace cugraph
