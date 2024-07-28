/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "prims/detail/optional_dataframe_buffer.hpp"
#include "prims/detail/prim_functors.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr extract_transform_v_frontier_e_kernel_block_size = 512;

template <typename BufferKeyOutputIterator,
          typename BufferValueOutputIterator,
          typename e_op_result_t>
__device__ void push_buffer_element(BufferKeyOutputIterator buffer_key_output_first,
                                    BufferValueOutputIterator buffer_value_output_first,
                                    size_t buffer_idx,
                                    e_op_result_t e_op_result)
{
  using output_key_t =
    typename optional_dataframe_buffer_iterator_value_type_t<BufferKeyOutputIterator>::value;
  using output_value_t =
    typename optional_dataframe_buffer_iterator_value_type_t<BufferValueOutputIterator>::value;

  assert(e_op_result.has_value());

  static_assert(!std::is_same_v<output_key_t, void> || !std::is_same_v<output_value_t, void>);
  if constexpr (!std::is_same_v<output_key_t, void> && !std::is_same_v<output_value_t, void>) {
    *(buffer_key_output_first + buffer_idx)   = thrust::get<0>(*e_op_result);
    *(buffer_value_output_first + buffer_idx) = thrust::get<1>(*e_op_result);
  } else if constexpr (!std::is_same_v<output_key_t, void>) {
    *(buffer_key_output_first + buffer_idx) = *e_op_result;
  } else {
    *(buffer_value_output_first + buffer_idx) = *e_op_result;
  }
}

template <typename BufferKeyOutputIterator,
          typename BufferValueOutputIterator,
          typename e_op_result_t>
__device__ void warp_push_buffer_elements(
  BufferKeyOutputIterator buffer_key_output_first,
  BufferValueOutputIterator buffer_value_output_first,
  cuda::atomic_ref<size_t, cuda::thread_scope_device>& buffer_idx,
  int lane_id,
  e_op_result_t e_op_result)
{
  auto ballot = __ballot_sync(raft::warp_full_mask(), e_op_result ? uint32_t{1} : uint32_t{0});
  if (ballot > 0) {
    size_t warp_buffer_start_idx{};
    if (lane_id == 0) {
      auto increment        = __popc(ballot);
      warp_buffer_start_idx = buffer_idx.fetch_add(increment, cuda::std::memory_order_relaxed);
    }
    warp_buffer_start_idx = __shfl_sync(raft::warp_full_mask(), warp_buffer_start_idx, int{0});
    if (e_op_result) {
      auto buffer_warp_offset = __popc(ballot & ~(raft::warp_full_mask() << lane_id));
      push_buffer_element(buffer_key_output_first,
                          buffer_value_output_first,
                          warp_buffer_start_idx + buffer_warp_offset,
                          e_op_result);
    }
  }
}

template <bool hypersparse,
          bool max_one_e_per_frontier_key,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename BufferKeyOutputIterator,
          typename BufferValueOutputIterator,
          typename EdgeOp>
__global__ static void extract_transform_v_frontier_e_hypersparse_or_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferValueOutputIterator buffer_value_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using e_op_result_t =
    typename edge_op_result_type<key_t,
                                 typename GraphViewType::vertex_type,
                                 typename EdgePartitionSrcValueInputWrapper::value_type,
                                 typename EdgePartitionDstValueInputWrapper::value_type,
                                 typename EdgePartitionEdgeValueInputWrapper::value_type,
                                 EdgeOp>::type;

  auto const tid     = threadIdx.x + blockIdx.x * blockDim.x;
  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = tid % raft::warp_size();
  [[maybe_unused]] vertex_t major_start_offset{};  // relevant only when hypersparse is true
  if constexpr (hypersparse) {
    major_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                             edge_partition.major_range_first());
  }
  auto idx = static_cast<size_t>(tid);

  cuda::atomic_ref<size_t, cuda::thread_scope_device> buffer_idx(*buffer_idx_ptr);

  int32_t constexpr shared_array_size = max_one_e_per_frontier_key
                                          ? int32_t{1} /* dummy */
                                          : extract_transform_v_frontier_e_kernel_block_size;
  __shared__ std::conditional_t<max_one_e_per_frontier_key, std::byte /* dummy */, edge_t>
    warp_local_degree_inclusive_sums[shared_array_size];
  __shared__ std::conditional_t<max_one_e_per_frontier_key, std::byte /* dummy */, edge_t>
    warp_key_local_edge_offsets[shared_array_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ std::
    conditional_t<max_one_e_per_frontier_key, std::byte /* dummy */, typename WarpScan::TempStorage>
      temp_storage;

  auto indices = edge_partition.indices();

  vertex_t num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
  auto rounded_up_num_keys =
    ((static_cast<size_t>(num_keys) + (raft::warp_size() - 1)) / raft::warp_size()) *
    raft::warp_size();
  while (idx < rounded_up_num_keys) {
    auto call_e_op = call_e_op_with_key_t<GraphViewType,
                                          key_t,
                                          EdgePartitionSrcValueInputWrapper,
                                          EdgePartitionDstValueInputWrapper,
                                          EdgePartitionEdgeValueInputWrapper,
                                          EdgeOp>{edge_partition,
                                                  edge_partition_src_value_input,
                                                  edge_partition_dst_value_input,
                                                  edge_partition_e_value_input,
                                                  e_op};

    edge_t edge_offset{0};
    edge_t local_degree{0};
    if (idx < num_keys) {
      auto key   = *(key_first + idx);
      auto major = thrust_tuple_get_or_identity<key_t, 0>(key);
      if constexpr (hypersparse) {
        auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
        if (major_hypersparse_idx) {
          auto major_idx = major_start_offset + *major_hypersparse_idx;
          edge_offset    = edge_partition.local_offset(major_idx);
          local_degree   = edge_partition.local_degree(major_idx);
        }
      } else {
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        edge_offset       = edge_partition.local_offset(major_offset);
        local_degree      = edge_partition.local_degree(major_offset);
      }
    }

    if constexpr (max_one_e_per_frontier_key) {
      // each thread processes one frontier key, exits if any edge returns a valid output

      e_op_result_t e_op_result{thrust::nullopt};
      auto key = *(key_first + idx);

      if (edge_partition_e_mask) {
        for (edge_t i = 0; i < local_degree; ++i) {
          if ((*edge_partition_e_mask).get(edge_offset + i)) {
            e_op_result = call_e_op(key, edge_offset + i);
            if (e_op_result) { break; }
          }
        }
      } else {
        for (edge_t i = 0; i < local_degree; ++i) {
          e_op_result = call_e_op(key, edge_offset + i);
          if (e_op_result) { break; }
        }
      }
      warp_push_buffer_elements(
        buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
    } else {
      auto min_key_idx = static_cast<vertex_t>(idx - (idx % raft::warp_size()));  // inclusive
      auto max_key_idx =
        static_cast<vertex_t>(std::min(static_cast<size_t>(min_key_idx) + raft::warp_size(),
                                       static_cast<size_t>(num_keys)));  // exclusive

      // update warp_local_degree_inclusive_sums & warp_key_local_edge_offsets

      warp_key_local_edge_offsets[threadIdx.x] = edge_offset;
      WarpScan(temp_storage)
        .InclusiveSum(local_degree, warp_local_degree_inclusive_sums[threadIdx.x]);
      __syncwarp();

      // all the threads in a warp collectively process local edges for the keys in [key_first +
      // min_key_idx, key_first + max_key_idx)

      auto num_edges_this_warp = warp_local_degree_inclusive_sums[warp_id * raft::warp_size() +
                                                                  (max_key_idx - min_key_idx) - 1];
      auto rounded_up_num_edges_this_warp =
        ((static_cast<size_t>(num_edges_this_warp) + (raft::warp_size() - 1)) / raft::warp_size()) *
        raft::warp_size();

      auto this_warp_inclusive_sum_first =
        warp_local_degree_inclusive_sums + warp_id * raft::warp_size();
      auto this_warp_inclusive_sum_last =
        this_warp_inclusive_sum_first + (max_key_idx - min_key_idx);

      if (edge_partition_e_mask) {
        for (size_t i = lane_id; i < rounded_up_num_edges_this_warp; i += raft::warp_size()) {
          e_op_result_t e_op_result{thrust::nullopt};

          if (i < static_cast<size_t>(num_edges_this_warp)) {
            auto key_idx_this_warp = static_cast<vertex_t>(thrust::distance(
              this_warp_inclusive_sum_first,
              thrust::upper_bound(
                thrust::seq, this_warp_inclusive_sum_first, this_warp_inclusive_sum_last, i)));
            auto local_edge_offset =
              warp_key_local_edge_offsets[warp_id * raft::warp_size() + key_idx_this_warp] +
              static_cast<edge_t>(i - ((key_idx_this_warp == 0) ? edge_t{0}
                                                                : *(this_warp_inclusive_sum_first +
                                                                    (key_idx_this_warp - 1))));
            if ((*edge_partition_e_mask).get(local_edge_offset)) {
              auto key    = *(key_first + (min_key_idx + key_idx_this_warp));
              e_op_result = call_e_op(key, local_edge_offset);
            }
          }

          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      } else {
        for (size_t i = lane_id; i < rounded_up_num_edges_this_warp; i += raft::warp_size()) {
          e_op_result_t e_op_result{thrust::nullopt};

          if (i < static_cast<size_t>(num_edges_this_warp)) {
            auto key_idx_this_warp = static_cast<vertex_t>(thrust::distance(
              this_warp_inclusive_sum_first,
              thrust::upper_bound(
                thrust::seq, this_warp_inclusive_sum_first, this_warp_inclusive_sum_last, i)));
            auto local_edge_offset =
              warp_key_local_edge_offsets[warp_id * raft::warp_size() + key_idx_this_warp] +
              static_cast<edge_t>(i - ((key_idx_this_warp == 0) ? edge_t{0}
                                                                : *(this_warp_inclusive_sum_first +
                                                                    (key_idx_this_warp - 1))));
            auto key    = *(key_first + (min_key_idx + key_idx_this_warp));
            e_op_result = call_e_op(key, local_edge_offset);
          }

          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <bool max_one_e_per_frontier_key,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename BufferKeyOutputIterator,
          typename BufferValueOutputIterator,
          typename EdgeOp>
__global__ static void extract_transform_v_frontier_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferValueOutputIterator buffer_value_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using e_op_result_t =
    typename edge_op_result_type<key_t,
                                 typename GraphViewType::vertex_type,
                                 typename EdgePartitionSrcValueInputWrapper::value_type,
                                 typename EdgePartitionDstValueInputWrapper::value_type,
                                 typename EdgePartitionEdgeValueInputWrapper::value_type,
                                 EdgeOp>::type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(extract_transform_v_frontier_e_kernel_block_size % raft::warp_size() == 0);
  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  cuda::atomic_ref<size_t, cuda::thread_scope_device> buffer_idx(*buffer_idx_ptr);

  using WarpReduce = cub::WarpReduce<int32_t>;
  __shared__ std::conditional_t<max_one_e_per_frontier_key,
                                typename WarpReduce::TempStorage,
                                std::byte /* dummy */>
    temp_storage[max_one_e_per_frontier_key
                   ? (extract_transform_v_frontier_e_kernel_block_size / raft::warp_size())
                   : int32_t{1} /* dummy */];

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key          = *(key_first + idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t local_edge_offset{};
    edge_t local_out_degree{};
    thrust::tie(indices, local_edge_offset, local_out_degree) =
      edge_partition.local_edges(major_offset);
    auto rounded_up_local_out_degree =
      ((static_cast<size_t>(local_out_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
      raft::warp_size();

    auto call_e_op = call_e_op_t<GraphViewType,
                                 key_t,
                                 EdgePartitionSrcValueInputWrapper,
                                 EdgePartitionDstValueInputWrapper,
                                 EdgePartitionEdgeValueInputWrapper,
                                 EdgeOp>{edge_partition,
                                         edge_partition_src_value_input,
                                         edge_partition_dst_value_input,
                                         edge_partition_e_value_input,
                                         e_op,
                                         key,
                                         major_offset,
                                         indices,
                                         local_edge_offset};

    if (edge_partition_e_mask) {
      for (size_t i = lane_id; i < rounded_up_local_out_degree; i += raft::warp_size()) {
        e_op_result_t e_op_result{thrust::nullopt};
        if ((i < static_cast<size_t>(local_out_degree)) &&
            ((*edge_partition_e_mask).get(local_edge_offset + i))) {
          e_op_result = call_e_op(i);
        }

        if constexpr (max_one_e_per_frontier_key) {
          auto first_valid_lane_id =
            WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
              .Reduce(e_op_result ? lane_id : raft::warp_size(), cub::Min());
          first_valid_lane_id = __shfl_sync(raft::warp_full_mask(), first_valid_lane_id, int{0});
          if (lane_id == first_valid_lane_id) {
            auto push_idx = buffer_idx.fetch_add(1, cuda::std::memory_order_relaxed);
            push_buffer_element(
              buffer_key_output_first, buffer_value_output_first, push_idx, e_op_result);
          }
          if (first_valid_lane_id != raft::warp_size()) { break; }
        } else {
          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      }
    } else {
      for (size_t i = lane_id; i < rounded_up_local_out_degree; i += raft::warp_size()) {
        e_op_result_t e_op_result{thrust::nullopt};
        if (i < static_cast<size_t>(local_out_degree)) { e_op_result = call_e_op(i); }

        if constexpr (max_one_e_per_frontier_key) {
          auto first_valid_lane_id =
            WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
              .Reduce(e_op_result ? lane_id : raft::warp_size(), cub::Min());
          first_valid_lane_id = __shfl_sync(raft::warp_full_mask(), first_valid_lane_id, int{0});
          if (lane_id == first_valid_lane_id) {
            auto push_buffer_idx = buffer_idx.fetch_add(1, cuda::std::memory_order_relaxed);
            push_buffer_element(
              buffer_key_output_first, buffer_value_output_first, push_buffer_idx, e_op_result);
          }
          if (first_valid_lane_id != raft::warp_size()) { break; }
        } else {
          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool max_one_e_per_frontier_key,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename BufferKeyOutputIterator,
          typename BufferValueOutputIterator,
          typename EdgeOp>
__global__ static void extract_transform_v_frontier_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferValueOutputIterator buffer_value_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using e_op_result_t =
    typename edge_op_result_type<key_t,
                                 typename GraphViewType::vertex_type,
                                 typename EdgePartitionSrcValueInputWrapper::value_type,
                                 typename EdgePartitionDstValueInputWrapper::value_type,
                                 typename EdgePartitionEdgeValueInputWrapper::value_type,
                                 EdgeOp>::type;

  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = threadIdx.x % raft::warp_size();
  auto idx           = static_cast<size_t>(blockIdx.x);

  cuda::atomic_ref<size_t, cuda::thread_scope_device> buffer_idx(*buffer_idx_ptr);

  using BlockReduce = cub::BlockReduce<int32_t, extract_transform_v_frontier_e_kernel_block_size>;
  __shared__ std::conditional_t<max_one_e_per_frontier_key,
                                typename BlockReduce::TempStorage,
                                std::byte /* dummy */>
    temp_storage;
  __shared__ int32_t output_thread_id;

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key          = *(key_first + idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t local_edge_offset{};
    edge_t local_out_degree{};
    thrust::tie(indices, local_edge_offset, local_out_degree) =
      edge_partition.local_edges(major_offset);
    auto rounded_up_local_out_degree = ((static_cast<size_t>(local_out_degree) +
                                         (extract_transform_v_frontier_e_kernel_block_size - 1)) /
                                        extract_transform_v_frontier_e_kernel_block_size) *
                                       extract_transform_v_frontier_e_kernel_block_size;

    auto call_e_op = call_e_op_t<GraphViewType,
                                 key_t,
                                 EdgePartitionSrcValueInputWrapper,
                                 EdgePartitionDstValueInputWrapper,
                                 EdgePartitionEdgeValueInputWrapper,
                                 EdgeOp>{edge_partition,
                                         edge_partition_src_value_input,
                                         edge_partition_dst_value_input,
                                         edge_partition_e_value_input,
                                         e_op,
                                         key,
                                         major_offset,
                                         indices,
                                         local_edge_offset};

    if (edge_partition_e_mask) {
      for (size_t i = threadIdx.x; i < rounded_up_local_out_degree; i += blockDim.x) {
        e_op_result_t e_op_result{thrust::nullopt};
        if ((i < static_cast<size_t>(local_out_degree)) &&
            ((*edge_partition_e_mask).get(local_edge_offset + i))) {
          e_op_result = call_e_op(i);
        }

        if constexpr (max_one_e_per_frontier_key) {
          auto first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result ? threadIdx.x : extract_transform_v_frontier_e_kernel_block_size,
                      cub::Min());
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          if (threadIdx.x == output_thread_id) {
            auto push_buffer_idx = buffer_idx.fetch_add(1, cuda::std::memory_order_relaxed);
            push_buffer_element(
              buffer_key_output_first, buffer_value_output_first, push_buffer_idx, e_op_result);
          }
          if (output_thread_id != extract_transform_v_frontier_e_kernel_block_size) { break; }
        } else {
          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      }
    } else {
      for (size_t i = threadIdx.x; i < rounded_up_local_out_degree; i += blockDim.x) {
        e_op_result_t e_op_result{thrust::nullopt};
        if (i < static_cast<size_t>(local_out_degree)) { e_op_result = call_e_op(i); }

        if constexpr (max_one_e_per_frontier_key) {
          auto first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result ? threadIdx.x : extract_transform_v_frontier_e_kernel_block_size,
                      cub::Min());
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          if (threadIdx.x == output_thread_id) {
            auto push_buffer_idx = buffer_idx.fetch_add(1, cuda::std::memory_order_relaxed);
            push_buffer_element(
              buffer_key_output_first, buffer_value_output_first, push_buffer_idx, e_op_result);
          }
          if (output_thread_id != extract_transform_v_frontier_e_kernel_block_size) { break; }
        } else {
          warp_push_buffer_elements(
            buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
        }
      }
    }

    idx += gridDim.x;
  }
}

#define EXTRACT_PERFORMANCE_MEASUREMENT 1

template <bool incoming,  // iterate over incoming edges (incoming == true) or outgoing edges
                          // (incoming == false)
          bool max_one_e_per_frontier_key,  // extract maximum one edge per key in the input
                                            // frontier (if multiple e_op calls return valid output
                                            // values, all but one will be discarded (the remaining
                                            // one will be arbitrarily selected)
          typename OutputKeyT,
          typename OutputValueT,
          typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp>
std::tuple<optional_dataframe_buffer_type_t<OutputKeyT>,
           optional_dataframe_buffer_type_t<OutputValueT>>
extract_transform_v_frontier_e(raft::handle_t const& handle,
                               GraphViewType const& graph_view,
                               KeyBucketType const& frontier,
                               EdgeSrcValueInputWrapper edge_src_value_input,
                               EdgeDstValueInputWrapper edge_dst_value_input,
                               EdgeValueInputWrapper edge_value_input,
                               EdgeOp e_op,
                               bool do_expensive_check = false)
{
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
#endif
  using vertex_t       = typename GraphViewType::vertex_type;
  using edge_t         = typename GraphViewType::edge_type;
  using key_t          = typename KeyBucketType::key_type;
  using output_key_t   = OutputKeyT;
  using output_value_t = OutputValueT;

  using e_op_result_t = typename edge_op_result_type<key_t,
                                                     typename GraphViewType::vertex_type,
                                                     typename EdgeSrcValueInputWrapper::value_type,
                                                     typename EdgeDstValueInputWrapper::value_type,
                                                     typename EdgeValueInputWrapper::value_type,
                                                     EdgeOp>::type;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  static_assert(GraphViewType::is_storage_transposed == incoming);
  static_assert(!std::is_same_v<output_key_t, void> ||
                !std::is_same_v<output_value_t, void>);  // otherwise, this function becomes no-op
  static_assert(!std::is_same_v<e_op_result_t, void>);
  static_assert(
    std::is_same_v<e_op_result_t,
                   std::conditional_t<!std::is_same_v<output_key_t, void> &&
                                        !std::is_same_v<output_value_t, void>,
                                      thrust::optional<thrust::tuple<output_key_t, output_value_t>>,
                                      std::conditional_t<!std::is_same_v<output_key_t, void>,
                                                         thrust::optional<output_key_t>,
                                                         thrust::optional<output_value_t>>>>);

  constexpr bool use_bitmap = GraphViewType::is_multi_gpu && std::is_same_v<key_t, vertex_t> &&
                              KeyBucketType::is_sorted_unique;

  if (do_expensive_check) {
    auto frontier_vertex_first =
      thrust_tuple_get_or_identity<decltype(frontier.begin()), 0>(frontier.begin());
    auto frontier_vertex_last =
      thrust_tuple_get_or_identity<decltype(frontier.end()), 0>(frontier.end());
    auto num_invalid_keys =
      frontier.size() -
      thrust::count_if(handle.get_thrust_policy(),
                       frontier_vertex_first,
                       frontier_vertex_last,
                       check_in_range_t<vertex_t>{graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_keys = host_scalar_allreduce(
        handle.get_comms(), num_invalid_keys, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_keys == size_t{0},
                    "Invalid input argument: frontier includes out-of-range keys.");
  }

  // 1. pre-process frontier data

  auto frontier_key_first = frontier.begin();
  auto frontier_key_last  = frontier.end();
  auto frontier_keys      = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
  if constexpr (!KeyBucketType::is_sorted_unique) {
    resize_dataframe_buffer(frontier_keys, frontier.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 frontier_key_first,
                 frontier_key_last,
                 get_dataframe_buffer_begin(frontier_keys));
    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(frontier_keys),
                 get_dataframe_buffer_end(frontier_keys));
    frontier_key_first = get_dataframe_buffer_begin(frontier_keys);
    frontier_key_last  = get_dataframe_buffer_end(frontier_keys);
  }

  {  // drop zero degree vertices
    size_t partition_idx{0};
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      partition_idx    = static_cast<size_t>(minor_comm.get_rank());
    }
    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);

    if (segment_offsets) {
      frontier_key_last = compute_key_lower_bound(
        frontier_key_first,
        frontier_key_last,
        graph_view.local_vertex_partition_range_first() + *((*segment_offsets).rbegin() + 1),
        handle.get_stream());
    }
  }

  std::vector<size_t> local_frontier_sizes{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    local_frontier_sizes = host_scalar_allgather(
      minor_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
  }

  // update frontier bitmap (used to reduce broadcast bandwidth size)

  std::
    conditional_t<use_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      frontier_bitmap{};
  std::conditional_t<use_bitmap, std::vector<bool>, std::byte /* dummy */> use_bitmap_flags{};
  if constexpr (use_bitmap) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto segment_offsets =
      graph_view.local_edge_partition_segment_offsets(static_cast<size_t>(minor_comm_rank));
    size_t bool_size = segment_offsets ? *((*segment_offsets).rbegin() + 1)
                                       : graph_view.local_vertex_partition_range_size();

    std::tie(frontier_bitmap, use_bitmap_flags) =
      compute_vertex_list_bitmap_info(minor_comm,
                                      frontier_key_first,
                                      frontier_key_last,
                                      graph_view.local_vertex_partition_range_first(),
                                      graph_view.local_vertex_partition_range_first() + bool_size,
                                      handle.get_stream());
  }

  // 2. fill the buffers

  std::vector<optional_dataframe_buffer_type_t<output_key_t>> key_buffers{};
  std::vector<optional_dataframe_buffer_type_t<output_value_t>> value_buffers{};
  key_buffers.reserve(graph_view.number_of_local_edge_partitions());
  value_buffers.reserve(graph_view.number_of_local_edge_partitions());

  auto edge_mask_view = graph_view.edge_mask_view();

#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1 = std::chrono::steady_clock::now();
#endif
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto subtime0 = std::chrono::steady_clock::now();
#endif
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

    auto edge_partition_frontier_key_first = frontier_key_first;
    auto edge_partition_frontier_key_last  = frontier_key_last;
    auto edge_partition_frontier_key_buffer =
      allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      if (minor_comm_size > 1) {
        auto const minor_comm_rank = minor_comm.get_rank();

        resize_dataframe_buffer(
          edge_partition_frontier_key_buffer, local_frontier_sizes[i], handle.get_stream());

        if constexpr (use_bitmap) {
          std::variant<raft::device_span<uint32_t const>, decltype(frontier_key_first)> v_list{};
          if (use_bitmap_flags[i]) {
            v_list = raft::device_span<uint32_t const>((*frontier_bitmap).data(),
                                                       (*frontier_bitmap).size());
          } else {
            v_list = frontier_key_first;
          }
          auto bool_size = segment_offsets ? *((*segment_offsets).rbegin() + 1)
                                           : edge_partition.major_range_size();
          device_bcast_vertex_list(
            minor_comm,
            v_list,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer),
            edge_partition.major_range_first(),
            edge_partition.major_range_first() + bool_size,
            static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
            static_cast<int>(i),
            handle.get_stream());
        } else {
          device_bcast(minor_comm,
                       frontier_key_first,
                       get_dataframe_buffer_begin(edge_partition_frontier_key_buffer),
                       local_frontier_sizes[i],
                       static_cast<int>(i),
                       handle.get_stream());
        }

        edge_partition_frontier_key_first =
          get_dataframe_buffer_begin(edge_partition_frontier_key_buffer);
        edge_partition_frontier_key_last =
          get_dataframe_buffer_end(edge_partition_frontier_key_buffer);
      }
    }

    auto edge_partition_frontier_major_first =
      thrust_tuple_get_or_identity<decltype(edge_partition_frontier_key_first), 0>(
        edge_partition_frontier_key_first);
    auto edge_partition_frontier_major_last =
      thrust_tuple_get_or_identity<decltype(edge_partition_frontier_key_last), 0>(
        edge_partition_frontier_key_last);

    auto max_pushes = max_one_e_per_frontier_key ? local_frontier_sizes[i]
                                                 : edge_partition.compute_number_of_edges(
                                                     edge_partition_frontier_major_first,
                                                     edge_partition_frontier_major_last,
                                                     handle.get_stream());

    auto tmp_key_buffer =
      allocate_optional_dataframe_buffer<output_key_t>(max_pushes, handle.get_stream());
    auto tmp_value_buffer =
      allocate_optional_dataframe_buffer<output_value_t>(max_pushes, handle.get_stream());
    rmm::device_scalar<size_t> tmp_buffer_idx(size_t{0}, handle.get_stream());

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input =
        edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
    } else {
      edge_partition_src_value_input =
        edge_partition_src_input_device_view_t(edge_src_value_input, i);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);

#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto subtime1 = std::chrono::steady_clock::now();
#endif
    if (segment_offsets) {
      auto h_offsets = compute_key_segment_offsets(
        edge_partition_frontier_major_first,
        edge_partition_frontier_major_last,
        raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
        edge_partition.major_range_first(),
        graph_view.use_dcs(),
        handle.get_stream());

      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one
      // more segment for very high degree vertices and running segmented reduction
      if (h_offsets[0] > 0) {
        raft::grid_1d_block_t update_grid(h_offsets[0],
                                          extract_transform_v_frontier_e_kernel_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        extract_transform_v_frontier_e_high_degree<max_one_e_per_frontier_key, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_frontier_key_first + h_offsets[0],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
            get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
            tmp_buffer_idx.data(),
            e_op);
      }
      if (h_offsets[1] - h_offsets[0] > 0) {
        raft::grid_1d_warp_t update_grid(h_offsets[1] - h_offsets[0],
                                         extract_transform_v_frontier_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        extract_transform_v_frontier_e_mid_degree<max_one_e_per_frontier_key, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first + h_offsets[0],
            edge_partition_frontier_key_first + h_offsets[1],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
            get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
            tmp_buffer_idx.data(),
            e_op);
      }
      if (h_offsets[2] - h_offsets[1] > 0) {
        raft::grid_1d_thread_t update_grid(h_offsets[2] - h_offsets[1],
                                           extract_transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        extract_transform_v_frontier_e_hypersparse_or_low_degree<false,
                                                                 max_one_e_per_frontier_key,
                                                                 GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first + h_offsets[1],
            edge_partition_frontier_key_first + h_offsets[2],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
            get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
            tmp_buffer_idx.data(),
            e_op);
      }
      if (edge_partition.dcs_nzd_vertex_count() && (h_offsets[3] - h_offsets[2] > 0)) {
        raft::grid_1d_thread_t update_grid(h_offsets[3] - h_offsets[2],
                                           extract_transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        extract_transform_v_frontier_e_hypersparse_or_low_degree<true,
                                                                 max_one_e_per_frontier_key,
                                                                 GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first + h_offsets[2],
            edge_partition_frontier_key_first + h_offsets[3],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
            get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
            tmp_buffer_idx.data(),
            e_op);
      }
    } else {
      if (local_frontier_sizes[i] > 0) {
        raft::grid_1d_thread_t update_grid(local_frontier_sizes[i],
                                           extract_transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        extract_transform_v_frontier_e_hypersparse_or_low_degree<false,
                                                                 max_one_e_per_frontier_key,
                                                                 GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_frontier_key_last,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
            get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
            tmp_buffer_idx.data(),
            e_op);
      }
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto subtime2 = std::chrono::steady_clock::now();
#endif

    auto tmp_buffer_size = tmp_buffer_idx.value(handle.get_stream());

    resize_optional_dataframe_buffer<output_key_t>(
      tmp_key_buffer, tmp_buffer_size, handle.get_stream());
    shrink_to_fit_optional_dataframe_buffer<output_key_t>(tmp_key_buffer, handle.get_stream());

    resize_optional_dataframe_buffer<output_value_t>(
      tmp_value_buffer, tmp_buffer_size, handle.get_stream());
    shrink_to_fit_optional_dataframe_buffer<output_value_t>(tmp_value_buffer, handle.get_stream());

    key_buffers.push_back(std::move(tmp_key_buffer));
    value_buffers.push_back(std::move(tmp_value_buffer));

#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    auto subtime3                         = std::chrono::steady_clock::now();
    std::chrono::duration<double> subdur0 = subtime1 - subtime0;
    std::chrono::duration<double> subdur1 = subtime2 - subtime1;
    std::chrono::duration<double> subdur2 = subtime3 - subtime2;
    std::cout << "\t\t\tdetail::extract i=" << i << " took (" << subdur0.count() << ","
              << subdur1.count() << "," << subdur2.count() << ")" << std::endl;
#endif
  }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();
#endif

  // 3. concatenate and return the buffers

  auto key_buffer   = allocate_optional_dataframe_buffer<output_key_t>(0, handle.get_stream());
  auto value_buffer = allocate_optional_dataframe_buffer<output_value_t>(0, handle.get_stream());
  if (key_buffers.size() == 0) {
    key_buffer   = std::move(key_buffers[0]);
    value_buffer = std::move(value_buffers[0]);
  } else {
    std::vector<size_t> buffer_sizes(key_buffers.size());
    static_assert(!std::is_same_v<output_key_t, void> || !std::is_same_v<output_value_t, void>);
    for (size_t i = 0; i < key_buffers.size(); ++i) {
      if constexpr (!std::is_same_v<key_t, void>) {
        buffer_sizes[i] = size_optional_dataframe_buffer<output_key_t>(key_buffers[i]);
      } else {
        buffer_sizes[i] = size_optional_dataframe_buffer<output_value_t>(value_buffers[i]);
      }
    }
    auto buffer_size = std::reduce(buffer_sizes.begin(), buffer_sizes.end());
    resize_optional_dataframe_buffer<output_key_t>(key_buffer, buffer_size, handle.get_stream());
    resize_optional_dataframe_buffer<output_value_t>(
      value_buffer, buffer_size, handle.get_stream());
    std::vector<size_t> buffer_displacements(buffer_sizes.size());
    std::exclusive_scan(
      buffer_sizes.begin(), buffer_sizes.end(), buffer_displacements.begin(), size_t{0});
    for (size_t i = 0; i < key_buffers.size(); ++i) {
      if constexpr (!std::is_same_v<output_key_t, void>) {
        thrust::copy(
          handle.get_thrust_policy(),
          get_optional_dataframe_buffer_cbegin<output_key_t>(key_buffers[i]),
          get_optional_dataframe_buffer_cend<output_key_t>(key_buffers[i]),
          get_optional_dataframe_buffer_begin<output_key_t>(key_buffer) + buffer_displacements[i]);
      }

      if constexpr (!std::is_same_v<output_value_t, void>) {
        thrust::copy(handle.get_thrust_policy(),
                     get_optional_dataframe_buffer_cbegin<output_value_t>(value_buffers[i]),
                     get_optional_dataframe_buffer_cend<output_value_t>(value_buffers[i]),
                     get_optional_dataframe_buffer_begin<output_value_t>(value_buffer) +
                       buffer_displacements[i]);
      }
    }
  }

#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time3                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::chrono::duration<double> dur1 = time2 - time1;
  std::chrono::duration<double> dur2 = time3 - time2;
  std::cout << "\t\tdetail::extract took (" << dur0.count() << "," << dur1.count() << ","
            << dur2.count() << ")" << std::endl;
#endif

  return std::make_tuple(std::move(key_buffer), std::move(value_buffer));
}

}  // namespace detail

}  // namespace cugraph
