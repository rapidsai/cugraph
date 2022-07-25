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

#include <prims/property_op_utils.cuh>
#include <prims/reduce_op.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr update_v_frontier_from_outgoing_e_kernel_block_size = 512;

// we cannot use thrust::iterator_traits<Iterator>::value_type if Iterator is void* (reference to
// void is not allowed)
template <typename PayloadIterator, typename Enable = void>
struct optional_payload_buffer_value_type_t;

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<!std::is_same_v<PayloadIterator, void*>>> {
  using value = typename thrust::iterator_traits<PayloadIterator>::value_type;
};

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<std::is_same_v<PayloadIterator, void*>>> {
  using value = void;
};

template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
std::byte allocate_optional_payload_buffer(size_t size, rmm::cuda_stream_view stream)
{
  return std::byte{0};  // dummy
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
auto allocate_optional_payload_buffer(size_t size, rmm::cuda_stream_view stream)
{
  return allocate_dataframe_buffer<payload_t>(size, stream);
}

template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
void* get_optional_payload_buffer_begin(std::byte& optional_payload_buffer)
{
  return static_cast<void*>(nullptr);
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
auto get_optional_payload_buffer_begin(
  std::add_lvalue_reference_t<decltype(allocate_dataframe_buffer<payload_t>(
    size_t{0}, rmm::cuda_stream_view{}))> optional_payload_buffer)
{
  return get_dataframe_buffer_begin(optional_payload_buffer);
}

template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
void resize_optional_payload_buffer(std::byte& optional_payload_buffer,
                                    size_t new_buffer_size,
                                    rmm::cuda_stream_view stream_view)
{
  return;
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
void resize_optional_payload_buffer(
  std::add_lvalue_reference_t<decltype(allocate_dataframe_buffer<payload_t>(
    size_t{0}, rmm::cuda_stream_view{}))> optional_payload_buffer,
  size_t new_buffer_size,
  rmm::cuda_stream_view stream_view)
{
  return resize_dataframe_buffer(optional_payload_buffer, new_buffer_size, stream_view);
}

template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
void shrink_to_fit_optional_payload_buffer(std::byte& optional_payload_buffer,
                                           rmm::cuda_stream_view stream_view)
{
  return;
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
void shrink_to_fit_optional_payload_buffer(
  std::add_lvalue_reference_t<decltype(allocate_dataframe_buffer<payload_t>(
    size_t{0}, rmm::cuda_stream_view{}))> optional_payload_buffer,
  rmm::cuda_stream_view stream_view)
{
  return shrink_to_fit_dataframe_buffer(optional_payload_buffer, stream_view);
}

template <typename vertex_t,
          typename e_op_result_t,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator>
__device__ void push_buffer_element(vertex_t dst,
                                    e_op_result_t e_op_result,
                                    BufferKeyOutputIterator buffer_key_output_first,
                                    BufferPayloadOutputIterator buffer_payload_output_first,
                                    size_t buffer_idx)
{
  using key_t = typename thrust::iterator_traits<BufferKeyOutputIterator>::value_type;
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  assert(e_op_result.has_value());

  if constexpr (std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
    *(buffer_key_output_first + buffer_idx) = dst;
  } else if constexpr (std::is_same_v<key_t, vertex_t> && !std::is_same_v<payload_t, void>) {
    *(buffer_key_output_first + buffer_idx)     = dst;
    *(buffer_payload_output_first + buffer_idx) = *e_op_result;
  } else if constexpr (!std::is_same_v<key_t, vertex_t> && std::is_same_v<payload_t, void>) {
    *(buffer_key_output_first + buffer_idx) = thrust::make_tuple(dst, *e_op_result);
  } else {
    *(buffer_key_output_first + buffer_idx) = thrust::make_tuple(dst, thrust::get<0>(*e_op_result));
    *(buffer_payload_output_first + buffer_idx) = thrust::get<1>(*e_op_result);
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void update_v_frontier_from_outgoing_e_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename thrust::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;
  using e_op_result_t = typename evaluate_edge_op<GraphViewType,
                                                  key_t,
                                                  EdgePartitionSrcValueInputWrapper,
                                                  EdgePartitionDstValueInputWrapper,
                                                  EdgeOp>::result_type;

  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const tid        = threadIdx.x + blockIdx.x * blockDim.x;
  auto const warp_id    = threadIdx.x / raft::warp_size();
  auto const lane_id    = tid % raft::warp_size();
  auto src_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                              edge_partition.major_range_first());
  auto idx              = static_cast<size_t>(tid);

  __shared__ edge_t
    warp_local_degree_inclusive_sums[update_v_frontier_from_outgoing_e_kernel_block_size];
  __shared__ edge_t
    warp_key_local_edge_offsets[update_v_frontier_from_outgoing_e_kernel_block_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  __shared__ size_t buffer_warp_start_indices[update_v_frontier_from_outgoing_e_kernel_block_size /
                                              raft::warp_size()];

  auto indices = edge_partition.indices();
  auto weights = edge_partition.weights();

  vertex_t num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
  auto rounded_up_num_keys =
    ((static_cast<size_t>(num_keys) + (raft::warp_size() - 1)) / raft::warp_size()) *
    raft::warp_size();
  while (idx < rounded_up_num_keys) {
    auto min_key_idx = static_cast<vertex_t>(idx - (idx % raft::warp_size()));  // inclusive
    auto max_key_idx =
      static_cast<vertex_t>(std::min(static_cast<size_t>(min_key_idx) + raft::warp_size(),
                                     static_cast<size_t>(num_keys)));  // exclusive

    // update warp_local_degree_inclusive_sums & warp_key_local_edge_offsets

    edge_t local_degree{0};
    if (lane_id < static_cast<int32_t>(max_key_idx - min_key_idx)) {
      auto key = *(key_first + idx);
      vertex_t src{};
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        src = key;
      } else {
        src = thrust::get<0>(key);
      }
      auto src_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(src);
      if (src_hypersparse_idx) {
        auto src_idx                             = src_start_offset + *src_hypersparse_idx;
        local_degree                             = edge_partition.local_degree(src_idx);
        warp_key_local_edge_offsets[threadIdx.x] = edge_partition.local_offset(src_idx);
      } else {
        local_degree                             = edge_t{0};
        warp_key_local_edge_offsets[threadIdx.x] = edge_t{0};  // dummy
      }
    }
    WarpScan(temp_storage)
      .InclusiveSum(local_degree, warp_local_degree_inclusive_sums[threadIdx.x]);
    __syncwarp();

    // process local edges for the keys in [key_first + min_key_idx, key_first + max_key_idx)

    auto num_edges_this_warp = warp_local_degree_inclusive_sums[warp_id * raft::warp_size() +
                                                                (max_key_idx - min_key_idx) - 1];
    auto rounded_up_num_edges_this_warp =
      ((static_cast<size_t>(num_edges_this_warp) + (raft::warp_size() - 1)) / raft::warp_size()) *
      raft::warp_size();

    for (size_t i = lane_id; i < rounded_up_num_edges_this_warp; i += raft::warp_size()) {
      e_op_result_t e_op_result{};
      vertex_t dst{};

      if (i < static_cast<size_t>(num_edges_this_warp)) {
        auto key_idx_this_warp = static_cast<vertex_t>(thrust::distance(
          warp_local_degree_inclusive_sums + warp_id * raft::warp_size(),
          thrust::upper_bound(thrust::seq,
                              warp_local_degree_inclusive_sums + warp_id * raft::warp_size(),
                              warp_local_degree_inclusive_sums + warp_id * raft::warp_size() +
                                (max_key_idx - min_key_idx),
                              i)));
        auto local_edge_offset =
          warp_key_local_edge_offsets[warp_id * raft::warp_size() + key_idx_this_warp] +
          static_cast<edge_t>(i -
                              ((key_idx_this_warp == 0)
                                 ? edge_t{0}
                                 : warp_local_degree_inclusive_sums[warp_id * raft::warp_size() +
                                                                    key_idx_this_warp - 1]));
        auto key = *(key_first + (min_key_idx + key_idx_this_warp));
        vertex_t src{};
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          src = key;
        } else {
          src = thrust::get<0>(key);
        }
        dst             = indices[local_edge_offset];
        auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
        auto dst_offset = edge_partition.minor_offset_from_minor_nocheck(dst);
        e_op_result     = evaluate_edge_op<GraphViewType,
                                       key_t,
                                       EdgePartitionSrcValueInputWrapper,
                                       EdgePartitionDstValueInputWrapper,
                                       EdgeOp>()
                        .compute(key,
                                 dst,
                                 weights ? (*weights)[local_edge_offset] : weight_t{1.0},
                                 edge_partition_src_value_input.get(src_offset),
                                 edge_partition_dst_value_input.get(dst_offset),
                                 e_op);
      }
      auto ballot_e_op =
        __ballot_sync(uint32_t{0xffffffff}, e_op_result ? uint32_t{1} : uint32_t{0});
      if (ballot_e_op) {
        if (lane_id == 0) {
          auto increment = __popc(ballot_e_op);
          static_assert(sizeof(unsigned long long int) == sizeof(size_t));
          buffer_warp_start_indices[warp_id] =
            static_cast<size_t>(atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                          static_cast<unsigned long long int>(increment)));
        }
        __syncwarp();
        if (e_op_result) {
          auto buffer_warp_offset =
            static_cast<edge_t>(__popc(ballot_e_op & ~(uint32_t{0xffffffff} << lane_id)));
          push_buffer_element(dst,
                              e_op_result,
                              buffer_key_output_first,
                              buffer_payload_output_first,
                              buffer_warp_start_indices[warp_id] + buffer_warp_offset);
        }
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void update_v_frontier_from_outgoing_e_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename thrust::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;
  using e_op_result_t = typename evaluate_edge_op<GraphViewType,
                                                  key_t,
                                                  EdgePartitionSrcValueInputWrapper,
                                                  EdgePartitionDstValueInputWrapper,
                                                  EdgeOp>::result_type;

  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const tid     = threadIdx.x + blockIdx.x * blockDim.x;
  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid);

  __shared__ edge_t
    warp_local_degree_inclusive_sums[update_v_frontier_from_outgoing_e_kernel_block_size];
  __shared__ edge_t
    warp_key_local_edge_offsets[update_v_frontier_from_outgoing_e_kernel_block_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  __shared__ size_t buffer_warp_start_indices[update_v_frontier_from_outgoing_e_kernel_block_size /
                                              raft::warp_size()];

  auto indices = edge_partition.indices();
  auto weights = edge_partition.weights();

  vertex_t num_keys = static_cast<size_t>(thrust::distance(key_first, key_last));
  auto rounded_up_num_keys =
    ((static_cast<size_t>(num_keys) + (raft::warp_size() - 1)) / raft::warp_size()) *
    raft::warp_size();
  while (idx < rounded_up_num_keys) {
    auto min_key_idx = static_cast<vertex_t>(idx - (idx % raft::warp_size()));  // inclusive
    auto max_key_idx =
      static_cast<vertex_t>(std::min(static_cast<size_t>(min_key_idx) + raft::warp_size(),
                                     static_cast<size_t>(num_keys)));  // exclusive

    // update warp_local_degree_inclusive_sums & warp_key_local_edge_offsets

    edge_t local_degree{0};
    if (lane_id < static_cast<int32_t>(max_key_idx - min_key_idx)) {
      auto key = *(key_first + idx);
      vertex_t src{};
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        src = key;
      } else {
        src = thrust::get<0>(key);
      }
      auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
      local_degree    = edge_partition.local_degree(src_offset);
      warp_key_local_edge_offsets[threadIdx.x] = edge_partition.local_offset(src_offset);
    }
    WarpScan(temp_storage)
      .InclusiveSum(local_degree, warp_local_degree_inclusive_sums[threadIdx.x]);
    __syncwarp();

    // processes local edges for the keys in [key_first + min_key_idx, key_first + max_key_idx)

    auto num_edges_this_warp = warp_local_degree_inclusive_sums[warp_id * raft::warp_size() +
                                                                (max_key_idx - min_key_idx) - 1];
    auto rounded_up_num_edges_this_warp =
      ((static_cast<size_t>(num_edges_this_warp) + (raft::warp_size() - 1)) / raft::warp_size()) *
      raft::warp_size();
    for (size_t i = lane_id; i < rounded_up_num_edges_this_warp; i += raft::warp_size()) {
      e_op_result_t e_op_result{};
      vertex_t dst{};

      if (i < static_cast<size_t>(num_edges_this_warp)) {
        auto key_idx_this_warp = static_cast<vertex_t>(thrust::distance(
          warp_local_degree_inclusive_sums + warp_id * raft::warp_size(),
          thrust::upper_bound(thrust::seq,
                              warp_local_degree_inclusive_sums + warp_id * raft::warp_size(),
                              warp_local_degree_inclusive_sums + warp_id * raft::warp_size() +
                                (max_key_idx - min_key_idx),
                              i)));
        auto local_edge_offset =
          warp_key_local_edge_offsets[warp_id * raft::warp_size() + key_idx_this_warp] +
          static_cast<edge_t>(i -
                              ((key_idx_this_warp == 0)
                                 ? edge_t{0}
                                 : warp_local_degree_inclusive_sums[warp_id * raft::warp_size() +
                                                                    key_idx_this_warp - 1]));
        auto key = *(key_first + (min_key_idx + key_idx_this_warp));
        vertex_t src{};
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          src = key;
        } else {
          src = thrust::get<0>(key);
        }
        dst             = indices[local_edge_offset];
        auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
        auto dst_offset = edge_partition.minor_offset_from_minor_nocheck(dst);
        e_op_result     = evaluate_edge_op<GraphViewType,
                                       key_t,
                                       EdgePartitionSrcValueInputWrapper,
                                       EdgePartitionDstValueInputWrapper,
                                       EdgeOp>()
                        .compute(key,
                                 dst,
                                 weights ? (*weights)[local_edge_offset] : weight_t{1.0},
                                 edge_partition_src_value_input.get(src_offset),
                                 edge_partition_dst_value_input.get(dst_offset),
                                 e_op);
      }
      auto ballot = __ballot_sync(uint32_t{0xffffffff}, e_op_result ? uint32_t{1} : uint32_t{0});
      if (ballot > 0) {
        if (lane_id == 0) {
          auto increment = __popc(ballot);
          static_assert(sizeof(unsigned long long int) == sizeof(size_t));
          buffer_warp_start_indices[warp_id] =
            static_cast<size_t>(atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                          static_cast<unsigned long long int>(increment)));
        }
        __syncwarp();
        if (e_op_result) {
          auto buffer_warp_offset =
            static_cast<edge_t>(__popc(ballot & ~(uint32_t{0xffffffff} << lane_id)));
          push_buffer_element(dst,
                              e_op_result,
                              buffer_key_output_first,
                              buffer_payload_output_first,
                              buffer_warp_start_indices[warp_id] + buffer_warp_offset);
        }
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void update_v_frontier_from_outgoing_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename thrust::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;
  using e_op_result_t = typename evaluate_edge_op<GraphViewType,
                                                  key_t,
                                                  EdgePartitionSrcValueInputWrapper,
                                                  EdgePartitionDstValueInputWrapper,
                                                  EdgeOp>::result_type;

  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(update_v_frontier_from_outgoing_e_kernel_block_size % raft::warp_size() == 0);
  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  __shared__ size_t buffer_warp_start_indices[update_v_frontier_from_outgoing_e_kernel_block_size /
                                              raft::warp_size()];
  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t src{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      src = key;
    } else {
      src = thrust::get<0>(key);
    }
    auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = edge_partition.local_edges(src_offset);
    auto rounded_up_local_out_degree =
      ((static_cast<size_t>(local_out_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
      raft::warp_size();
    for (size_t i = lane_id; i < rounded_up_local_out_degree; i += raft::warp_size()) {
      e_op_result_t e_op_result{};
      vertex_t dst{};

      if (i < static_cast<size_t>(local_out_degree)) {
        dst             = indices[i];
        auto dst_offset = edge_partition.minor_offset_from_minor_nocheck(dst);
        e_op_result     = evaluate_edge_op<GraphViewType,
                                       key_t,
                                       EdgePartitionSrcValueInputWrapper,
                                       EdgePartitionDstValueInputWrapper,
                                       EdgeOp>()
                        .compute(key,
                                 dst,
                                 weights ? (*weights)[i] : weight_t{1.0},
                                 edge_partition_src_value_input.get(src_offset),
                                 edge_partition_dst_value_input.get(dst_offset),
                                 e_op);
      }
      auto ballot = __ballot_sync(uint32_t{0xffffffff}, e_op_result ? uint32_t{1} : uint32_t{0});
      if (ballot > 0) {
        if (lane_id == 0) {
          auto increment = __popc(ballot);
          static_assert(sizeof(unsigned long long int) == sizeof(size_t));
          buffer_warp_start_indices[warp_id] =
            static_cast<size_t>(atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                          static_cast<unsigned long long int>(increment)));
        }
        __syncwarp();
        if (e_op_result) {
          auto buffer_warp_offset =
            static_cast<edge_t>(__popc(ballot & ~(uint32_t{0xffffffff} << lane_id)));
          push_buffer_element(dst,
                              e_op_result,
                              buffer_key_output_first,
                              buffer_payload_output_first,
                              buffer_warp_start_indices[warp_id] + buffer_warp_offset);
        }
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void update_v_frontier_from_outgoing_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename thrust::iterator_traits<BufferKeyOutputIterator>::value_type>);
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;
  using e_op_result_t = typename evaluate_edge_op<GraphViewType,
                                                  key_t,
                                                  EdgePartitionSrcValueInputWrapper,
                                                  EdgePartitionDstValueInputWrapper,
                                                  EdgeOp>::result_type;

  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockScan = cub::BlockScan<edge_t, update_v_frontier_from_outgoing_e_kernel_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ size_t buffer_block_start_idx;

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
    auto key = *(key_first + idx);
    vertex_t src{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      src = key;
    } else {
      src = thrust::get<0>(key);
    }
    auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = edge_partition.local_edges(src_offset);
    auto rounded_up_local_out_degree =
      ((static_cast<size_t>(local_out_degree) +
        (update_v_frontier_from_outgoing_e_kernel_block_size - 1)) /
       update_v_frontier_from_outgoing_e_kernel_block_size) *
      update_v_frontier_from_outgoing_e_kernel_block_size;
    for (size_t i = threadIdx.x; i < rounded_up_local_out_degree; i += blockDim.x) {
      e_op_result_t e_op_result{};
      vertex_t dst{};
      edge_t buffer_block_offset{0};

      if (i < static_cast<size_t>(local_out_degree)) {
        dst             = indices[i];
        auto dst_offset = edge_partition.minor_offset_from_minor_nocheck(dst);
        e_op_result     = evaluate_edge_op<GraphViewType,
                                       key_t,
                                       EdgePartitionSrcValueInputWrapper,
                                       EdgePartitionDstValueInputWrapper,
                                       EdgeOp>()
                        .compute(key,
                                 dst,
                                 weights ? (*weights)[i] : weight_t{1.0},
                                 edge_partition_src_value_input.get(src_offset),
                                 edge_partition_dst_value_input.get(dst_offset),
                                 e_op);
      }
      BlockScan(temp_storage)
        .ExclusiveSum(e_op_result ? edge_t{1} : edge_t{0}, buffer_block_offset);
      if (threadIdx.x == (blockDim.x - 1)) {
        auto increment = buffer_block_offset + (e_op_result ? edge_t{1} : edge_t{0});
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        buffer_block_start_idx = increment > 0
                                   ? static_cast<size_t>(atomicAdd(
                                       reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                       static_cast<unsigned long long int>(increment)))
                                   : size_t{0} /* dummy */;
      }
      __syncthreads();
      if (e_op_result) {
        push_buffer_element(dst,
                            e_op_result,
                            buffer_key_output_first,
                            buffer_payload_output_first,
                            buffer_block_start_idx + buffer_block_offset);
      }
    }

    idx += gridDim.x;
  }
}

template <typename key_t, typename payload_t, typename ReduceOp>
auto sort_and_reduce_buffer_elements(
  raft::handle_t const& handle,
  decltype(allocate_dataframe_buffer<key_t>(0, rmm::cuda_stream_view{}))&& key_buffer,
  decltype(allocate_optional_payload_buffer<payload_t>(0,
                                                       rmm::cuda_stream_view{}))&& payload_buffer,
  ReduceOp reduce_op)
{
  if constexpr (std::is_same_v<payload_t, void>) {
    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(key_buffer),
                 get_dataframe_buffer_end(key_buffer));
  } else {
    thrust::sort_by_key(handle.get_thrust_policy(),
                        get_dataframe_buffer_begin(key_buffer),
                        get_dataframe_buffer_end(key_buffer),
                        get_optional_payload_buffer_begin<payload_t>(payload_buffer));
  }

  if constexpr (std::is_same_v<payload_t, void>) {
    auto it = thrust::unique(handle.get_thrust_policy(),
                             get_dataframe_buffer_begin(key_buffer),
                             get_dataframe_buffer_end(key_buffer));
    resize_dataframe_buffer(
      key_buffer,
      static_cast<size_t>(thrust::distance(get_dataframe_buffer_begin(key_buffer), it)),
      handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
  } else if constexpr (std::is_same_v<ReduceOp, reduce_op::any<typename ReduceOp::value_type>>) {
    auto it = thrust::unique_by_key(handle.get_thrust_policy(),
                                    get_dataframe_buffer_begin(key_buffer),
                                    get_dataframe_buffer_end(key_buffer),
                                    get_optional_payload_buffer_begin<payload_t>(payload_buffer));
    resize_dataframe_buffer(key_buffer,
                            static_cast<size_t>(thrust::distance(
                              get_dataframe_buffer_begin(key_buffer), thrust::get<0>(it))),
                            handle.get_stream());
    resize_dataframe_buffer(payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
    shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
  } else {
    auto num_uniques =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator(size_t{0}),
                       thrust::make_counting_iterator(size_dataframe_buffer(key_buffer)),
                       is_first_in_run_t<decltype(get_dataframe_buffer_begin(key_buffer))>{
                         get_dataframe_buffer_begin(key_buffer)});

    auto new_key_buffer = allocate_dataframe_buffer<key_t>(num_uniques, handle.get_stream());
    auto new_payload_buffer =
      allocate_dataframe_buffer<payload_t>(num_uniques, handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          get_dataframe_buffer_begin(key_buffer),
                          get_dataframe_buffer_end(key_buffer),
                          get_optional_payload_buffer_begin<payload_t>(payload_buffer),
                          get_dataframe_buffer_begin(new_key_buffer),
                          get_dataframe_buffer_begin(new_payload_buffer),
                          thrust::equal_to<key_t>(),
                          reduce_op);

    key_buffer     = std::move(new_key_buffer);
    payload_buffer = std::move(new_payload_buffer);
  }

  return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
}

}  // namespace detail

template <typename GraphViewType, typename VertexFrontierType>
typename GraphViewType::edge_type compute_num_out_nbrs_from_frontier(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType const& frontier,
  size_t cur_frontier_bucket_idx)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;
  using key_t    = typename VertexFrontierType::key_type;

  edge_t ret{0};

  auto const& cur_frontier_bucket = frontier.bucket(cur_frontier_bucket_idx);
  vertex_t const* local_frontier_vertex_first{nullptr};
  if constexpr (std::is_same_v<key_t, vertex_t>) {
    local_frontier_vertex_first = cur_frontier_bucket.begin();
  } else {
    local_frontier_vertex_first = thrust::get<0>(cur_frontier_bucket.begin().get_iterator_tuple());
  }

  std::vector<size_t> local_frontier_sizes{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    local_frontier_sizes =
      host_scalar_allgather(col_comm, cur_frontier_bucket.size(), handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(cur_frontier_bucket.size())};
  }
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    if constexpr (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      rmm::device_uvector<vertex_t> frontier_vertices(local_frontier_sizes[i], handle.get_stream());
      device_bcast(col_comm,
                   local_frontier_vertex_first,
                   frontier_vertices.data(),
                   local_frontier_sizes[i],
                   static_cast<int>(i),
                   handle.get_stream());

      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
      ret += graph_view.use_dcs()
               ? thrust::transform_reduce(
                   handle.get_thrust_policy(),
                   frontier_vertices.begin(),
                   frontier_vertices.end(),
                   [edge_partition,
                    major_hypersparse_first =
                      *(edge_partition.major_hypersparse_first())] __device__(auto major) {
                     if (major < major_hypersparse_first) {
                       auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
                       return edge_partition.local_degree(major_offset);
                     } else {
                       auto major_hypersparse_idx =
                         edge_partition.major_hypersparse_idx_from_major_nocheck(major);
                       return major_hypersparse_idx
                                ? edge_partition.local_degree(
                                    edge_partition.major_offset_from_major_nocheck(
                                      major_hypersparse_first) +
                                    *major_hypersparse_idx)
                                : edge_t{0};
                     }
                   },
                   edge_t{0},
                   thrust::plus<edge_t>())
               : thrust::transform_reduce(
                   handle.get_thrust_policy(),
                   frontier_vertices.begin(),
                   frontier_vertices.end(),
                   [edge_partition] __device__(auto major) {
                     auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
                     return edge_partition.local_degree(major_offset);
                   },
                   edge_t{0},
                   thrust::plus<edge_t>());
    } else {
      assert(i == 0);
      ret += thrust::transform_reduce(
        handle.get_thrust_policy(),
        local_frontier_vertex_first,
        local_frontier_vertex_first + cur_frontier_bucket.size(),
        [edge_partition] __device__(auto major) {
          auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
          return edge_partition.local_degree(major_offset);
        },
        edge_t{0},
        thrust::plus<edge_t>());
    }
  }

  return ret;
}

/**
 * @brief Iterate over outgoing edges from the current vertex frontier and reduce valid edge functor
 * outputs by (tagged-)destination ID.
 *
 * Edge functor outputs are thrust::optional objects and invalid if thrust::nullopt. Vertices are
 * assumed to be tagged if VertexFrontierType::key_type is a tuple of a vertex type and a tag type
 * (VertexFrontierType::key_type is identical to a vertex type otherwise).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierType Type of the vertex frontier class which abstracts vertex frontier
 * managements.
 * @tparam EdgePartitionSrcValueInputWrapper Type of the wrapper for edge partition source property
 * values.
 * @tparam EdgePartitionDstValueInputWrapper Type of the wrapper for edge partition destination
 * property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierType class object for vertex frontier managements. This object
 * includes multiple bucket objects.
 * @param cur_frontier_bucket_idx Index of the vertex frontier bucket holding vertices for the
 * current iteration.
 * @param edge_partition_src_value_input Device-copyable wrapper used to access source input
 * property values (for the edge sources assigned to this process in multi-GPU). Use either
 * cugraph::edge_partition_src_property_t::device_view() (if @p e_op needs to access source property
 * values) or cugraph::dummy_property_t::device_view() (if @p e_op does not access source property
 * values). Use update_edge_partition_src_property to fill the wrapper.
 * @param edge_partition_dst_value_input Device-copyable wrapper used to access destination input
 * property values (for the edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_partition_dst_property_t::device_view() (if @p e_op needs to access destination
 * property values) or cugraph::dummy_property_t::device_view() (if @p e_op does not access
 * destination property values). Use update_edge_partition_dst_property to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), property values for the source, and property values for the destination and returns
 * 1) thrust::nullopt (if invalid and to be discarded); 2) dummy (but valid) thrust::optional object
 * (e.g. thrust::optional<std::byte>{std::byte{0}}, if vertices are not tagged and
 * ReduceOp::value_type is void); 3) a tag (if vertices are tagged and ReduceOp::value_type is
 * void); 4) a value to be reduced using the @p reduce_op (if vertices are not tagged and
 * ReduceOp::value_type is not void); or 5) a tuple of a tag and a value to be reduced (if vertices
 * are tagged and ReduceOp::value_type is not void).
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @return Tuple of key values and payload values (if ReduceOp::value_type is not void) or just key
 * values (if ReduceOp::value_type is void).
 */
template <typename GraphViewType,
          typename VertexFrontierType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp>
std::conditional_t<
  !std::is_same_v<typename ReduceOp::value_type, void>,
  std::tuple<decltype(allocate_dataframe_buffer<typename VertexFrontierType::key_type>(
               0, rmm::cuda_stream_view{})),
             decltype(detail::allocate_optional_payload_buffer<typename ReduceOp::value_type>(
               0, rmm::cuda_stream_view{}))>,
  decltype(
    allocate_dataframe_buffer<typename VertexFrontierType::key_type>(0, rmm::cuda_stream_view{}))>
transform_reduce_v_frontier_outgoing_e_by_dst(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType const& frontier,
  size_t cur_frontier_bucket_idx,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgeOp e_op,
  ReduceOp reduce_op,
  bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  using weight_t  = typename GraphViewType::weight_type;
  using key_t     = typename VertexFrontierType::key_type;
  using payload_t = typename ReduceOp::value_type;

  CUGRAPH_EXPECTS(cur_frontier_bucket_idx < frontier.num_buckets(),
                  "Invalid input argument: invalid current bucket index.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  auto frontier_key_first = frontier.bucket(cur_frontier_bucket_idx).begin();
  auto frontier_key_last  = frontier.bucket(cur_frontier_bucket_idx).end();

  // 1. fill the buffer

  auto key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
  auto payload_buffer =
    detail::allocate_optional_payload_buffer<payload_t>(size_t{0}, handle.get_stream());
  rmm::device_scalar<size_t> buffer_idx(size_t{0}, handle.get_stream());
  std::vector<size_t> local_frontier_sizes{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& col_comm       = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    local_frontier_sizes = host_scalar_allgather(
      col_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
  }
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_frontier_key_buffer =
      allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
    vertex_t edge_partition_frontier_size = static_cast<vertex_t>(local_frontier_sizes[i]);
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      resize_dataframe_buffer(
        edge_partition_frontier_key_buffer, edge_partition_frontier_size, handle.get_stream());

      device_bcast(col_comm,
                   frontier_key_first,
                   get_dataframe_buffer_begin(edge_partition_frontier_key_buffer),
                   edge_partition_frontier_size,
                   i,
                   handle.get_stream());
    } else {
      resize_dataframe_buffer(
        edge_partition_frontier_key_buffer, edge_partition_frontier_size, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   frontier_key_first,
                   frontier_key_last,
                   get_dataframe_buffer_begin(edge_partition_frontier_key_buffer));
    }

    vertex_t const* edge_partition_frontier_src_first{nullptr};
    vertex_t const* edge_partition_frontier_src_last{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      edge_partition_frontier_src_first =
        get_dataframe_buffer_begin(edge_partition_frontier_key_buffer);
      edge_partition_frontier_src_last =
        get_dataframe_buffer_end(edge_partition_frontier_key_buffer);
    } else {
      edge_partition_frontier_src_first = thrust::get<0>(
        get_dataframe_buffer_begin(edge_partition_frontier_key_buffer).get_iterator_tuple());
      edge_partition_frontier_src_last = thrust::get<0>(
        get_dataframe_buffer_end(edge_partition_frontier_key_buffer).get_iterator_tuple());
    }

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    auto max_pushes =
      graph_view.use_dcs()
        ? thrust::transform_reduce(
            handle.get_thrust_policy(),
            edge_partition_frontier_src_first,
            edge_partition_frontier_src_last,
            [edge_partition,
             major_hypersparse_first =
               *(edge_partition.major_hypersparse_first())] __device__(auto src) {
              if (src < major_hypersparse_first) {
                auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
                return edge_partition.local_degree(src_offset);
              } else {
                auto src_hypersparse_idx =
                  edge_partition.major_hypersparse_idx_from_major_nocheck(src);
                return src_hypersparse_idx ? edge_partition.local_degree(
                                               edge_partition.major_offset_from_major_nocheck(
                                                 major_hypersparse_first) +
                                               *src_hypersparse_idx)
                                           : edge_t{0};
              }
            },
            edge_t{0},
            thrust::plus<edge_t>())
        : thrust::transform_reduce(
            handle.get_thrust_policy(),
            edge_partition_frontier_src_first,
            edge_partition_frontier_src_last,
            [edge_partition] __device__(auto src) {
              auto src_offset = edge_partition.major_offset_from_major_nocheck(src);
              return edge_partition.local_degree(src_offset);
            },
            edge_t{0},
            thrust::plus<edge_t>());

    auto new_buffer_size = buffer_idx.value(handle.get_stream()) + max_pushes;
    resize_dataframe_buffer(key_buffer, new_buffer_size, handle.get_stream());
    if constexpr (!std::is_same_v<payload_t, void>) {
      resize_dataframe_buffer(payload_buffer, new_buffer_size, handle.get_stream());
    }

    auto edge_partition_src_value_input_copy = edge_partition_src_value_input;
    auto edge_partition_dst_value_input_copy = edge_partition_dst_value_input;
    edge_partition_src_value_input_copy.set_local_edge_partition_idx(i);

    if (segment_offsets) {
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      std::vector<vertex_t> h_thresholds(detail::num_sparse_segments_per_vertex_partition +
                                         (graph_view.use_dcs() ? 1 : 0) - 1);
      h_thresholds[0] = edge_partition.major_range_first() + (*segment_offsets)[1];
      h_thresholds[1] = edge_partition.major_range_first() + (*segment_offsets)[2];
      if (graph_view.use_dcs()) {
        h_thresholds[2] = edge_partition.major_range_first() + (*segment_offsets)[3];
      }
      rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), handle.get_stream());
      raft::update_device(
        d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> d_offsets(d_thresholds.size(), handle.get_stream());
      thrust::lower_bound(handle.get_thrust_policy(),
                          edge_partition_frontier_src_first,
                          edge_partition_frontier_src_last,
                          d_thresholds.begin(),
                          d_thresholds.end(),
                          d_offsets.begin());
      std::vector<vertex_t> h_offsets(d_offsets.size());
      raft::update_host(h_offsets.data(), d_offsets.data(), d_offsets.size(), handle.get_stream());
      RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
      h_offsets.push_back(edge_partition_frontier_size);
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      if (h_offsets[0] > 0) {
        raft::grid_1d_block_t update_grid(
          h_offsets[0],
          detail::update_v_frontier_from_outgoing_e_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::update_v_frontier_from_outgoing_e_high_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer),
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[0],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (h_offsets[1] - h_offsets[0] > 0) {
        raft::grid_1d_warp_t update_grid(
          h_offsets[1] - h_offsets[0],
          detail::update_v_frontier_from_outgoing_e_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::update_v_frontier_from_outgoing_e_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[0],
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[1],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (h_offsets[2] - h_offsets[1] > 0) {
        raft::grid_1d_thread_t update_grid(
          h_offsets[2] - h_offsets[1],
          detail::update_v_frontier_from_outgoing_e_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::update_v_frontier_from_outgoing_e_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[1],
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[2],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
      if (edge_partition.dcs_nzd_vertex_count() && (h_offsets[3] - h_offsets[2] > 0)) {
        raft::grid_1d_thread_t update_grid(
          h_offsets[3] - h_offsets[2],
          detail::update_v_frontier_from_outgoing_e_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::update_v_frontier_from_outgoing_e_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[2],
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer) + h_offsets[3],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
    } else {
      if (edge_partition_frontier_size > 0) {
        raft::grid_1d_thread_t update_grid(
          edge_partition_frontier_size,
          detail::update_v_frontier_from_outgoing_e_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::update_v_frontier_from_outgoing_e_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            get_dataframe_buffer_begin(edge_partition_frontier_key_buffer),
            get_dataframe_buffer_end(edge_partition_frontier_key_buffer),
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(key_buffer),
            detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
            buffer_idx.data(),
            e_op);
      }
    }
  }

  // 2. reduce the buffer

  resize_dataframe_buffer(key_buffer, buffer_idx.value(handle.get_stream()), handle.get_stream());
  detail::resize_optional_payload_buffer<payload_t>(
    payload_buffer, size_dataframe_buffer(key_buffer), handle.get_stream());
  shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());
  detail::shrink_to_fit_optional_payload_buffer<payload_t>(payload_buffer, handle.get_stream());

  std::tie(key_buffer, payload_buffer) =
    detail::sort_and_reduce_buffer_elements<key_t, payload_t, ReduceOp>(
      handle, std::move(key_buffer), std::move(payload_buffer), reduce_op);
  if constexpr (GraphViewType::is_multi_gpu) {
    // FIXME: this step is unnecessary if row_comm_size== 1
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();

    std::vector<vertex_t> h_vertex_lasts(row_comm_size);
    for (size_t i = 0; i < h_vertex_lasts.size(); ++i) {
      h_vertex_lasts[i] = graph_view.vertex_partition_range_last(col_comm_rank * row_comm_size + i);
    }

    rmm::device_uvector<vertex_t> d_vertex_lasts(h_vertex_lasts.size(), handle.get_stream());
    raft::update_device(
      d_vertex_lasts.data(), h_vertex_lasts.data(), h_vertex_lasts.size(), handle.get_stream());
    rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(d_vertex_lasts.size(),
                                                            handle.get_stream());
    vertex_t const* src_first{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      src_first = get_dataframe_buffer_begin(key_buffer);
    } else {
      src_first = thrust::get<0>(get_dataframe_buffer_begin(key_buffer).get_iterator_tuple());
    }
    thrust::lower_bound(handle.get_thrust_policy(),
                        src_first,
                        src_first + size_dataframe_buffer(key_buffer),
                        d_vertex_lasts.begin(),
                        d_vertex_lasts.end(),
                        d_tx_buffer_last_boundaries.begin());
    std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
    raft::update_host(h_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.size(),
                      handle.get_stream());
    handle.sync_stream();
    std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
    std::adjacent_difference(
      h_tx_buffer_last_boundaries.begin(), h_tx_buffer_last_boundaries.end(), tx_counts.begin());

    auto rx_key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
    std::tie(rx_key_buffer, std::ignore) = shuffle_values(
      row_comm, get_dataframe_buffer_begin(key_buffer), tx_counts, handle.get_stream());
    key_buffer = std::move(rx_key_buffer);

    if constexpr (!std::is_same_v<payload_t, void>) {
      auto rx_payload_buffer = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
      std::tie(rx_payload_buffer, std::ignore) = shuffle_values(
        row_comm, get_dataframe_buffer_begin(payload_buffer), tx_counts, handle.get_stream());
      payload_buffer = std::move(rx_payload_buffer);
    }

    std::tie(key_buffer, payload_buffer) =
      detail::sort_and_reduce_buffer_elements<key_t, payload_t, ReduceOp>(
        handle, std::move(key_buffer), std::move(payload_buffer), reduce_op);
  }

  if constexpr (!std::is_same_v<payload_t, void>) {
    return std::make_tuple(std::move(key_buffer), std::move(payload_buffer));
  } else {
    return key_buffer;
  }
}

}  // namespace cugraph
