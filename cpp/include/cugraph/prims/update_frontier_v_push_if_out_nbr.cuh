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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/device_comm.cuh>
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
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {

namespace detail {

int32_t constexpr update_frontier_v_push_if_out_nbr_for_all_block_size = 512;

// we cannot use std::iterator_traits<Iterator>::value_type if Iterator is void* (reference to void
// is not allowed)
template <typename PayloadIterator, typename Enable = void>
struct optional_payload_buffer_value_type_t;

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<!std::is_same_v<PayloadIterator, void*>>> {
  using value = typename std::iterator_traits<PayloadIterator>::value_type;
};

template <typename PayloadIterator>
struct optional_payload_buffer_value_type_t<
  PayloadIterator,
  std::enable_if_t<std::is_same_v<PayloadIterator, void*>>> {
  using value = void;
};

// FIXME: to silence the spurious warning (missing return statement ...) due to the nvcc bug
// (https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-
// function-in-constexpr-if-fun)
#if 1
template <typename payload_t, std::enable_if_t<std::is_same_v<payload_t, void>>* = nullptr>
std::byte allocate_optional_payload_buffer(size_t size, cudaStream_t stream)
{
  return std::byte{0};  // dummy
}

template <typename payload_t, std::enable_if_t<!std::is_same_v<payload_t, void>>* = nullptr>
auto allocate_optional_payload_buffer(size_t size, cudaStream_t stream)
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
    size_t{0}, cudaStream_t{nullptr}))> optional_payload_buffer)
{
  return get_dataframe_buffer_begin(optional_payload_buffer);
}
#else
auto allocate_optional_payload_buffer = [](size_t size, cudaStream_t stream) {
  if constexpr (std::is_same_v<payload_t, void>) {
    return std::byte{0};  // dummy
  } else {
    return allocate_dataframe_buffer<payload_t>(size, stream);
  }
};

auto get_optional_payload_buffer_begin = [](auto& optional_payload_buffer) {
  if constexpr (std::is_same_v<payload_t, void>) {
    return static_cast<std::byte*>(nullptr);
  } else {
    return get_dataframe_buffer_begin(optional_payload_buffer);
  }
};
#endif

// FIXME: a temporary workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
// in the else part in if constexpr else statement that involves device lambda
template <typename vertex_t,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp,
          typename key_t,
          bool multi_gpu>
struct call_v_op_t {
  VertexValueInputIterator vertex_value_input_first{};
  VertexValueOutputIterator vertex_value_output_first{};
  VertexOp v_op{};
  vertex_partition_device_view_t<vertex_t, multi_gpu> vertex_partition{};
  size_t invalid_bucket_idx;

  template <typename key_type = key_t, typename vertex_type = vertex_t>
  __device__ std::enable_if_t<std::is_same_v<key_type, vertex_type>, uint8_t> operator()(
    key_t key) const
  {
    auto v_offset    = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(key);
    auto v_val       = *(vertex_value_input_first + v_offset);
    auto v_op_result = v_op(key, v_val);
    if (v_op_result) {
      *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
      return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
    } else {
      return std::numeric_limits<uint8_t>::max();
    }
  }

  template <typename key_type = key_t, typename vertex_type = vertex_t>
  __device__ std::enable_if_t<!std::is_same_v<key_type, vertex_type>, uint8_t> operator()(
    key_t key) const
  {
    auto v_offset =
      vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(thrust::get<0>(key));
    auto v_val       = *(vertex_value_input_first + v_offset);
    auto v_op_result = v_op(key, v_val);
    if (v_op_result) {
      *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
      return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
    } else {
      return std::numeric_limits<uint8_t>::max();
    }
  }
};

// FIXME: a temporary workaround for cudaErrorInvalidDeviceFunction error when device lambda is used
// after if constexpr else statement that involves device lambda (bug report submitted)
template <typename key_t>
struct check_invalid_bucket_idx_t {
  __device__ bool operator()(thrust::tuple<uint8_t, key_t> pair)
  {
    return thrust::get<0>(pair) == std::numeric_limits<uint8_t>::max();
  }
};

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
  using key_t = typename std::iterator_traits<BufferKeyOutputIterator>::value_type;
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
__global__ void for_all_frontier_src_for_all_nbr_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_hypersparse_first,
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
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
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
  auto src_start_offset =
    static_cast<size_t>(major_hypersparse_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(tid);

  __shared__ edge_t
    warp_local_degree_inclusive_sums[update_frontier_v_push_if_out_nbr_for_all_block_size];
  __shared__ edge_t
    warp_key_local_edge_offsets[update_frontier_v_push_if_out_nbr_for_all_block_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  __shared__ size_t buffer_warp_start_indices[update_frontier_v_push_if_out_nbr_for_all_block_size /
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
__global__ void for_all_frontier_src_for_all_nbr_low_degree(
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
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
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
    warp_local_degree_inclusive_sums[update_frontier_v_push_if_out_nbr_for_all_block_size];
  __shared__ edge_t
    warp_key_local_edge_offsets[update_frontier_v_push_if_out_nbr_for_all_block_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  __shared__ size_t buffer_warp_start_indices[update_frontier_v_push_if_out_nbr_for_all_block_size /
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
__global__ void for_all_frontier_src_for_all_nbr_mid_degree(
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
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
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
  static_assert(update_frontier_v_push_if_out_nbr_for_all_block_size % raft::warp_size() == 0);
  auto const warp_id = threadIdx.x / raft::warp_size();
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  __shared__ size_t buffer_warp_start_indices[update_frontier_v_push_if_out_nbr_for_all_block_size /
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
__global__ void for_all_frontier_src_for_all_nbr_high_degree(
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
  using key_t    = typename std::iterator_traits<KeyIterator>::value_type;
  static_assert(
    std::is_same_v<key_t, typename std::iterator_traits<BufferKeyOutputIterator>::value_type>);
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

  using BlockScan = cub::BlockScan<edge_t, update_frontier_v_push_if_out_nbr_for_all_block_size>;
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
        (update_frontier_v_push_if_out_nbr_for_all_block_size - 1)) /
       update_frontier_v_push_if_out_nbr_for_all_block_size) *
      update_frontier_v_push_if_out_nbr_for_all_block_size;
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

template <typename BufferKeyOutputIterator, typename BufferPayloadOutputIterator, typename ReduceOp>
size_t sort_and_reduce_buffer_elements(raft::handle_t const& handle,
                                       BufferKeyOutputIterator buffer_key_output_first,
                                       BufferPayloadOutputIterator buffer_payload_output_first,
                                       size_t num_buffer_elements,
                                       ReduceOp reduce_op)
{
  using key_t = typename std::iterator_traits<BufferKeyOutputIterator>::value_type;
  using payload_t =
    typename optional_payload_buffer_value_type_t<BufferPayloadOutputIterator>::value;

  auto execution_policy = handle.get_thrust_policy();
  if constexpr (std::is_same_v<payload_t, void>) {
    thrust::sort(
      execution_policy, buffer_key_output_first, buffer_key_output_first + num_buffer_elements);
  } else {
    thrust::sort_by_key(execution_policy,
                        buffer_key_output_first,
                        buffer_key_output_first + num_buffer_elements,
                        buffer_payload_output_first);
  }

  size_t num_reduced_buffer_elements{};
  if constexpr (std::is_same_v<payload_t, void>) {
    auto it = thrust::unique(
      execution_policy, buffer_key_output_first, buffer_key_output_first + num_buffer_elements);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(buffer_key_output_first, it));
  } else if constexpr (std::is_same<ReduceOp, reduce_op::any<typename ReduceOp::type>>::value) {
    auto it = thrust::unique_by_key(execution_policy,
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(buffer_key_output_first, thrust::get<0>(it)));
  } else {
    rmm::device_uvector<key_t> keys(num_buffer_elements, handle.get_stream());
    auto value_buffer =
      allocate_dataframe_buffer<payload_t>(num_buffer_elements, handle.get_stream());
    auto it = thrust::reduce_by_key(execution_policy,
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first,
                                    keys.begin(),
                                    get_dataframe_buffer_begin(value_buffer),
                                    thrust::equal_to<key_t>(),
                                    reduce_op);
    num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(keys.begin(), thrust::get<0>(it)));
    // FIXME: this copy can be replaced by move
    thrust::copy(execution_policy,
                 keys.begin(),
                 keys.begin() + num_reduced_buffer_elements,
                 buffer_key_output_first);
    thrust::copy(execution_policy,
                 get_dataframe_buffer_begin(value_buffer),
                 get_dataframe_buffer_begin(value_buffer) + num_reduced_buffer_elements,
                 buffer_payload_output_first);
  }

  return num_reduced_buffer_elements;
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

  auto const& cur_frontier_bucket = frontier.get_bucket(cur_frontier_bucket_idx);
  vertex_t const* local_frontier_vertex_first{nullptr};
  vertex_t const* local_frontier_vertex_last{nullptr};
  if constexpr (std::is_same_v<key_t, vertex_t>) {
    local_frontier_vertex_first = cur_frontier_bucket.begin();
    local_frontier_vertex_last  = cur_frontier_bucket.end();
  } else {
    local_frontier_vertex_first = thrust::get<0>(cur_frontier_bucket.begin().get_iterator_tuple());
    local_frontier_vertex_last  = thrust::get<0>(cur_frontier_bucket.end().get_iterator_tuple());
  }

  std::vector<size_t> local_frontier_sizes{};
  if (GraphViewType::is_multi_gpu) {
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

    auto execution_policy = handle.get_thrust_policy();
    if (GraphViewType::is_multi_gpu) {
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
      auto use_dcs =
        segment_offsets
          ? ((*segment_offsets).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
          : false;

      ret += use_dcs
               ? thrust::transform_reduce(
                   execution_policy,
                   frontier_vertices.begin(),
                   frontier_vertices.end(),
                   [edge_partition,
                    major_hypersparse_first =
                      edge_partition.major_range_first() +
                      (*segment_offsets)
                        [detail::num_sparse_segments_per_vertex_partition]] __device__(auto major) {
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
                   execution_policy,
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
        execution_policy,
        local_frontier_vertex_first,
        local_frontier_vertex_last,
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

// FIXME: this documentation needs to be updated due to (tagged-)vertex support
/**
 * @brief Update (tagged-)vertex frontier and (tagged-)vertex property values iterating over the
 * outgoing edges from the frontier.
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
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex property variables.
 * @tparam VertexOp Type of the binary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontier class object for vertex frontier managements. This object includes
 * multiple bucket objects.
 * @param cur_frontier_bucket_idx Index of the VertexFrontier bucket holding vertices for the
 * current iteration.
 * @param next_frontier_bucket_indices Indices of the VertexFrontier buckets to store new frontier
 * vertices for the next iteration.
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
 * weight), property values for the source, and property values for the destination and returns a
 * value to be reduced the @p reduce_op.
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param v_op Ternary operator takes (tagged-)vertex ID, *(@p vertex_value_input_first + i) (where
 * i is [0, @p graph_view.local_vertex_partition_range_size())) and reduced value of the @p e_op
 * outputs for this vertex and returns the target bucket index (for frontier update) and new verrtex
 * property values (to update *(@p vertex_value_output_first + i)). The target bucket index should
 * either be VertexFrontierType::kInvalidBucketIdx or an index in @p next_frontier_bucket_indices.
 */
template <typename GraphViewType,
          typename VertexFrontierType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp>
void update_frontier_v_push_if_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType& frontier,
  size_t cur_frontier_bucket_idx,
  std::vector<size_t> const& next_frontier_bucket_indices,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgeOp e_op,
  ReduceOp reduce_op,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  VertexValueInputIterator vertex_value_input_first,
  // FIXME: if vertices in the frontier are tagged, we should have an option to access with (vertex,
  // tag) pair (currently we can access only with vertex, we may use cuco::static_map for this
  // purpose)
  // FIXME: currently, it is undefined behavior if vertices in the frontier are tagged and the same
  // vertex property is updated by multiple v_op invocations with the same vertex but with different
  // tags.
  VertexValueOutputIterator vertex_value_output_first,
  // FIXME: this takes (tagged-)vertex ID in addition, think about consistency with the other
  // primitives.
  VertexOp v_op)
{
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  using weight_t  = typename GraphViewType::weight_type;
  using key_t     = typename VertexFrontierType::key_type;
  using payload_t = typename ReduceOp::type;

  auto frontier_key_first = frontier.get_bucket(cur_frontier_bucket_idx).begin();
  auto frontier_key_last  = frontier.get_bucket(cur_frontier_bucket_idx).end();

  // 1. fill the buffer

  auto key_buffer = allocate_dataframe_buffer<key_t>(size_t{0}, handle.get_stream());
  auto payload_buffer =
    detail::allocate_optional_payload_buffer<payload_t>(size_t{0}, handle.get_stream());
  rmm::device_scalar<size_t> buffer_idx(size_t{0}, handle.get_stream());
  std::vector<size_t> local_frontier_sizes{};
  if (GraphViewType::is_multi_gpu) {
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
    if (GraphViewType::is_multi_gpu) {
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
    auto use_dcs =
      segment_offsets
        ? ((*segment_offsets).size() > (detail::num_sparse_segments_per_vertex_partition + 1))
        : false;

    auto execution_policy = handle.get_thrust_policy();
    auto max_pushes =
      use_dcs ? thrust::transform_reduce(
                  execution_policy,
                  edge_partition_frontier_src_first,
                  edge_partition_frontier_src_last,
                  [edge_partition,
                   major_hypersparse_first =
                     edge_partition.major_range_first() +
                     (*segment_offsets)
                       [detail::num_sparse_segments_per_vertex_partition]] __device__(auto src) {
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
                  execution_policy,
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
                                         (use_dcs ? 1 : 0) - 1);
      h_thresholds[0] = edge_partition.major_range_first() + (*segment_offsets)[1];
      h_thresholds[1] = edge_partition.major_range_first() + (*segment_offsets)[2];
      if (use_dcs) { h_thresholds[2] = edge_partition.major_range_first() + (*segment_offsets)[3]; }
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
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_src_for_all_nbr_high_degree<GraphViewType>
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
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_src_for_all_nbr_mid_degree<GraphViewType>
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
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_src_for_all_nbr_low_degree<GraphViewType>
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
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_frontier_src_for_all_nbr_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[3],
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
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_src_for_all_nbr_low_degree<GraphViewType>
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

  auto num_buffer_elements = detail::sort_and_reduce_buffer_elements(
    handle,
    get_dataframe_buffer_begin(key_buffer),
    detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
    buffer_idx.value(handle.get_stream()),
    reduce_op);
  if (GraphViewType::is_multi_gpu) {
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
                        src_first + num_buffer_elements,
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

    num_buffer_elements = detail::sort_and_reduce_buffer_elements(
      handle,
      get_dataframe_buffer_begin(key_buffer),
      detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer),
      size_dataframe_buffer(key_buffer),
      reduce_op);
  }

  // 3. update vertex properties and frontier

  if (num_buffer_elements > 0) {
    static_assert(VertexFrontierType::kNumBuckets <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(num_buffer_elements, handle.get_stream());

    auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
      graph_view.local_vertex_partition_view());

    if constexpr (!std::is_same_v<payload_t, void>) {
      auto key_payload_pair_first = thrust::make_zip_iterator(
        thrust::make_tuple(get_dataframe_buffer_begin(key_buffer),
                           detail::get_optional_payload_buffer_begin<payload_t>(payload_buffer)));
      thrust::transform(
        handle.get_thrust_policy(),
        key_payload_pair_first,
        key_payload_pair_first + num_buffer_elements,
        bucket_indices.begin(),
        [vertex_value_input_first,
         vertex_value_output_first,
         v_op,
         vertex_partition,
         invalid_bucket_idx = VertexFrontierType::kInvalidBucketIdx] __device__(auto pair) {
          auto key     = thrust::get<0>(pair);
          auto payload = thrust::get<1>(pair);
          vertex_t v_offset{};
          if constexpr (std::is_same_v<key_t, vertex_t>) {
            v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(key);
          } else {
            v_offset = vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(
              thrust::get<0>(key));
          }
          auto v_val       = *(vertex_value_input_first + v_offset);
          auto v_op_result = v_op(key, v_val, payload);
          if (v_op_result) {
            *(vertex_value_output_first + v_offset) = thrust::get<1>(*v_op_result);
            return static_cast<uint8_t>(thrust::get<0>(*v_op_result));
          } else {
            return std::numeric_limits<uint8_t>::max();
          }
        });

      resize_dataframe_buffer(payload_buffer, size_t{0}, handle.get_stream());
      shrink_to_fit_dataframe_buffer(payload_buffer, handle.get_stream());
    } else {
      thrust::transform(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(key_buffer),
        get_dataframe_buffer_begin(key_buffer) + num_buffer_elements,
        bucket_indices.begin(),
        detail::call_v_op_t<vertex_t,
                            VertexValueInputIterator,
                            VertexValueOutputIterator,
                            VertexOp,
                            key_t,
                            GraphViewType::is_multi_gpu>{vertex_value_input_first,
                                                         vertex_value_output_first,
                                                         v_op,
                                                         vertex_partition,
                                                         VertexFrontierType::kInvalidBucketIdx});
    }

    auto bucket_key_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(bucket_indices.begin(), get_dataframe_buffer_begin(key_buffer)));
    bucket_indices.resize(
      thrust::distance(bucket_key_pair_first,
                       thrust::remove_if(handle.get_thrust_policy(),
                                         bucket_key_pair_first,
                                         bucket_key_pair_first + num_buffer_elements,
                                         detail::check_invalid_bucket_idx_t<key_t>())),
      handle.get_stream());
    resize_dataframe_buffer(key_buffer, bucket_indices.size(), handle.get_stream());
    bucket_indices.shrink_to_fit(handle.get_stream());
    shrink_to_fit_dataframe_buffer(key_buffer, handle.get_stream());

    frontier.insert_to_buckets(bucket_indices.begin(),
                               bucket_indices.end(),
                               get_dataframe_buffer_begin(key_buffer),
                               next_frontier_bucket_indices);
  }
}

}  // namespace cugraph
