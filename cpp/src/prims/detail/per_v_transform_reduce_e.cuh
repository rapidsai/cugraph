/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/detail/multi_stream_utils.cuh"
#include "prims/detail/optional_dataframe_buffer.hpp"
#include "prims/detail/prim_functors.cuh"
#include "prims/detail/prim_utils.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/reduce_op.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/host_staging_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/set_operations.h>
#include <thrust/transform_reduce.h>
#include <thrust/type_traits/integer_sequence.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

int32_t constexpr per_v_transform_reduce_e_kernel_block_size                        = 256;
int32_t constexpr per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size = 128;

template <typename Iterator, typename default_t, typename Enable = void>
struct iterator_value_type_or_default_t;

template <typename Iterator, typename default_t>
struct iterator_value_type_or_default_t<Iterator,
                                        default_t,
                                        std::enable_if_t<std::is_same_v<Iterator, void*>>> {
  using value_type = default_t;  // if Iterator is invalid (void*), value_type = default_t
};

template <typename Iterator, typename default_t>
struct iterator_value_type_or_default_t<Iterator,
                                        default_t,
                                        std::enable_if_t<!std::is_same_v<Iterator, void*>>> {
  using value_type = typename thrust::iterator_traits<
    Iterator>::value_type;  // if iterator is valid, value_type = typename
                            // thrust::iterator_traits<Iterator>::value_type
};

template <typename GraphViewType,
          typename key_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename PredOp>
__device__ auto init_pred_op(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> const& edge_partition,
  EdgePartitionSrcValueInputWrapper const& edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper const& edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper const& edge_partition_e_value_input,
  PredOp const& pred_op,
  key_t key,
  typename GraphViewType::vertex_type major_offset,
  typename GraphViewType::vertex_type const* indices,
  typename GraphViewType::edge_type edge_offset)
{
  if constexpr (std::is_same_v<
                  PredOp,
                  const_true_e_op_t<key_t,
                                    typename GraphViewType::vertex_type,
                                    typename EdgePartitionSrcValueInputWrapper::value_type,
                                    typename EdgePartitionDstValueInputWrapper::value_type,
                                    typename EdgePartitionEdgeValueInputWrapper::value_type,
                                    GraphViewType::is_storage_transposed>>) {
    return call_const_true_e_op_t<typename GraphViewType::edge_type>{};
  } else {
    return call_e_op_t<GraphViewType,
                       key_t,
                       EdgePartitionSrcValueInputWrapper,
                       EdgePartitionDstValueInputWrapper,
                       EdgePartitionEdgeValueInputWrapper,
                       PredOp>{edge_partition,
                               edge_partition_src_value_input,
                               edge_partition_dst_value_input,
                               edge_partition_e_value_input,
                               pred_op,
                               key,
                               major_offset,
                               indices,
                               edge_offset};
  }
}

template <typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          typename result_t,
          typename TransformOp,
          typename ReduceOp,
          typename PredOp,
          typename ResultValueOutputIteratorOrWrapper>
struct transform_and_atomic_reduce_t {
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> const& edge_partition{};
  vertex_t const* indices{nullptr};
  TransformOp const& transform_op{};
  PredOp const& pred_op{};
  ResultValueOutputIteratorOrWrapper& result_value_output{};

  __device__ void operator()(edge_t i) const
  {
    if (pred_op(i)) {
      auto e_op_result  = transform_op(i);
      auto minor        = indices[i];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      if constexpr (multi_gpu) {
        reduce_op::atomic_reduce<ReduceOp>(result_value_output, minor_offset, e_op_result);
      } else {
        reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
      }
    }
  }
};

template <bool update_major,
          typename vertex_t,
          typename edge_t,
          bool multi_gpu,
          typename result_t,
          typename TransformOp,
          typename ReduceOp,
          typename PredOp,
          typename ResultValueOutputIteratorOrWrapper>
__device__ void update_result_value_output(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> const& edge_partition,
  vertex_t const* indices,
  edge_t local_degree,
  TransformOp const& transform_op,
  result_t init,
  ReduceOp const& reduce_op,
  PredOp const& pred_op,
  size_t output_idx /* relevent only when update_major === true */,
  ResultValueOutputIteratorOrWrapper& result_value_output)
{
  if constexpr (update_major) {
    result_t val{};
    if constexpr (std::is_same_v<PredOp, call_const_true_e_op_t<edge_t>>) {
      if constexpr (std::is_same_v<ReduceOp,
                                   reduce_op::any<result_t>>) {  // init is selected only when no
                                                                 // edges return a valid value
        val = init;
        for (edge_t i = 0; i < local_degree; ++i) {
          val = transform_op(i);
          break;
        }
      } else {
        val = thrust::transform_reduce(thrust::seq,
                                       thrust::make_counting_iterator(edge_t{0}),
                                       thrust::make_counting_iterator(local_degree),
                                       transform_op,
                                       init,
                                       reduce_op);
      }
    } else {
      val = init;
      if constexpr (std::is_same_v<ReduceOp,
                                   reduce_op::any<result_t>>) {  // init is selected only when no
                                                                 // edges return a valid value
        auto first = thrust::make_counting_iterator(edge_t{0});
        auto last  = thrust::make_counting_iterator(local_degree);
        auto it    = thrust::find_if(thrust::seq, first, last, pred_op);
        if (it != last) { val = transform_op(*it); }
      } else {
        for (edge_t i = 0; i < local_degree; ++i) {
          if (pred_op(i)) {
            auto tmp = transform_op(i);
            val      = reduce_op(val, tmp);
          }
        }
      }
    }
    *(result_value_output + output_idx) = val;
  } else {
    thrust::for_each(thrust::seq,
                     thrust::make_counting_iterator(edge_t{0}),
                     thrust::make_counting_iterator(local_degree),
                     transform_and_atomic_reduce_t<vertex_t,
                                                   edge_t,
                                                   multi_gpu,
                                                   result_t,
                                                   TransformOp,
                                                   ReduceOp,
                                                   PredOp,
                                                   ResultValueOutputIteratorOrWrapper>{
                       edge_partition, indices, transform_op, pred_op, result_value_output});
  }
}

template <bool update_major,
          typename GraphViewType,
          typename OptionalKeyIterator,  // invalid if void*
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if !update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T>
__global__ static void per_v_transform_reduce_e_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  OptionalKeyIterator key_first,
  OptionalKeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  cuda::std::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevant only if update_major == true */,
  ReduceOp reduce_op,
  PredOp pred_op)
{
  constexpr bool use_input_key = !std::is_same_v<OptionalKeyIterator, void*>;
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true
  static_assert(update_major || !use_input_key);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t =
    typename iterator_value_type_or_default_t<OptionalKeyIterator, vertex_t>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  size_t key_count{};
  if constexpr (use_input_key) {
    key_count = static_cast<size_t>(cuda::std::distance(key_first, key_last));
  } else {
    key_count = *(edge_partition.dcs_nzd_vertex_count());
  }

  while (idx < key_count) {
    key_t key{};
    vertex_t major{};
    cuda::std::optional<vertex_t> major_idx{};
    if constexpr (use_input_key) {
      key       = *(key_first + idx);
      major     = thrust_tuple_get_or_identity<key_t, 0>(key);
      major_idx = edge_partition.major_idx_from_major_nocheck(major);
    } else {
      key = *(edge_partition.major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
      major                   = key;
      auto major_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                                    edge_partition.major_range_first());
      major_idx = major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    }

    size_t output_idx = use_input_key ? idx : (major - *(edge_partition).major_hypersparse_first());
    if (major_idx) {
      auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
      vertex_t const* indices{nullptr};
      edge_t edge_offset{};
      edge_t local_degree{};
      thrust::tie(indices, edge_offset, local_degree) =
        edge_partition.local_edges(static_cast<vertex_t>(*major_idx));

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
                                           edge_offset};

      auto call_pred_op = init_pred_op<GraphViewType, key_t>(edge_partition,
                                                             edge_partition_src_value_input,
                                                             edge_partition_dst_value_input,
                                                             edge_partition_e_value_input,
                                                             pred_op,
                                                             key,
                                                             major_offset,
                                                             indices,
                                                             edge_offset);

      if (edge_partition_e_mask) {
        update_result_value_output<update_major>(
          edge_partition,
          indices,
          local_degree,
          call_e_op,
          init,
          reduce_op,
          [&edge_partition_e_mask, &call_pred_op, edge_offset] __device__(edge_t i) {
            if ((*edge_partition_e_mask).get(edge_offset + i)) {
              return call_pred_op(i);
            } else {
              return false;
            }
          },
          output_idx,
          result_value_output);
      } else {
        update_result_value_output<update_major>(edge_partition,
                                                 indices,
                                                 local_degree,
                                                 call_e_op,
                                                 init,
                                                 reduce_op,
                                                 call_pred_op,
                                                 output_idx,
                                                 result_value_output);
      }
    } else {
      if constexpr (update_major) { *(result_value_output + output_idx) = init; }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if !update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T>
__global__ static void per_v_transform_reduce_e_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  cuda::std::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevant only if update_major == true */,
  ReduceOp reduce_op,
  PredOp pred_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  auto num_keys = static_cast<size_t>(cuda::std::distance(key_first, key_last));
  while (idx < num_keys) {
    auto key   = *(key_first + idx);
    auto major = thrust_tuple_get_or_identity<key_t, 0>(key);

    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));

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
                                         edge_offset};

    auto call_pred_op = init_pred_op<GraphViewType, key_t>(edge_partition,
                                                           edge_partition_src_value_input,
                                                           edge_partition_dst_value_input,
                                                           edge_partition_e_value_input,
                                                           pred_op,
                                                           key,
                                                           major_offset,
                                                           indices,
                                                           edge_offset);

    if (edge_partition_e_mask) {
      update_result_value_output<update_major>(
        edge_partition,
        indices,
        local_degree,
        call_e_op,
        init,
        reduce_op,
        [&edge_partition_e_mask, &call_pred_op, edge_offset] __device__(edge_t i) {
          if ((*edge_partition_e_mask).get(edge_offset + i)) {
            return call_pred_op(i);
          } else {
            return false;
          }
        },
        idx,
        result_value_output);
    } else {
      update_result_value_output<update_major>(edge_partition,
                                               indices,
                                               local_degree,
                                               call_e_op,
                                               init,
                                               reduce_op,
                                               call_pred_op,
                                               idx,
                                               result_value_output);
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if !update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T>
__global__ static void per_v_transform_reduce_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  cuda::std::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevant only if update_major == true */,
  T identity_element /* relevant only if update_major == true */,
  ReduceOp reduce_op,
  PredOp pred_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = T;
  using key_t         = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(per_v_transform_reduce_e_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  using WarpReduce = cub::WarpReduce<
    std::conditional_t<std::is_same_v<ReduceOp, reduce_op::any<T>>, int32_t, e_op_result_t>>;
  [[maybe_unused]] __shared__
    std::conditional_t<update_major, typename WarpReduce::TempStorage, std::byte /* dummy */>
      temp_storage[update_major ? (per_v_transform_reduce_e_kernel_block_size / raft::warp_size())
                                : int32_t{1} /* dummy */];

  while (idx < static_cast<size_t>(cuda::std::distance(key_first, key_last))) {
    auto key   = *(key_first + idx);
    auto major = thrust_tuple_get_or_identity<key_t, 0>(key);

    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);

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
                                         edge_offset};

    auto call_pred_op = init_pred_op<GraphViewType, key_t>(edge_partition,
                                                           edge_partition_src_value_input,
                                                           edge_partition_dst_value_input,
                                                           edge_partition_e_value_input,
                                                           pred_op,
                                                           key,
                                                           major_offset,
                                                           indices,
                                                           edge_offset);

    [[maybe_unused]] std::conditional_t<update_major, T, std::byte /* dummy */>
      reduced_e_op_result{};
    [[maybe_unused]] std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                        int32_t,
                                        std::byte /* dummy */> first_valid_lane_id{};
    if constexpr (update_major) {
      reduced_e_op_result =
        (lane_id == 0) ? init : identity_element;  // init == identity_element for reduce_op::any<T>
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        first_valid_lane_id = raft::warp_size();
      }
    }

    // FIXME: Remove once upgraded to CCCL version 3.x
#if CCCL_MAJOR_VERSION >= 3
    using cuda::minimum;
#else
    using minimum = cub::Min;
#endif

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
          raft::warp_size();
        for (size_t i = lane_id; i < rounded_up_local_degree; i += raft::warp_size()) {
          cuda::std::optional<T> e_op_result{cuda::std::nullopt};
          if ((i < static_cast<size_t>(local_degree)) &&
              (*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_lane_id = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                  .Reduce(e_op_result ? lane_id : raft::warp_size(), minimum{});
          first_valid_lane_id = __shfl_sync(raft::warp_full_mask(), first_valid_lane_id, int{0});
          if (lane_id == first_valid_lane_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_lane_id != raft::warp_size()) { break; }
        }
      } else {
        for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
          if ((*edge_partition_e_mask).get(edge_offset + i) & call_pred_op(i)) {
            auto e_op_result = call_e_op(i);
            if constexpr (update_major) {
              reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
            } else {
              auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(indices[i]);
              if constexpr (GraphViewType::is_multi_gpu) {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output, minor_offset, e_op_result);
              } else {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
              }
            }
          }
        }
      }
    } else {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
          raft::warp_size();
        for (size_t i = lane_id; i < rounded_up_local_degree; i += raft::warp_size()) {
          cuda::std::optional<T> e_op_result{cuda::std::nullopt};
          if (i < static_cast<size_t>(local_degree) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_lane_id = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                  .Reduce(e_op_result ? lane_id : raft::warp_size(), minimum{});
          first_valid_lane_id = __shfl_sync(raft::warp_full_mask(), first_valid_lane_id, int{0});
          if (lane_id == first_valid_lane_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_lane_id != raft::warp_size()) { break; }
        }
      } else {
        for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
          if (call_pred_op(i)) {
            auto e_op_result = call_e_op(i);
            if constexpr (update_major) {
              reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
            } else {
              auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(indices[i]);
              if constexpr (GraphViewType::is_multi_gpu) {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output, minor_offset, e_op_result);
              } else {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
              }
            }
          }
        }
      }
    }

    if constexpr (update_major) {
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        if (lane_id == ((first_valid_lane_id == raft::warp_size()) ? 0 : first_valid_lane_id)) {
          *(result_value_output + idx) = reduced_e_op_result;
        }
      } else {
        reduced_e_op_result = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                .Reduce(reduced_e_op_result, reduce_op);
        if (lane_id == 0) { *(result_value_output + idx) = reduced_e_op_result; }
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool update_major,
          typename GraphViewType,
          typename KeyIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if !update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T>
__global__ static void per_v_transform_reduce_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator key_first,
  KeyIterator key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  cuda::std::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  ResultValueOutputIteratorOrWrapper result_value_output,
  EdgeOp e_op,
  T init /* relevant only if update_major == true */,
  T identity_element /* relevant only if update_major == true */,
  ReduceOp reduce_op,
  PredOp pred_op)
{
  static_assert(update_major || reduce_op::has_compatible_raft_comms_op_v<
                                  ReduceOp>);  // atomic_reduce is defined only when
                                               // has_compatible_raft_comms_op_t<ReduceOp> is true

  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using e_op_result_t = T;
  using key_t         = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<
    std::conditional_t<std::is_same_v<ReduceOp, reduce_op::any<T>>, int32_t, e_op_result_t>,
    std::is_same_v<ReduceOp, reduce_op::any<T>>
      ? per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size
      : per_v_transform_reduce_e_kernel_block_size>;
  [[maybe_unused]] __shared__
    std::conditional_t<update_major, typename BlockReduce::TempStorage, std::byte /* dummy */>
      temp_storage;
  [[maybe_unused]] __shared__
    std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                       int32_t,
                       std::byte /* dummy */> output_thread_id;

  while (idx < static_cast<size_t>(cuda::std::distance(key_first, key_last))) {
    auto key   = *(key_first + idx);
    auto major = thrust_tuple_get_or_identity<key_t, 0>(key);

    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);

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
                                         edge_offset};

    auto call_pred_op = init_pred_op<GraphViewType, key_t>(edge_partition,
                                                           edge_partition_src_value_input,
                                                           edge_partition_dst_value_input,
                                                           edge_partition_e_value_input,
                                                           pred_op,
                                                           key,
                                                           major_offset,
                                                           indices,
                                                           edge_offset);

    [[maybe_unused]] std::conditional_t<update_major, T, std::byte /* dummy */>
      reduced_e_op_result{};
    [[maybe_unused]] std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                        int32_t,
                                        std::byte /* dummy */> first_valid_thread_id{};
    if constexpr (update_major) {
      reduced_e_op_result = threadIdx.x == 0
                              ? init
                              : identity_element;  // init == identity_element for reduce_op::any<T>
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        first_valid_thread_id = per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size;
      }
    }

    // FIXME: Remove once upgraded to CCCL version 3.x
#if CCCL_MAJOR_VERSION >= 3
    using cuda::minimum;
#else
    using minimum = cub::Min;
#endif

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) +
            (per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size - 1)) /
           per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size) *
          per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          cuda::std::optional<T> e_op_result{cuda::std::nullopt};
          if ((i < static_cast<size_t>(local_degree)) &&
              (*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result
                        ? threadIdx.x
                        : per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size,
                      minimum{});
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          first_valid_thread_id = output_thread_id;
          if (threadIdx.x == first_valid_thread_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_thread_id !=
              per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size) {
            break;
          }
        }
      } else {
        for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
          if ((*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            auto e_op_result = call_e_op(i);
            if constexpr (update_major) {
              reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
            } else {
              auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(indices[i]);
              if constexpr (GraphViewType::is_multi_gpu) {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output, minor_offset, e_op_result);
              } else {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
              }
            }
          }
        }
      }
    } else {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) +
            (per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size - 1)) /
           per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size) *
          per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          cuda::std::optional<T> e_op_result{cuda::std::nullopt};
          if ((i < static_cast<size_t>(local_degree)) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result
                        ? threadIdx.x
                        : per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size,
                      minimum{});
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          first_valid_thread_id = output_thread_id;
          if (threadIdx.x == first_valid_thread_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_thread_id !=
              per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size) {
            break;
          }
        }
      } else {
        for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
          if (call_pred_op(i)) {
            auto e_op_result = call_e_op(i);
            if constexpr (update_major) {
              reduced_e_op_result = reduce_op(reduced_e_op_result, e_op_result);
            } else {
              auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(indices[i]);
              if constexpr (GraphViewType::is_multi_gpu) {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output, minor_offset, e_op_result);
              } else {
                reduce_op::atomic_reduce<ReduceOp>(result_value_output + minor_offset, e_op_result);
              }
            }
          }
        }
      }
    }

    if constexpr (update_major) {
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        if (threadIdx.x == ((first_valid_thread_id ==
                             per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size)
                              ? 0
                              : first_valid_thread_id)) {
          *(result_value_output + idx) = reduced_e_op_result;
        }
      } else {
        reduced_e_op_result = BlockReduce(temp_storage).Reduce(reduced_e_op_result, reduce_op);
        if (threadIdx.x == 0) { *(result_value_output + idx) = reduced_e_op_result; }
      }
    }

    idx += gridDim.x;
  }
}

template <typename InputValueIterator, typename OutputOffsetIterator, typename OutputValueIterator>
void copy_valid_offset_value_pairs(
  InputValueIterator input_value_first,
  InputValueIterator input_value_last,
  OutputOffsetIterator output_offset_first,
  OutputValueIterator output_value_first,
  raft::device_span<size_t> count /* size = 1*/,
  std::optional<
    raft::device_span<typename thrust::iterator_traits<OutputOffsetIterator>::value_type const>>
    hypersparse_key_offsets,
  typename thrust::iterator_traits<OutputValueIterator>::value_type invalid_value,
  rmm::cuda_stream_view stream)
{
  using offset_t = std::decay_t<typename thrust::iterator_traits<OutputOffsetIterator>::value_type>;
  using value_t  = std::decay_t<typename thrust::iterator_traits<OutputValueIterator>::value_type>;
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<InputValueIterator>::value_type, value_t>);

  auto input_size = static_cast<size_t>(cuda::std::distance(input_value_first, input_value_last));
  auto output_pair_first = thrust::make_zip_iterator(output_offset_first, output_value_first);

  bool copied = false;
  if (hypersparse_key_offsets) {
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(offset_t{0}),
        cuda::proclaim_return_type<offset_t>(
          [contiguous_size = input_size - hypersparse_key_offsets->size(),
           offsets         = raft::device_span<offset_t const>(
             hypersparse_key_offsets->data(), hypersparse_key_offsets->size())] __device__(auto i) {
            if (i < contiguous_size) {
              return i;
            } else {
              return offsets[i - contiguous_size];
            }
          })),
      input_value_first);
    copy_if_nosync(
      input_pair_first,
      input_pair_first + input_size,
      thrust::make_transform_iterator(
        input_value_first, cuda::proclaim_return_type<bool>([invalid_value] __device__(auto value) {
          return value != invalid_value;
        })),
      output_pair_first,
      count,
      stream);
    copied = true;
  }
  if (!copied) {
    auto input_pair_first =
      thrust::make_zip_iterator(thrust::make_counting_iterator(offset_t{0}), input_value_first);
    copy_if_nosync(
      input_pair_first,
      input_pair_first + input_size,
      thrust::make_transform_iterator(
        input_value_first, cuda::proclaim_return_type<bool>([invalid_value] __device__(auto value) {
          return value != invalid_value;
        })),
      output_pair_first,
      count,
      stream);
  }
}

template <bool update_major,
          typename GraphViewType,
          typename OptionalKeyIterator,  // invalid if void*
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
                                                         GraphViewType::is_multi_gpu, iterator
                                                         otherwise */
          ,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T>
void per_v_transform_reduce_e_edge_partition(
  raft::handle_t const& handle,
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  OptionalKeyIterator edge_partition_key_first,
  OptionalKeyIterator edge_partition_key_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionValueInputWrapper edge_partition_e_value_input,
  cuda::std::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
  ResultValueOutputIteratorOrWrapper output_buffer,
  EdgeOp e_op,
  T major_init,
  T major_identity_element,
  ReduceOp reduce_op,
  PredOp pred_op,
  std::optional<raft::host_span<size_t const>> key_segment_offsets,
  std::optional<raft::host_span<size_t const>> const& edge_partition_stream_pool_indices)
{
  constexpr bool use_input_key = !std::is_same_v<OptionalKeyIterator, void*>;

  using vertex_t = typename GraphViewType::vertex_type;
  using segment_key_iterator_t =
    std::conditional_t<use_input_key,
                       decltype(edge_partition_key_first),
                       decltype(thrust::make_counting_iterator(vertex_t{0}))>;

  size_t stream_pool_size{0};
  if (edge_partition_stream_pool_indices) {
    stream_pool_size = (*edge_partition_stream_pool_indices).size();
  }
  if (key_segment_offsets) {
    static_assert(detail::num_sparse_segments_per_vertex_partition == 3);

    if (edge_partition.dcs_nzd_vertex_count()) {
      auto exec_stream = edge_partition_stream_pool_indices
                           ? handle.get_stream_from_stream_pool(
                               (*edge_partition_stream_pool_indices)[0 % stream_pool_size])
                           : handle.get_stream();

      if constexpr (update_major && !use_input_key) {  // this is necessary as we don't visit
                                                       // every vertex in the hypersparse segment
        thrust::fill(rmm::exec_policy_nosync(exec_stream),
                     output_buffer + (*key_segment_offsets)[3],
                     output_buffer + (*key_segment_offsets)[4],
                     major_init);
      }

      auto segment_size = use_input_key
                            ? ((*key_segment_offsets)[4] - (*key_segment_offsets)[3])
                            : static_cast<size_t>(*(edge_partition.dcs_nzd_vertex_count()));
      if (segment_size > 0) {
        raft::grid_1d_thread_t update_grid(segment_size,
                                           detail::per_v_transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        auto segment_output_buffer = output_buffer;
        if constexpr (update_major) { segment_output_buffer += (*key_segment_offsets)[3]; }
        auto segment_key_first = edge_partition_key_first;
        auto segment_key_last  = edge_partition_key_last;
        if constexpr (use_input_key) {
          segment_key_first += (*key_segment_offsets)[3];
          segment_key_last =
            segment_key_first + ((*key_segment_offsets)[4] - (*key_segment_offsets)[3]);
        } else {
          assert(segment_key_first == nullptr);
          assert(segment_key_last == nullptr);
        }
        detail::per_v_transform_reduce_e_hypersparse<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            segment_key_first,
            segment_key_last,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            segment_output_buffer,
            e_op,
            major_init,
            reduce_op,
            pred_op);
      }
    }
    if ((*key_segment_offsets)[3] - (*key_segment_offsets)[2]) {
      auto exec_stream = edge_partition_stream_pool_indices
                           ? handle.get_stream_from_stream_pool(
                               (*edge_partition_stream_pool_indices)[1 % stream_pool_size])
                           : handle.get_stream();
      raft::grid_1d_thread_t update_grid((*key_segment_offsets)[3] - (*key_segment_offsets)[2],
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
      auto segment_output_buffer = output_buffer;
      if constexpr (update_major) { segment_output_buffer += (*key_segment_offsets)[2]; }
      std::optional<segment_key_iterator_t>
        segment_key_first{};  // std::optional as thrust::transform_iterator's default constructor
                              // is a deleted function, segment_key_first should always have a value
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      *segment_key_first += (*key_segment_offsets)[2];
      detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          *segment_key_first,
          *segment_key_first + ((*key_segment_offsets)[3] - (*key_segment_offsets)[2]),
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          segment_output_buffer,
          e_op,
          major_init,
          reduce_op,
          pred_op);
    }
    if ((*key_segment_offsets)[2] - (*key_segment_offsets)[1] > 0) {
      auto exec_stream = edge_partition_stream_pool_indices
                           ? handle.get_stream_from_stream_pool(
                               (*edge_partition_stream_pool_indices)[2 % stream_pool_size])
                           : handle.get_stream();
      raft::grid_1d_warp_t update_grid((*key_segment_offsets)[2] - (*key_segment_offsets)[1],
                                       detail::per_v_transform_reduce_e_kernel_block_size,
                                       handle.get_device_properties().maxGridSize[0]);
      auto segment_output_buffer = output_buffer;
      if constexpr (update_major) { segment_output_buffer += (*key_segment_offsets)[1]; }
      std::optional<segment_key_iterator_t>
        segment_key_first{};  // std::optional as thrust::transform_iterator's default constructor
                              // is a deleted function, segment_key_first should always have a value
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      *segment_key_first += (*key_segment_offsets)[1];
      detail::per_v_transform_reduce_e_mid_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          *segment_key_first,
          *segment_key_first + ((*key_segment_offsets)[2] - (*key_segment_offsets)[1]),
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          segment_output_buffer,
          e_op,
          major_init,
          major_identity_element,
          reduce_op,
          pred_op);
    }
    if ((*key_segment_offsets)[1] > 0) {
      auto exec_stream = edge_partition_stream_pool_indices
                           ? handle.get_stream_from_stream_pool(
                               (*edge_partition_stream_pool_indices)[3 % stream_pool_size])
                           : handle.get_stream();
      raft::grid_1d_block_t update_grid(
        (*key_segment_offsets)[1],
        std::is_same_v<ReduceOp, reduce_op::any<T>>
          ? detail::per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size
          : detail::per_v_transform_reduce_e_kernel_block_size,
        handle.get_device_properties().maxGridSize[0]);
      std::optional<segment_key_iterator_t>
        segment_key_first{};  // std::optional as thrust::transform_iterator's default constructor
                              // is a deleted function, segment_key_first should always have a value
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      detail::per_v_transform_reduce_e_high_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          *segment_key_first,
          *segment_key_first + (*key_segment_offsets)[1],
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          output_buffer,
          e_op,
          major_init,
          major_identity_element,
          reduce_op,
          pred_op);
    }
  } else {
    auto exec_stream = edge_partition_stream_pool_indices
                         ? handle.get_stream_from_stream_pool(
                             (*edge_partition_stream_pool_indices)[0 % stream_pool_size])
                         : handle.get_stream();

    size_t num_keys{};
    if constexpr (use_input_key) {
      num_keys =
        static_cast<size_t>(cuda::std::distance(edge_partition_key_first, edge_partition_key_last));
    } else {
      num_keys = static_cast<size_t>(edge_partition.major_range_size());
    }

    if (num_keys > size_t{0}) {
      raft::grid_1d_thread_t update_grid(num_keys,
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
      std::optional<segment_key_iterator_t>
        segment_key_first{};  // std::optional as thrust::transform_iterator's default constructor
                              // is a deleted function, segment_key_first should always have a value
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          *segment_key_first,
          *segment_key_first + num_keys,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          output_buffer,
          e_op,
          major_init,
          reduce_op,
          pred_op);
    }
  }
}

template <bool incoming,  // iterate over incoming edges (incoming == true) or outgoing edges
                          // (incoming == false)
          typename GraphViewType,
          typename OptionalKeyIterator,  // invalid if void*
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename PredOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_e(raft::handle_t const& handle,
                              GraphViewType const& graph_view,
                              OptionalKeyIterator sorted_unique_key_first,
                              OptionalKeyIterator sorted_unique_key_last,
                              EdgeSrcValueInputWrapper edge_src_value_input,
                              EdgeDstValueInputWrapper edge_dst_value_input,
                              EdgeValueInputWrapper edge_value_input,
                              EdgeOp e_op,
                              T init,
                              ReduceOp reduce_op,
                              PredOp pred_op,
                              VertexValueOutputIterator vertex_value_output_first)
{
  constexpr bool update_major  = (incoming == GraphViewType::is_storage_transposed);
  constexpr bool use_input_key = !std::is_same_v<OptionalKeyIterator, void*>;
  static_assert(update_major || !use_input_key);
  constexpr bool filter_input_key =
    GraphViewType::is_multi_gpu && use_input_key &&
    std::is_same_v<ReduceOp,
                   reduce_op::any<T>>;  // if GraphViewType::is_multi_gpu && update_major &&
                                        // std::is_same_v<ReduceOp, reduce_op::any<T>>, for any
                                        // vertex in the frontier, we need to visit only local edges
                                        // if we find any valid local edge (FIXME: this is
                                        // applicable even when use_input_key is false).

  static_assert(
    ReduceOp::pure_function &&
    ((reduce_op::has_compatible_raft_comms_op_v<ReduceOp> &&
      reduce_op::has_identity_element_v<ReduceOp>) ||
     (update_major &&
      std::is_same_v<ReduceOp, reduce_op::any<T>>)));  // current restriction, to support general
                                                       // reduction, we may need to take a less
                                                       // efficient code path

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t =
    typename iterator_value_type_or_default_t<OptionalKeyIterator, vertex_t>::value_type;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_iterator, void*>,
    std::conditional_t<
      std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
      detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
      detail::edge_partition_edge_multi_index_property_device_view_t<edge_t, vertex_t>>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  constexpr bool try_bitmap =
    GraphViewType::is_multi_gpu && use_input_key && std::is_same_v<key_t, vertex_t>;

  [[maybe_unused]] constexpr auto max_segments =
    detail::num_sparse_segments_per_vertex_partition + size_t{1};

  // we should consider reducing the life-time of this variable
  // oncermm::rm::pool_memory_resource<rmm::mr::pinned_memory_resource> is updated to honor stream
  // semantics (github.com/rapidsai/rmm/issues/2053)
  rmm::device_uvector<int64_t> h_staging_buffer(0, handle.get_stream());
  {
    size_t staging_buffer_size{};  // should be large enough to cover all update_host &
                                   // update_device calls in this primitive
    if (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      staging_buffer_size        = static_cast<size_t>(minor_comm_size) *
                            std::max(size_t{16}, static_cast<size_t>(minor_comm_size));
    } else {
      staging_buffer_size = size_t{16};
    }
    h_staging_buffer = host_staging_buffer_manager::allocate_staging_buffer<int64_t>(
      staging_buffer_size, handle.get_stream());
  }

  // 1. drop zero degree keys & compute key_segment_offsets

  auto const& local_vertex_partition_segment_offsets =
    graph_view.local_vertex_partition_segment_offsets();

  std::conditional_t<use_input_key, std::optional<std::vector<size_t>>, std::byte /* dummy */>
    key_segment_offsets{};
  auto sorted_unique_nzd_key_last = sorted_unique_key_last;
  if constexpr (use_input_key) {
    if (local_vertex_partition_segment_offsets) {
      key_segment_offsets = compute_key_segment_offsets(
        sorted_unique_key_first,
        sorted_unique_nzd_key_last,
        raft::host_span<vertex_t const>(local_vertex_partition_segment_offsets->data(),
                                        local_vertex_partition_segment_offsets->size()),
        graph_view.local_vertex_partition_range_first(),
        handle.get_stream());
      key_segment_offsets->back() = *(key_segment_offsets->rbegin() + 1);
      sorted_unique_nzd_key_last  = sorted_unique_key_first + key_segment_offsets->back();
    }
  }

  // 2. initialize vertex value output buffer

  if constexpr (update_major) {  // no vertices in the zero degree segment are visited (otherwise,
                                 // no need to initialize)
    if constexpr (use_input_key) {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first +
                     cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
                   vertex_value_output_first +
                     cuda::std::distance(sorted_unique_key_first, sorted_unique_key_last),
                   init);
    } else {
      if (local_vertex_partition_segment_offsets) {
        thrust::fill(
          handle.get_thrust_policy(),
          vertex_value_output_first + *(local_vertex_partition_segment_offsets->rbegin() + 1),
          vertex_value_output_first + *(local_vertex_partition_segment_offsets->rbegin()),
          init);
      }
    }
  } else {
    if constexpr (GraphViewType::is_multi_gpu) {
      /* no need to initialize (we use minor_tmp_buffer) */
    } else {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
                   init);
    }
  }

  // 3. filter input keys & update key_segment_offsets

  auto edge_mask_view = graph_view.edge_mask_view();

  auto tmp_key_buffer =
    allocate_optional_dataframe_buffer<std::conditional_t<filter_input_key, key_t, void>>(
      0, handle.get_stream());
  auto tmp_output_indices =
    allocate_optional_dataframe_buffer<std::conditional_t<filter_input_key, size_t, void>>(
      0, handle.get_stream());
  std::conditional_t<filter_input_key,
                     thrust::permutation_iterator<VertexValueOutputIterator, size_t const*>,
                     VertexValueOutputIterator>
    tmp_vertex_value_output_first{};
  if constexpr (filter_input_key) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();

    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(static_cast<size_t>(minor_comm_rank)));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, static_cast<size_t>(minor_comm_rank))
        : cuda::std::nullopt;

    std::optional<std::vector<size_t>> edge_partition_stream_pool_indices{std::nullopt};
    if (local_vertex_partition_segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      edge_partition_stream_pool_indices = std::vector<size_t>(max_segments);
      std::iota(edge_partition_stream_pool_indices->begin(),
                edge_partition_stream_pool_indices->end(),
                size_t{0});
    }

    if (edge_partition_stream_pool_indices) { handle.sync_stream(); }

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(
        edge_dst_value_input, static_cast<size_t>(minor_comm_rank));
    } else {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(
        edge_src_value_input, static_cast<size_t>(minor_comm_rank));
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }
    auto edge_partition_e_value_input =
      edge_partition_e_input_device_view_t(edge_value_input, static_cast<size_t>(minor_comm_rank));

    per_v_transform_reduce_e_edge_partition<update_major, GraphViewType>(
      handle,
      edge_partition,
      sorted_unique_key_first,
      sorted_unique_nzd_key_last,
      edge_partition_src_value_input,
      edge_partition_dst_value_input,
      edge_partition_e_value_input,
      edge_partition_e_mask,
      vertex_value_output_first,
      e_op,
      init,
      init,
      reduce_op,
      pred_op,
      key_segment_offsets ? std::make_optional<raft::host_span<size_t const>>(
                              key_segment_offsets->data(), key_segment_offsets->size())
                          : std::nullopt,
      edge_partition_stream_pool_indices
        ? std::make_optional<raft::host_span<size_t const>>(
            edge_partition_stream_pool_indices->data(), edge_partition_stream_pool_indices->size())
        : std::nullopt);
    if (edge_partition_stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }

    auto input_size = cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last);
    resize_optional_dataframe_buffer<key_t>(tmp_key_buffer, input_size, handle.get_stream());
    resize_optional_dataframe_buffer<size_t>(tmp_output_indices, input_size, handle.get_stream());
    auto input_first =
      thrust::make_zip_iterator(sorted_unique_key_first, thrust::make_counting_iterator(size_t{0}));
    auto output_first =
      thrust::make_zip_iterator(get_optional_dataframe_buffer_begin<key_t>(tmp_key_buffer),
                                get_optional_dataframe_buffer_begin<size_t>(tmp_output_indices));
    auto num_tmp_keys = cuda::std::distance(
      output_first,
      thrust::copy_if(
        handle.get_thrust_policy(),
        input_first,
        input_first + cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
        vertex_value_output_first,
        output_first,
        is_equal_t<T>{init}));
    resize_optional_dataframe_buffer<key_t>(tmp_key_buffer, num_tmp_keys, handle.get_stream());
    resize_optional_dataframe_buffer<size_t>(tmp_output_indices, num_tmp_keys, handle.get_stream());
    // skip shrink_to_fit to cut execution time

    sorted_unique_key_first       = get_optional_dataframe_buffer_begin<key_t>(tmp_key_buffer);
    sorted_unique_nzd_key_last    = get_optional_dataframe_buffer_end<key_t>(tmp_key_buffer);
    tmp_vertex_value_output_first = thrust::make_permutation_iterator(
      vertex_value_output_first, get_optional_dataframe_buffer_begin<size_t>(tmp_output_indices));

    if (key_segment_offsets) {
      key_segment_offsets = compute_key_segment_offsets(
        sorted_unique_key_first,
        sorted_unique_nzd_key_last,
        raft::host_span<vertex_t const>(local_vertex_partition_segment_offsets->data(),
                                        local_vertex_partition_segment_offsets->size()),
        edge_partition.major_range_first(),
        handle.get_stream());
      assert(key_segment_offsets->back() == *(key_segment_offsets->rbegin() + 1));
      assert(sorted_unique_nzd_key_last == sorted_unique_key_first + key_segment_offsets->back());
    }
  } else {
    tmp_vertex_value_output_first = vertex_value_output_first;
  }

  /* 4. compute subgroup_size (used to compute priority in device_gatherv) */

  [[maybe_unused]] std::conditional_t<GraphViewType::is_multi_gpu && update_major &&
                                        std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                      int,
                                      std::byte /* dummy */> subgroup_size{};
  if constexpr (GraphViewType::is_multi_gpu && update_major &&
                std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    int num_gpus_per_domain{};  // domain: a group of GPUs that can communicate fast (e.g. NVLink
                                // domain)
#if 1  // SK: we should get this from NCCL (once NCCL is updated to provide this information)
    num_gpus_per_domain = 64;
#else
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_domain));
#endif
    num_gpus_per_domain = std::min(num_gpus_per_domain, comm_size);
    if (comm_size == num_gpus_per_domain) {
      subgroup_size = minor_comm_size;
    } else {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      subgroup_size              = partition_manager::map_major_comm_to_gpu_row_comm
                                     ? std::max(num_gpus_per_domain / major_comm_size, int{1})
                                     : std::min(minor_comm_size, num_gpus_per_domain);
    }
  }

  // 5. collect max_tmp_buffer_size, approx_tmp_buffer_size_per_loop, local_key_list_sizes,
  // local_v_list_range_firsts, local_v_list_range_lasts,  key_segment_offset_vectors

  std::conditional_t<GraphViewType::is_multi_gpu, std::vector<size_t>, std::byte /* dummy */>
    max_tmp_buffer_sizes{};  // to decide on concurrency vs memory footprint trade-off
  std::conditional_t<GraphViewType::is_multi_gpu, std::vector<size_t>, std::byte /* dummy */>
    tmp_buffer_size_per_loop_approximations{};  // to decide on concurrency vs memory footprint
                                                // trade-off
  std::conditional_t<use_input_key, std::vector<size_t>, std::byte /* dummy */>
    local_key_list_sizes{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_v_list_range_firsts{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_v_list_range_lasts{};
  std::conditional_t<use_input_key,
                     std::optional<std::vector<std::vector<size_t>>>,
                     std::byte /* dummy */>
    key_segment_offset_vectors{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto max_tmp_buffer_size =
      static_cast<size_t>(static_cast<double>(handle.get_device_properties().totalGlobalMem) * 0.2);
    size_t approx_tmp_buffer_size_per_loop{0};
    if constexpr (update_major) {
      size_t key_size{0};
      if constexpr (use_input_key) {
        if constexpr (std::is_arithmetic_v<key_t>) {
          key_size = sizeof(key_t);
        } else {
          key_size = sum_thrust_tuple_element_sizes<key_t>();
        }
      }
      size_t value_size{0};
      if constexpr (std::is_arithmetic_v<key_t>) {
        value_size = sizeof(T);
      } else {
        value_size = sum_thrust_tuple_element_sizes<T>();
      }

      size_t major_range_size{};
      if constexpr (use_input_key) {
        major_range_size = static_cast<size_t>(
          cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last));
        ;
      } else {
        major_range_size = graph_view.local_vertex_partition_range_size();
      }
      size_t size_per_key{};
      if constexpr (filter_input_key) {
        size_per_key =
          key_size +
          value_size / 2;  // to reflect that many keys will be filtered out, note that this is a
                           // simple approximation, memory requirement in this case is much more
                           // complex as we store additional temporary variables

      } else {
        size_per_key = key_size + value_size;
      }
      approx_tmp_buffer_size_per_loop = major_range_size * size_per_key;
    }

    size_t num_scalars = 2;  // max_tmp_buffer_size, approx_tmp_buffer_size_per_loop
    size_t num_scalars_less_key_segment_offsets = num_scalars;
    if constexpr (use_input_key) {
      num_scalars += 1;  // local_key_list_size
      if constexpr (try_bitmap) {
        num_scalars += 2;  // local_key_list_range_first, local_key_list_range_last
      }
      num_scalars_less_key_segment_offsets = num_scalars;
      if (key_segment_offsets) { num_scalars += key_segment_offsets->size(); }
    }

    std::vector<size_t> h_aggregate_tmps(minor_comm_size * num_scalars);
    h_aggregate_tmps[minor_comm_rank * num_scalars]     = max_tmp_buffer_size;
    h_aggregate_tmps[minor_comm_rank * num_scalars + 1] = approx_tmp_buffer_size_per_loop;
    if constexpr (use_input_key) {
      auto v_list_size = static_cast<size_t>(
        cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last));
      h_aggregate_tmps[minor_comm_rank * num_scalars + 2] = v_list_size;
      if constexpr (try_bitmap) {
        vertex_t range_first = graph_view.local_vertex_partition_range_first();
        auto range_last      = range_first;
        if (v_list_size > 0) {
          auto h_staging_buffer_ptr = reinterpret_cast<vertex_t*>(h_staging_buffer.data());
          assert(h_staging_buffer.size() >= size_t{2});
          if constexpr (std::is_pointer_v<std::decay<OptionalKeyIterator>>) {
            raft::update_host(
              h_staging_buffer_ptr, sorted_unique_key_first, size_t{1}, handle.get_stream());
            raft::update_host(h_staging_buffer_ptr + 1,
                              sorted_unique_key_first + (v_list_size - 1),
                              size_t{1},
                              handle.get_stream());
          } else {
            rmm::device_uvector<vertex_t> tmps(2, handle.get_stream());
            thrust::tabulate(handle.get_thrust_policy(),
                             tmps.begin(),
                             tmps.end(),
                             cuda::proclaim_return_type<vertex_t>(
                               [sorted_unique_key_first, v_list_size] __device__(size_t i) {
                                 if (i == 0) {
                                   return *sorted_unique_key_first;
                                 } else {
                                   assert(i == 1);
                                   return *(sorted_unique_key_first + (v_list_size - 1));
                                 }
                               }));
            raft::update_host(h_staging_buffer_ptr, tmps.data(), size_t{2}, handle.get_stream());
          }
          handle.sync_stream();
          range_first = h_staging_buffer_ptr[0];
          range_last  = h_staging_buffer_ptr[1] + 1;
        }
        h_aggregate_tmps[minor_comm_rank * num_scalars + 3] = static_cast<size_t>(range_first);
        h_aggregate_tmps[minor_comm_rank * num_scalars + 4] = static_cast<size_t>(range_last);
      }
      if (key_segment_offsets) {
        std::copy(key_segment_offsets->begin(),
                  key_segment_offsets->end(),
                  h_aggregate_tmps.begin() +
                    (minor_comm_rank * num_scalars + num_scalars_less_key_segment_offsets));
      }
    }

    if (minor_comm_size > 1) {
      minor_comm.host_allgather(h_aggregate_tmps.data(), h_aggregate_tmps.data(), num_scalars);
    }
    max_tmp_buffer_sizes                    = std::vector<size_t>(minor_comm_size);
    tmp_buffer_size_per_loop_approximations = std::vector<size_t>(minor_comm_size);
    if constexpr (use_input_key) {
      local_key_list_sizes = std::vector<size_t>(minor_comm_size);
      if constexpr (try_bitmap) {
        local_v_list_range_firsts = std::vector<vertex_t>(minor_comm_size);
        local_v_list_range_lasts  = std::vector<vertex_t>(minor_comm_size);
      }
      if (key_segment_offsets) {
        key_segment_offset_vectors = std::vector<std::vector<size_t>>{};
        (*key_segment_offset_vectors).reserve(minor_comm_size);
      }
    }
    for (int i = 0; i < minor_comm_size; ++i) {
      max_tmp_buffer_sizes[i]                    = h_aggregate_tmps[i * num_scalars];
      tmp_buffer_size_per_loop_approximations[i] = h_aggregate_tmps[i * num_scalars + 1];
      if constexpr (use_input_key) {
        local_key_list_sizes[i] = h_aggregate_tmps[i * num_scalars + 2];
        if constexpr (try_bitmap) {
          local_v_list_range_firsts[i] =
            static_cast<vertex_t>(h_aggregate_tmps[i * num_scalars + 3]);
          local_v_list_range_lasts[i] =
            static_cast<vertex_t>(h_aggregate_tmps[i * num_scalars + 4]);
        }
        if (key_segment_offsets) {
          key_segment_offset_vectors->emplace_back(
            h_aggregate_tmps.begin() + i * num_scalars + num_scalars_less_key_segment_offsets,
            h_aggregate_tmps.begin() + i * num_scalars + num_scalars_less_key_segment_offsets +
              key_segment_offsets->size());
        }
      }
    }
  } else {
    if constexpr (use_input_key) {
      local_key_list_sizes = std::vector<size_t>{static_cast<size_t>(
        cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last))};
      if (key_segment_offsets) {
        key_segment_offset_vectors       = std::vector<std::vector<size_t>>(1);
        (*key_segment_offset_vectors)[0] = *key_segment_offsets;
      }
    }
  }

  if constexpr (use_input_key) {
    if (std::reduce(local_key_list_sizes.begin(), local_key_list_sizes.end()) ==
        0) {  // nothing to do
      return;
    }
  }

  // 6. compute optional bitmap info & compressed vertex list

  bool v_compressible{false};
  std::
    conditional_t<try_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      v_list_bitmap{};
  std::
    conditional_t<try_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      compressed_v_list{};
  if constexpr (try_bitmap) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    if (minor_comm_size > 1) {
      auto const minor_comm_rank = minor_comm.get_rank();

      if constexpr (sizeof(vertex_t) == 8) {
        vertex_t max_local_v_list_range_size{0};
        for (int i = 0; i < minor_comm_size; ++i) {
          auto range_size             = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
          max_local_v_list_range_size = std::max(range_size, max_local_v_list_range_size);
        }
        if (max_local_v_list_range_size <=
            std::numeric_limits<uint32_t>::max()) {  // broadcast 32bit offset values instead of 64
                                                     // bit vertex IDs
          v_compressible = true;
        }
      }

      double avg_fill_ratio{0.0};
      for (int i = 0; i < minor_comm_size; ++i) {
        auto num_keys   = static_cast<double>(local_key_list_sizes[i]);
        auto range_size = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
        avg_fill_ratio +=
          (range_size > 0) ? (num_keys / static_cast<double>(range_size)) : double{0.0};
      }
      avg_fill_ratio /= static_cast<double>(minor_comm_size);
      double threshold_ratio =
        2.0 /* tuning parameter (consider that we need to reprodce vertex list from bitmap)*/ /
        static_cast<double>((v_compressible ? sizeof(uint32_t) : sizeof(vertex_t)) * 8);
      auto avg_key_list_size =
        std::reduce(local_key_list_sizes.begin(), local_key_list_sizes.end()) /
        static_cast<vertex_t>(minor_comm_size);

      if ((avg_fill_ratio > threshold_ratio) &&
          (static_cast<size_t>(avg_key_list_size) >
           packed_bools_per_word() *
             32 /* tuning parameter, to considerr additional kernel launch overhead */)) {
        v_list_bitmap = compute_vertex_list_bitmap_info(sorted_unique_key_first,
                                                        sorted_unique_nzd_key_last,
                                                        local_v_list_range_firsts[minor_comm_rank],
                                                        local_v_list_range_lasts[minor_comm_rank],
                                                        handle.get_stream());
      } else if (v_compressible) {
        rmm::device_uvector<uint32_t> tmps(local_key_list_sizes[minor_comm_rank],
                                           handle.get_stream());
        thrust::transform(handle.get_thrust_policy(),
                          sorted_unique_key_first,
                          sorted_unique_nzd_key_last,
                          tmps.begin(),
                          cuda::proclaim_return_type<uint32_t>(
                            [range_first = local_v_list_range_firsts[minor_comm_rank]] __device__(
                              auto v) { return static_cast<uint32_t>(v - range_first); }));
        compressed_v_list = std::move(tmps);
      }

      if constexpr (filter_input_key) {
        if (v_list_bitmap || compressed_v_list) {
          resize_optional_dataframe_buffer<key_t>(tmp_key_buffer, 0, handle.get_stream());
          shrink_to_fit_optional_dataframe_buffer<key_t>(tmp_key_buffer, handle.get_stream());
          sorted_unique_key_first    = OptionalKeyIterator{};
          sorted_unique_nzd_key_last = sorted_unique_key_first;
        }
      }
    }
  }

  bool uint32_key_output_offset = false;
  if constexpr (GraphViewType::is_multi_gpu && update_major &&
                std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    size_t max_key_offset_size{0};
    if constexpr (filter_input_key) {
      max_key_offset_size = std::reduce(
        local_key_list_sizes.begin(), local_key_list_sizes.end(), size_t{0}, [](auto l, auto r) {
          return std::max(l, r);
        });
    } else {
      static_assert(!use_input_key);
      for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
        auto edge_partition =
          edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
            graph_view.local_edge_partition_view(i));
        auto const& segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

        auto output_range_size =
          segment_offsets ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                          : edge_partition.major_range_size();

        max_key_offset_size = std::max(static_cast<size_t>(output_range_size), max_key_offset_size);
      }
    }
    uint32_key_output_offset =
      (max_key_offset_size <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
  }

  size_t min_element_size{cache_line_size};
  if (uint32_key_output_offset) {
    min_element_size = std::min(sizeof(uint32_t), min_element_size);
  } else {
    min_element_size = std::min(sizeof(size_t), min_element_size);
  }
  if constexpr (std::is_arithmetic_v<T>) {
    min_element_size = std::min(sizeof(T), min_element_size);
  } else {
    min_element_size = std::min(min_thrust_tuple_element_sizes<T>(), min_element_size);
  }
  assert((cache_line_size % min_element_size) == 0);
  auto alignment = cache_line_size / min_element_size;

  // 7. set-up stream pool

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    auto max_tmp_buffer_size =
      std::reduce(max_tmp_buffer_sizes.begin(), max_tmp_buffer_sizes.end()) /
      static_cast<size_t>(minor_comm_size);
    auto approx_tmp_buffer_size_per_loop =
      std::reduce(tmp_buffer_size_per_loop_approximations.begin(),
                  tmp_buffer_size_per_loop_approximations.end()) /
      static_cast<size_t>(minor_comm_size);
    size_t num_streams_per_loop{1};
    if (local_vertex_partition_segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      num_streams_per_loop = std::max(
        std::min(size_t{8} / graph_view.number_of_local_edge_partitions(), max_segments),
        size_t{
          1});  // Note that "CUDA_DEVICE_MAX_CONNECTIONS (default: 8, can be set to [1, 32])" sets
                // the number of queues, if the total number of streams exceeds this number, jobs on
                // different streams can be sent to one queue leading to false dependency. Setting
                // num_concurrent_loops above the number of queues has some benefits in NCCL
                // communications but creating too many streams just for compute may not help.
    }
    stream_pool_indices = init_stream_pool_indices(max_tmp_buffer_size,
                                                   approx_tmp_buffer_size_per_loop,
                                                   graph_view.number_of_local_edge_partitions(),
                                                   num_streams_per_loop,
                                                   handle.get_stream_pool_size());
    if ((*stream_pool_indices).size() <= 1) { stream_pool_indices = std::nullopt; }
  }

  size_t num_concurrent_loops{1};
  std::optional<std::vector<size_t>> loop_stream_pool_indices{
    std::nullopt};  // first num_concurrent_loops streams from stream_pool_indices
  if (stream_pool_indices) {
    num_concurrent_loops =
      std::min(graph_view.number_of_local_edge_partitions(), (*stream_pool_indices).size());
    loop_stream_pool_indices = std::vector<size_t>(num_concurrent_loops);
    std::iota((*loop_stream_pool_indices).begin(), (*loop_stream_pool_indices).end(), size_t{0});
  }

  // 8. set-up temporary buffers

  using minor_tmp_buffer_type = detail::edge_minor_property_t<vertex_t, T>;
  [[maybe_unused]] std::unique_ptr<minor_tmp_buffer_type> minor_tmp_buffer{};
  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    minor_tmp_buffer  = std::make_unique<minor_tmp_buffer_type>(handle, graph_view);
    auto minor_init   = init;
    auto mutable_view = minor_tmp_buffer->mutable_view();
    if (mutable_view.minor_keys()) {  // defer applying the initial value to the end as
                                      // minor_tmp_buffer ma not
                                      // store values for the entire minor rangey
      minor_init = ReduceOp::identity_element;
    } else {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_rank = major_comm.get_rank();
      minor_init                 = (major_comm_rank == 0) ? init : ReduceOp::identity_element;
    }
    fill_edge_minor_property(handle, graph_view, mutable_view, minor_init);
  }

  using edge_partition_minor_output_device_view_t =
    std::conditional_t<GraphViewType::is_multi_gpu && !update_major,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename minor_tmp_buffer_type::value_iterator,
                         T>,
                       void /* dummy */>;

  auto counters = allocate_optional_dataframe_buffer<
    std::conditional_t<GraphViewType::is_multi_gpu && update_major, size_t, void>>(
    num_concurrent_loops, handle.get_stream());

  if constexpr (!GraphViewType::is_multi_gpu || !use_input_key) {
    if (loop_stream_pool_indices) { handle.sync_stream(); }
  }

  // 9. process local edge partitions

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i += num_concurrent_loops) {
    auto loop_count =
      std::min(num_concurrent_loops, graph_view.number_of_local_edge_partitions() - i);

    std::conditional_t<
      GraphViewType::is_multi_gpu && use_input_key,
      std::conditional_t<
        try_bitmap,
        std::vector<std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<vertex_t>>>,
        std::vector<dataframe_buffer_type_t<key_t>>>,
      std::byte /* dummy */>
      edge_partition_key_buffers{};
    std::conditional_t<filter_input_key,
                       std::optional<std::vector<
                         std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>>>,
                       std::byte /* dummy */>
      edge_partition_hypersparse_key_offset_vectors{};  // drop zero local degree keys in the
                                                        // hypersparse region (keep output offset
                                                        // values of the keys in the hypersparse
                                                        // region that have non-zero local degrees)
    std::conditional_t<use_input_key, std::vector<bool>, std::byte /* dummy */> nonzero_key_lists{};
    if constexpr (use_input_key) { nonzero_key_lists = std::vector<bool>(loop_count, true); }
    std::vector<bool> process_local_edges(loop_count, true);

    if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      auto const minor_comm_rank = minor_comm.get_rank();

      edge_partition_key_buffers.reserve(loop_count);

      std::conditional_t<try_bitmap,
                         std::optional<std::vector<rmm::device_uvector<uint32_t>>>,
                         std::byte /* dummy */>
        edge_partition_bitmap_buffers{std::nullopt};
      if constexpr (try_bitmap) {
        if (v_list_bitmap) {
          edge_partition_bitmap_buffers = std::vector<rmm::device_uvector<uint32_t>>{};
          edge_partition_bitmap_buffers->reserve(loop_count);
        }
      }

      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx = i + j;

        if constexpr (use_input_key) {
          if (local_key_list_sizes[partition_idx] == 0) { nonzero_key_lists[j] = false; }
          if constexpr (filter_input_key) {
            if (static_cast<int>(partition_idx) == minor_comm_rank) {
              process_local_edges[j] = false;
            }
          }
        }

        bool use_bitmap_buffer = false;
        if constexpr (try_bitmap) {
          if (edge_partition_bitmap_buffers) {
            use_bitmap_buffer = true;
            edge_partition_bitmap_buffers->emplace_back(
              packed_bool_size(local_v_list_range_lasts[partition_idx] -
                               local_v_list_range_firsts[partition_idx]),
              handle.get_stream());
          }
        }
        if (!use_bitmap_buffer) {
          bool allocated{false};
          if constexpr (try_bitmap) {
            if (v_compressible) {
              edge_partition_key_buffers.push_back(rmm::device_uvector<uint32_t>(
                local_key_list_sizes[partition_idx], handle.get_stream()));
              allocated = true;
            }
          }
          if (!allocated) {
            edge_partition_key_buffers.push_back(allocate_dataframe_buffer<key_t>(
              local_key_list_sizes[partition_idx], handle.get_stream()));
          }
        }
      }

      device_group_start(minor_comm);
      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx = i + j;

        if (!nonzero_key_lists[j]) { continue; }

        if constexpr (try_bitmap) {
          if (v_list_bitmap) {
            device_bcast(minor_comm,
                         v_list_bitmap->data(),
                         get_dataframe_buffer_begin((*edge_partition_bitmap_buffers)[j]),
                         size_dataframe_buffer((*edge_partition_bitmap_buffers)[j]),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          } else if (compressed_v_list) {
            device_bcast(minor_comm,
                         compressed_v_list->data(),
                         get_dataframe_buffer_begin(std::get<0>(edge_partition_key_buffers[j])),
                         local_key_list_sizes[partition_idx],
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          } else {
            device_bcast(minor_comm,
                         sorted_unique_key_first,
                         get_dataframe_buffer_begin(std::get<1>(edge_partition_key_buffers[j])),
                         local_key_list_sizes[partition_idx],
                         static_cast<int>(partition_idx),
                         handle.get_stream());
          }
        } else {
          device_bcast(minor_comm,
                       sorted_unique_key_first,
                       get_dataframe_buffer_begin(edge_partition_key_buffers[j]),
                       local_key_list_sizes[partition_idx],
                       static_cast<int>(partition_idx),
                       handle.get_stream());
        }
      }
      device_group_end(minor_comm);
      if (loop_stream_pool_indices) { handle.sync_stream(); }

      if constexpr (try_bitmap) {
        if (edge_partition_bitmap_buffers) {
          // copy keys from temporary bitmap buffers to key buffers (copy only the sparse segments
          // if filter_input_key is true)

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream =
              loop_stream_pool_indices
                ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                : handle.get_stream();

            std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<vertex_t>> keys =
              rmm::device_uvector<uint32_t>(0, loop_stream);
            if (v_compressible) {
              std::get<0>(keys).resize(
                process_local_edges[j] ? local_key_list_sizes[partition_idx] : size_t{0},
                loop_stream);
            } else {
              keys = rmm::device_uvector<vertex_t>(
                process_local_edges[j] ? local_key_list_sizes[partition_idx] : size_t{0},
                loop_stream);
            }

            auto& rx_bitmap = (*edge_partition_bitmap_buffers)[j];
            if (nonzero_key_lists[j] && process_local_edges[j]) {
              auto range_first = local_v_list_range_firsts[partition_idx];
              auto range_last  = local_v_list_range_lasts[partition_idx];
              if constexpr (filter_input_key) {
                if (graph_view.use_dcs()) {  // skip copying the hypersparse segment (we will filter
                                             // out the vertices in the hypersparse region before
                                             // copying if the local degree is 0)
                  auto edge_partition =
                    edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                      graph_view.local_edge_partition_view(partition_idx));
                  range_last = std::min(range_last, *(edge_partition.major_hypersparse_first()));
                }
              }
              if (range_first < range_last) {
                if (keys.index() == 0) {
                  retrieve_vertex_list_from_bitmap(
                    raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                    get_dataframe_buffer_begin(std::get<0>(keys)),
                    raft::device_span<size_t>(
                      counters.data() + j,
                      size_t{1}),  // dummy, we already know the counts (i.e.
                                   // (*key_segment_offset_vectors)[partition_idx][3])
                    uint32_t{0},
                    static_cast<uint32_t>(range_last - range_first),
                    loop_stream);
                } else {
                  retrieve_vertex_list_from_bitmap(
                    raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                    get_dataframe_buffer_begin(std::get<1>(keys)),
                    raft::device_span<size_t>(
                      counters.data() + j,
                      size_t{1}),  // dummy, we already know the counts (i.e.
                                   // (*key_segment_offset_vectors)[partition_idx][3])
                    range_first,
                    range_last,
                    loop_stream);
                }
              }
            } else {
              rx_bitmap.resize(0, loop_stream);
              rx_bitmap.shrink_to_fit(loop_stream);
            }
            edge_partition_key_buffers.push_back(std::move(keys));
          }
        }
      }

      if constexpr (filter_input_key) {
        if (graph_view.use_dcs()) {
          edge_partition_hypersparse_key_offset_vectors =
            std::vector<std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>>{};
          edge_partition_hypersparse_key_offset_vectors->reserve(loop_count);

          std::conditional_t<GraphViewType::is_multi_gpu && use_input_key,
                             std::optional<std::conditional_t<
                               try_bitmap,
                               std::vector<std::variant<rmm::device_uvector<uint32_t>,
                                                        rmm::device_uvector<vertex_t>>>,
                               std::vector<dataframe_buffer_type_t<key_t>>>>,
                             std::byte /* dummy */>
            edge_partition_new_key_buffers{};
          bool allocate_new_key_buffer{true};
          if constexpr (try_bitmap) {
            if (edge_partition_bitmap_buffers) { allocate_new_key_buffer = false; }
          }
          if (allocate_new_key_buffer) {  // allocate new key buffers and copy the sparse segment
                                          // keys to the new key buffers (the cost of this copy is
                                          // mostly moderate as key_segment_offsets[4] -
                                          // key_segment_offsets[3] is mostly significantly larger
                                          // than  key_segment_offsets[3] once we filter out the
                                          // vertices that have valid local edges)
            if constexpr (try_bitmap) {
              edge_partition_new_key_buffers = std::vector<
                std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<vertex_t>>>{};
            } else {
              edge_partition_new_key_buffers = std::vector<dataframe_buffer_type_t<key_t>>{};
            }
            edge_partition_new_key_buffers->reserve(loop_count);

            for (size_t j = 0; j < loop_count; ++j) {
              auto partition_idx = i + j;
              auto loop_stream =
                loop_stream_pool_indices
                  ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                  : handle.get_stream();

              auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

              if constexpr (try_bitmap) {
                if (v_compressible) {
                  auto new_key_buffer = rmm::device_uvector<uint32_t>(
                    process_local_edges[j] ? local_key_list_sizes[partition_idx] : size_t{0},
                    loop_stream);
                  if (nonzero_key_lists[j] && process_local_edges[j]) {
                    thrust::copy(
                      rmm::exec_policy_nosync(loop_stream),
                      get_dataframe_buffer_begin(std::get<0>(edge_partition_key_buffers[j])),
                      get_dataframe_buffer_begin(std::get<0>(edge_partition_key_buffers[j])) +
                        key_segment_offsets[3],
                      get_dataframe_buffer_begin(new_key_buffer));
                  } else {
                    std::get<0>(edge_partition_key_buffers[j]).resize(0, loop_stream);
                    std::get<0>(edge_partition_key_buffers[j]).shrink_to_fit(loop_stream);
                  }
                  edge_partition_new_key_buffers->push_back(std::move(new_key_buffer));
                } else {
                  auto new_key_buffer = rmm::device_uvector<vertex_t>(
                    process_local_edges[j] ? local_key_list_sizes[partition_idx] : size_t{0},
                    loop_stream);
                  if (nonzero_key_lists[j] && process_local_edges[j]) {
                    thrust::copy(
                      rmm::exec_policy_nosync(loop_stream),
                      get_dataframe_buffer_begin(std::get<1>(edge_partition_key_buffers[j])),
                      get_dataframe_buffer_begin(std::get<1>(edge_partition_key_buffers[j])) +
                        key_segment_offsets[3],
                      get_dataframe_buffer_begin(new_key_buffer));
                  } else {
                    std::get<1>(edge_partition_key_buffers[j]).resize(0, loop_stream);
                    std::get<1>(edge_partition_key_buffers[j]).shrink_to_fit(loop_stream);
                  }
                  edge_partition_new_key_buffers->push_back(std::move(new_key_buffer));
                }
              } else {
                auto new_key_buffer = allocate_dataframe_buffer<key_t>(
                  process_local_edges[j] ? local_key_list_sizes[partition_idx] : size_t{0},
                  loop_stream);
                if (nonzero_key_lists[j] && process_local_edges[j]) {
                  thrust::copy(rmm::exec_policy_nosync(loop_stream),
                               get_dataframe_buffer_begin(edge_partition_key_buffers[j]),
                               get_dataframe_buffer_begin(edge_partition_key_buffers[j]) +
                                 key_segment_offsets[3],
                               get_dataframe_buffer_begin(new_key_buffer));
                } else {
                  edge_partition_key_buffers[j].resize(0, loop_stream);
                  edge_partition_key_buffers[j].shrink_to_fit(loop_stream);
                }
                edge_partition_new_key_buffers->push_back(std::move(new_key_buffer));
              }
            }
          }

          if constexpr (try_bitmap) {  // if we are using a bitmap buffer (filter out the vertices
                                       // in the hypersparse region if local degree is 0)
            if (v_list_bitmap) {
              std::vector<rmm::device_uvector<vertex_t>>
                input_count_offset_vectors{};  // count per 32-bit word
              input_count_offset_vectors.reserve(loop_count);

              std::vector<rmm::device_uvector<uint32_t>> filtered_bitmap_vectors{};
              std::vector<rmm::device_uvector<vertex_t>>
                output_count_offset_vectors{};  // count per 32-bit word
              filtered_bitmap_vectors.reserve(loop_count);
              output_count_offset_vectors.reserve(loop_count);

              std::vector<vertex_t> range_offset_firsts(loop_count, 0);
              std::vector<vertex_t> range_offset_lasts(loop_count, 0);

              for (size_t j = 0; j < loop_count; ++j) {
                auto partition_idx = i + j;
                auto loop_stream =
                  loop_stream_pool_indices
                    ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                    : handle.get_stream();

                rmm::device_uvector<vertex_t> input_count_offsets(0, loop_stream);
                if (nonzero_key_lists[j] && process_local_edges[j]) {
                  auto edge_partition =
                    edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                      graph_view.local_edge_partition_view(partition_idx));
                  auto const& segment_offsets =
                    graph_view.local_edge_partition_segment_offsets(partition_idx);

                  auto range_offset_first =
                    std::min((edge_partition.major_range_first() + (*segment_offsets)[3] >
                              local_v_list_range_firsts[partition_idx])
                               ? ((edge_partition.major_range_first() + (*segment_offsets)[3]) -
                                  local_v_list_range_firsts[partition_idx])
                               : vertex_t{0},
                             local_v_list_range_lasts[partition_idx] -
                               local_v_list_range_firsts[partition_idx]);
                  auto range_offset_last =
                    std::min(((edge_partition.major_range_first() + (*segment_offsets)[4]) >
                              local_v_list_range_firsts[partition_idx])
                               ? ((edge_partition.major_range_first() + (*segment_offsets)[4]) -
                                  local_v_list_range_firsts[partition_idx])
                               : vertex_t{0},
                             local_v_list_range_lasts[partition_idx] -
                               local_v_list_range_firsts[partition_idx]);
                  if (range_offset_first < range_offset_last) {
                    auto const& rx_bitmap  = (*edge_partition_bitmap_buffers)[j];
                    auto input_count_first = thrust::make_transform_iterator(
                      thrust::make_counting_iterator(packed_bool_offset(range_offset_first)),
                      cuda::proclaim_return_type<vertex_t>(
                        [range_bitmap =
                           raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                         range_offset_first] __device__(size_t i) {
                          auto word = range_bitmap[i];
                          if (i == packed_bool_offset(range_offset_first)) {
                            word &= ~packed_bool_partial_mask(
                              range_offset_first %
                              packed_bools_per_word());  // clear the bits in the sparse region
                          }
                          return static_cast<vertex_t>(__popc(word));
                        }));
                    input_count_offsets.resize(
                      (rx_bitmap.size() - packed_bool_offset(range_offset_first)) + 1, loop_stream);
                    input_count_offsets.set_element_to_zero_async(0, loop_stream);
                    thrust::inclusive_scan(
                      rmm::exec_policy_nosync(loop_stream),
                      input_count_first,
                      input_count_first +
                        (rx_bitmap.size() - packed_bool_offset(range_offset_first)),
                      input_count_offsets.begin() + 1);
                  }
                  range_offset_firsts[j] = range_offset_first;
                  range_offset_lasts[j]  = range_offset_last;
                }
                input_count_offset_vectors.push_back(std::move(input_count_offsets));
              }

              for (size_t j = 0; j < loop_count; ++j) {
                auto partition_idx = i + j;
                auto loop_stream =
                  loop_stream_pool_indices
                    ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                    : handle.get_stream();

                rmm::device_uvector<uint32_t> filtered_bitmap(0, loop_stream);
                rmm::device_uvector<vertex_t> output_count_offsets(0, loop_stream);
                if (nonzero_key_lists[j] && process_local_edges[j]) {
                  auto edge_partition =
                    edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                      graph_view.local_edge_partition_view(partition_idx));

                  auto segment_bitmap = *(edge_partition.dcs_nzd_range_bitmap());

                  auto range_offset_first = range_offset_firsts[j];
                  auto range_offset_last  = range_offset_lasts[j];
                  if (range_offset_first < range_offset_last) {
                    auto const& rx_bitmap = (*edge_partition_bitmap_buffers)[j];
                    filtered_bitmap.resize(
                      rx_bitmap.size() - packed_bool_offset(range_offset_first), loop_stream);
                    thrust::tabulate(
                      rmm::exec_policy_nosync(loop_stream),
                      filtered_bitmap.begin(),
                      filtered_bitmap.end(),
                      cuda::proclaim_return_type<uint32_t>(
                        [range_bitmap =
                           raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                         segment_bitmap = raft::device_span<uint32_t const>(segment_bitmap.data(),
                                                                            segment_bitmap.size()),
                         range_first    = local_v_list_range_firsts[partition_idx],
                         range_offset_first,
                         range_offset_last,
                         major_hypersparse_first =
                           *(edge_partition.major_hypersparse_first())] __device__(size_t i) {
                          auto this_word_range_offset_first = cuda::std::max(
                            static_cast<vertex_t>((packed_bool_offset(range_offset_first) + i) *
                                                  packed_bools_per_word()),
                            range_offset_first);
                          auto this_word_range_offset_last =
                            cuda::std::min(static_cast<vertex_t>(
                                             (packed_bool_offset(range_offset_first) + (i + 1)) *
                                             packed_bools_per_word()),
                                           range_offset_last);
                          auto range_lead_bits = static_cast<size_t>(this_word_range_offset_first %
                                                                     packed_bools_per_word());
                          auto range_bitmap_word =
                            range_bitmap[packed_bool_offset(range_offset_first) + i];
                          if (i == 0) {  // clear the bits in the sparse region
                            range_bitmap_word &= ~packed_bool_partial_mask(range_offset_first %
                                                                           packed_bools_per_word());
                          }
                          auto this_word_hypersparse_offset_first =
                            (range_first + this_word_range_offset_first) - major_hypersparse_first;
                          auto num_bits = static_cast<size_t>(this_word_range_offset_last -
                                                              this_word_range_offset_first);
                          auto hypersparse_lead_bits =
                            static_cast<size_t>(this_word_hypersparse_offset_first) %
                            packed_bools_per_word();
                          auto segment_bitmap_word = ((segment_bitmap[packed_bool_offset(
                                                         this_word_hypersparse_offset_first)] >>
                                                       hypersparse_lead_bits))
                                                     << range_lead_bits;
                          auto remaining_bits =
                            (num_bits > (packed_bools_per_word() - hypersparse_lead_bits))
                              ? (num_bits - (packed_bools_per_word() - hypersparse_lead_bits))
                              : size_t{0};
                          if (remaining_bits > 0) {
                            segment_bitmap_word |=
                              ((segment_bitmap
                                  [packed_bool_offset(this_word_hypersparse_offset_first) + 1] &
                                packed_bool_partial_mask(remaining_bits))
                               << ((packed_bools_per_word() - hypersparse_lead_bits) +
                                   range_lead_bits));
                          }
                          return range_bitmap_word & segment_bitmap_word;
                        }));
                    auto output_count_first = thrust::make_transform_iterator(
                      filtered_bitmap.begin(),
                      cuda::proclaim_return_type<vertex_t>([] __device__(uint32_t word) {
                        return static_cast<vertex_t>(__popc(word));
                      }));
                    output_count_offsets.resize(filtered_bitmap.size() + 1, loop_stream);
                    output_count_offsets.set_element_to_zero_async(0, loop_stream);
                    thrust::inclusive_scan(rmm::exec_policy_nosync(loop_stream),
                                           output_count_first,
                                           output_count_first + filtered_bitmap.size(),
                                           output_count_offsets.begin() + 1);
                  }
                }
                filtered_bitmap_vectors.push_back(std::move(filtered_bitmap));
                output_count_offset_vectors.push_back(std::move(output_count_offsets));
              }

              for (size_t j = 0; j < loop_count; ++j) {
                auto partition_idx = i + j;
                auto loop_stream =
                  loop_stream_pool_indices
                    ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                    : handle.get_stream();

                auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

                auto& keys = edge_partition_key_buffers[j];
                std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>> offsets =
                  rmm::device_uvector<uint32_t>(0, loop_stream);
                if (uint32_key_output_offset) {
                  std::get<0>(offsets).resize(process_local_edges[j]
                                                ? (key_segment_offsets[4] - key_segment_offsets[3])
                                                : vertex_t{0},
                                              loop_stream);
                } else {
                  offsets = rmm::device_uvector<size_t>(
                    process_local_edges[j] ? (key_segment_offsets[4] - key_segment_offsets[3])
                                           : vertex_t{0},
                    loop_stream);
                }

                if (nonzero_key_lists[j] && process_local_edges[j]) {
                  auto range_offset_first = range_offset_firsts[j];
                  auto range_offset_last  = range_offset_lasts[j];
                  if (range_offset_first < range_offset_last) {
                    auto const& rx_bitmap            = (*edge_partition_bitmap_buffers)[j];
                    auto const& input_count_offsets  = input_count_offset_vectors[j];
                    auto const& filtered_bitmap      = filtered_bitmap_vectors[j];
                    auto const& output_count_offsets = output_count_offset_vectors[j];

                    if (keys.index() == 0) {
                      if (offsets.index() == 0) {
                        thrust::for_each(
                          rmm::exec_policy_nosync(loop_stream),
                          thrust::make_counting_iterator(size_t{0}),
                          thrust::make_counting_iterator(filtered_bitmap.size()),
                          [range_bitmap =
                             raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                           filtered_bitmap = raft::device_span<uint32_t const>(
                             filtered_bitmap.data(), filtered_bitmap.size()),
                           input_count_offsets = raft::device_span<vertex_t const>(
                             input_count_offsets.data(), input_count_offsets.size()),
                           output_count_offsets = raft::device_span<vertex_t const>(
                             output_count_offsets.data(), output_count_offsets.size()),
                           output_key_first =
                             get_dataframe_buffer_begin(std::get<0>(keys)) + key_segment_offsets[3],
                           output_offset_first = std::get<0>(offsets).begin(),
                           range_offset_first,
                           start_key_offset = key_segment_offsets[3]] __device__(size_t i) {
                            auto range_bitmap_word =
                              range_bitmap[packed_bool_offset(range_offset_first) + i];
                            if (i == 0) {  // clear the bits in the sparse region
                              range_bitmap_word &= ~packed_bool_partial_mask(
                                range_offset_first % packed_bools_per_word());
                            }
                            auto filtered_bitmap_word = filtered_bitmap[i];
                            auto lead_bits            = (i == 0)
                                                          ? static_cast<unsigned int>(range_offset_first %
                                                                           packed_bools_per_word())
                                                          : static_cast<unsigned int>(0);
                            auto this_word_start_v_offset =
                              static_cast<uint32_t>((packed_bool_offset(range_offset_first) + i) *
                                                    packed_bools_per_word());
                            auto this_word_start_key_offset =
                              static_cast<uint32_t>(start_key_offset + input_count_offsets[i]);
                            auto this_word_output_start_offset = output_count_offsets[i];
                            for (int j = 0; j < __popc(filtered_bitmap_word); ++j) {
                              auto jth_set_bit_pos = static_cast<uint32_t>(
                                __fns(filtered_bitmap_word, lead_bits, j + 1));
                              *(output_key_first + (this_word_output_start_offset + j)) =
                                this_word_start_v_offset + jth_set_bit_pos;
                              *(output_offset_first + (this_word_output_start_offset + j)) =
                                this_word_start_key_offset +
                                static_cast<uint32_t>(__popc(
                                  range_bitmap_word & packed_bool_partial_mask(jth_set_bit_pos)));
                            }
                          });
                      } else {
                        thrust::for_each(
                          rmm::exec_policy_nosync(loop_stream),
                          thrust::make_counting_iterator(size_t{0}),
                          thrust::make_counting_iterator(filtered_bitmap.size()),
                          [range_bitmap =
                             raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                           filtered_bitmap = raft::device_span<uint32_t const>(
                             filtered_bitmap.data(), filtered_bitmap.size()),
                           input_count_offsets = raft::device_span<vertex_t const>(
                             input_count_offsets.data(), input_count_offsets.size()),
                           output_count_offsets = raft::device_span<vertex_t const>(
                             output_count_offsets.data(), output_count_offsets.size()),
                           output_key_first =
                             get_dataframe_buffer_begin(std::get<0>(keys)) + key_segment_offsets[3],
                           output_offset_first = std::get<1>(offsets).begin(),
                           range_offset_first,
                           start_key_offset = key_segment_offsets[3]] __device__(size_t i) {
                            auto range_bitmap_word =
                              range_bitmap[packed_bool_offset(range_offset_first) + i];
                            if (i == 0) {  // clear the bits in the sparse region
                              range_bitmap_word &= ~packed_bool_partial_mask(
                                range_offset_first % packed_bools_per_word());
                            }
                            auto filtered_bitmap_word = filtered_bitmap[i];
                            auto lead_bits            = (i == 0)
                                                          ? static_cast<unsigned int>(range_offset_first %
                                                                           packed_bools_per_word())
                                                          : static_cast<unsigned int>(0);
                            auto this_word_start_v_offset =
                              static_cast<uint32_t>((packed_bool_offset(range_offset_first) + i) *
                                                    packed_bools_per_word());
                            auto this_word_start_key_offset =
                              static_cast<size_t>(start_key_offset + input_count_offsets[i]);
                            auto this_word_output_start_offset = output_count_offsets[i];
                            for (int j = 0; j < __popc(filtered_bitmap_word); ++j) {
                              auto jth_set_bit_pos = static_cast<uint32_t>(
                                __fns(filtered_bitmap_word, lead_bits, j + 1));
                              *(output_key_first + (this_word_output_start_offset + j)) =
                                this_word_start_v_offset + jth_set_bit_pos;
                              *(output_offset_first + (this_word_output_start_offset + j)) =
                                this_word_start_key_offset +
                                static_cast<size_t>(__popc(
                                  range_bitmap_word & packed_bool_partial_mask(jth_set_bit_pos)));
                            }
                          });
                      }
                    } else {
                      if (offsets.index() == 0) {
                        thrust::for_each(
                          rmm::exec_policy_nosync(loop_stream),
                          thrust::make_counting_iterator(size_t{0}),
                          thrust::make_counting_iterator(filtered_bitmap.size()),
                          [range_bitmap =
                             raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                           filtered_bitmap = raft::device_span<uint32_t const>(
                             filtered_bitmap.data(), filtered_bitmap.size()),
                           input_count_offsets = raft::device_span<vertex_t const>(
                             input_count_offsets.data(), input_count_offsets.size()),
                           output_count_offsets = raft::device_span<vertex_t const>(
                             output_count_offsets.data(), output_count_offsets.size()),
                           output_key_first =
                             get_dataframe_buffer_begin(std::get<0>(keys)) + key_segment_offsets[3],
                           output_offset_first = std::get<0>(offsets).begin(),
                           range_first         = local_v_list_range_firsts[partition_idx],
                           range_offset_first,
                           start_key_offset = key_segment_offsets[3]] __device__(size_t i) {
                            auto range_bitmap_word =
                              range_bitmap[packed_bool_offset(range_offset_first) + i];
                            if (i == 0) {  // clear the bits in the sparse region
                              range_bitmap_word &= ~packed_bool_partial_mask(
                                range_offset_first % packed_bools_per_word());
                            }
                            auto filtered_bitmap_word = filtered_bitmap[i];
                            auto lead_bits            = (i == 0)
                                                          ? static_cast<unsigned int>(range_offset_first %
                                                                           packed_bools_per_word())
                                                          : static_cast<unsigned int>(0);
                            auto this_word_start_v =
                              range_first +
                              static_cast<vertex_t>((packed_bool_offset(range_offset_first) + i) *
                                                    packed_bools_per_word());
                            auto this_word_start_key_offset =
                              static_cast<uint32_t>(start_key_offset + input_count_offsets[i]);
                            auto this_word_output_start_offset = output_count_offsets[i];
                            for (int j = 0; j < __popc(filtered_bitmap_word); ++j) {
                              auto jth_set_bit_pos = static_cast<vertex_t>(
                                __fns(filtered_bitmap_word, lead_bits, j + 1));
                              *(output_key_first + (this_word_output_start_offset + j)) =
                                this_word_start_v + jth_set_bit_pos;
                              *(output_offset_first + (this_word_output_start_offset + j)) =
                                this_word_start_key_offset +
                                static_cast<uint32_t>(__popc(
                                  range_bitmap_word & packed_bool_partial_mask(jth_set_bit_pos)));
                            }
                          });
                      } else {
                        thrust::for_each(
                          rmm::exec_policy_nosync(loop_stream),
                          thrust::make_counting_iterator(size_t{0}),
                          thrust::make_counting_iterator(filtered_bitmap.size()),
                          [range_bitmap =
                             raft::device_span<uint32_t const>(rx_bitmap.data(), rx_bitmap.size()),
                           filtered_bitmap = raft::device_span<uint32_t const>(
                             filtered_bitmap.data(), filtered_bitmap.size()),
                           input_count_offsets = raft::device_span<vertex_t const>(
                             input_count_offsets.data(), input_count_offsets.size()),
                           output_count_offsets = raft::device_span<vertex_t const>(
                             output_count_offsets.data(), output_count_offsets.size()),
                           output_key_first =
                             get_dataframe_buffer_begin(std::get<0>(keys)) + key_segment_offsets[3],
                           output_offset_first = std::get<1>(offsets).begin(),
                           range_first         = local_v_list_range_firsts[partition_idx],
                           range_offset_first,
                           start_key_offset = key_segment_offsets[3]] __device__(size_t i) {
                            auto range_bitmap_word =
                              range_bitmap[packed_bool_offset(range_offset_first) + i];
                            if (i == 0) {  // clear the bits in the sparse region
                              range_bitmap_word &= ~packed_bool_partial_mask(
                                range_offset_first % packed_bools_per_word());
                            }
                            auto filtered_bitmap_word = filtered_bitmap[i];
                            auto lead_bits            = (i == 0)
                                                          ? static_cast<unsigned int>(range_offset_first %
                                                                           packed_bools_per_word())
                                                          : static_cast<unsigned int>(0);
                            auto this_word_start_v =
                              range_first +
                              static_cast<vertex_t>((packed_bool_offset(range_offset_first) + i) *
                                                    packed_bools_per_word());
                            auto this_word_start_key_offset =
                              static_cast<size_t>(start_key_offset + input_count_offsets[i]);
                            auto this_word_output_start_offset = output_count_offsets[i];
                            for (int j = 0; j < __popc(filtered_bitmap_word); ++j) {
                              auto jth_set_bit_pos = static_cast<vertex_t>(
                                __fns(filtered_bitmap_word, lead_bits, j + 1));
                              *(output_key_first + (this_word_output_start_offset + j)) =
                                this_word_start_v + jth_set_bit_pos;
                              *(output_offset_first + (this_word_output_start_offset + j)) =
                                this_word_start_key_offset +
                                static_cast<size_t>(__popc(
                                  range_bitmap_word & packed_bool_partial_mask(jth_set_bit_pos)));
                            }
                          });
                      }
                    }
                    thrust::transform(
                      rmm::exec_policy_nosync(loop_stream),
                      output_count_offsets.begin() + (output_count_offsets.size() - 1),
                      output_count_offsets.end(),
                      counters.data() + j,
                      typecast_t<vertex_t, size_t>{});
                  } else {
                    thrust::fill(rmm::exec_policy_nosync(loop_stream),
                                 counters.data() + j,
                                 counters.data() + (j + 1),
                                 size_t{0});
                  }
                }

                edge_partition_hypersparse_key_offset_vectors->push_back(std::move(offsets));
              }
            }
          }
          if (edge_partition_new_key_buffers) {  // if (!try_bitmap || !v_list_bitmap)
            for (size_t j = 0; j < loop_count; ++j) {
              auto partition_idx = i + j;
              auto loop_stream =
                loop_stream_pool_indices
                  ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                  : handle.get_stream();

              auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

              auto& keys = edge_partition_key_buffers[j];
              std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>> offsets =
                rmm::device_uvector<uint32_t>(0, loop_stream);
              if (uint32_key_output_offset) {
                std::get<0>(offsets).resize(process_local_edges[j]
                                              ? (key_segment_offsets[4] - key_segment_offsets[3])
                                              : vertex_t{0},
                                            loop_stream);
              } else {
                offsets = rmm::device_uvector<size_t>(
                  process_local_edges[j] ? (key_segment_offsets[4] - key_segment_offsets[3])
                                         : vertex_t{0},
                  loop_stream);
              }

              if (nonzero_key_lists[j] && process_local_edges[j]) {
                auto edge_partition =
                  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                    graph_view.local_edge_partition_view(partition_idx));
                auto const& segment_offsets =
                  graph_view.local_edge_partition_segment_offsets(partition_idx);

                auto segment_bitmap = *(edge_partition.dcs_nzd_range_bitmap());

                auto& new_keys = (*edge_partition_new_key_buffers)[j];
                if constexpr (try_bitmap) {
                  assert(!v_list_bitmap);
                  if (keys.index() == 0) {
                    auto flag_first = thrust::make_transform_iterator(
                      get_dataframe_buffer_begin(std::get<0>(keys)) + key_segment_offsets[3],
                      cuda::proclaim_return_type<bool>(
                        [segment_bitmap = raft::device_span<uint32_t const>(segment_bitmap.data(),
                                                                            segment_bitmap.size()),
                         range_first    = local_v_list_range_firsts[partition_idx],
                         major_hypersparse_first =
                           *(edge_partition
                               .major_hypersparse_first())] __device__(uint32_t v_offset) {
                          auto v              = range_first + static_cast<vertex_t>(v_offset);
                          auto segment_offset = v - major_hypersparse_first;
                          return ((segment_bitmap[packed_bool_offset(segment_offset)] &
                                   packed_bool_mask(segment_offset)) != packed_bool_empty_mask());
                        }));
                    if (offsets.index() == 0) {
                      auto input_pair_first =
                        thrust::make_zip_iterator(get_dataframe_buffer_begin(std::get<0>(keys)),
                                                  thrust::make_counting_iterator(uint32_t{0})) +
                        key_segment_offsets[3];
                      detail::copy_if_nosync(
                        input_pair_first,
                        input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                        flag_first,
                        thrust::make_zip_iterator(
                          get_dataframe_buffer_begin(std::get<0>(new_keys)) +
                            key_segment_offsets[3],
                          std::get<0>(offsets).begin()),
                        raft::device_span<size_t>(counters.data() + j, size_t{1}),
                        loop_stream);
                    } else {
                      auto input_pair_first =
                        thrust::make_zip_iterator(get_dataframe_buffer_begin(std::get<0>(keys)),
                                                  thrust::make_counting_iterator(size_t{0})) +
                        key_segment_offsets[3];
                      detail::copy_if_nosync(
                        input_pair_first,
                        input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                        flag_first,
                        thrust::make_zip_iterator(
                          get_dataframe_buffer_begin(std::get<0>(new_keys)) +
                            key_segment_offsets[3],
                          std::get<1>(offsets).begin()),
                        raft::device_span<size_t>(counters.data() + j, size_t{1}),
                        loop_stream);
                    }
                  } else {
                    auto flag_first = thrust::make_transform_iterator(
                      get_dataframe_buffer_begin(std::get<1>(keys)) + key_segment_offsets[3],
                      cuda::proclaim_return_type<bool>(
                        [segment_bitmap = raft::device_span<uint32_t const>(segment_bitmap.data(),
                                                                            segment_bitmap.size()),
                         major_hypersparse_first =
                           *(edge_partition.major_hypersparse_first())] __device__(vertex_t v) {
                          auto segment_offset = v - major_hypersparse_first;
                          return ((segment_bitmap[packed_bool_offset(segment_offset)] &
                                   packed_bool_mask(segment_offset)) != packed_bool_empty_mask());
                        }));
                    if (offsets.index() == 0) {
                      auto input_pair_first =
                        thrust::make_zip_iterator(get_dataframe_buffer_begin(std::get<1>(keys)),
                                                  thrust::make_counting_iterator(uint32_t{0})) +
                        key_segment_offsets[3];
                      detail::copy_if_nosync(
                        input_pair_first,
                        input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                        flag_first,
                        thrust::make_zip_iterator(
                          get_dataframe_buffer_begin(std::get<1>(new_keys)) +
                            key_segment_offsets[3],
                          std::get<0>(offsets).begin()),
                        raft::device_span<size_t>(counters.data() + j, size_t{1}),
                        loop_stream);
                    } else {
                      auto input_pair_first =
                        thrust::make_zip_iterator(get_dataframe_buffer_begin(std::get<1>(keys)),
                                                  thrust::make_counting_iterator(size_t{0})) +
                        key_segment_offsets[3];
                      detail::copy_if_nosync(
                        input_pair_first,
                        input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                        flag_first,
                        thrust::make_zip_iterator(
                          get_dataframe_buffer_begin(std::get<1>(new_keys)) +
                            key_segment_offsets[3],
                          std::get<1>(offsets).begin()),
                        raft::device_span<size_t>(counters.data() + j, size_t{1}),
                        loop_stream);
                    }
                  }
                } else {
                  auto flag_first = thrust::make_transform_iterator(
                    get_dataframe_buffer_begin(keys) + key_segment_offsets[3],
                    cuda::proclaim_return_type<bool>(
                      [segment_bitmap = raft::device_span<uint32_t const>(segment_bitmap.data(),
                                                                          segment_bitmap.size()),
                       major_hypersparse_first =
                         *(edge_partition.major_hypersparse_first())] __device__(auto key) {
                        auto segment_offset =
                          thrust_tuple_get_or_identity<key_t, 0>(key) - major_hypersparse_first;
                        return ((segment_bitmap[packed_bool_offset(segment_offset)] &
                                 packed_bool_mask(segment_offset)) != packed_bool_empty_mask());
                      }));
                  if (offsets.index() == 0) {
                    auto input_pair_first =
                      thrust::make_zip_iterator(get_dataframe_buffer_begin(keys),
                                                thrust::make_counting_iterator(uint32_t{0})) +
                      key_segment_offsets[3];
                    detail::copy_if_nosync(
                      input_pair_first,
                      input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                      flag_first,
                      thrust::make_zip_iterator(
                        get_dataframe_buffer_begin(new_keys) + key_segment_offsets[3],
                        std::get<0>(offsets).begin()),
                      raft::device_span<size_t>(counters.data() + j, size_t{1}),
                      loop_stream);
                  } else {
                    auto input_pair_first =
                      thrust::make_zip_iterator(get_dataframe_buffer_begin(keys),
                                                thrust::make_counting_iterator(size_t{0})) +
                      key_segment_offsets[3];
                    detail::copy_if_nosync(
                      input_pair_first,
                      input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                      flag_first,
                      thrust::make_zip_iterator(
                        get_dataframe_buffer_begin(new_keys) + key_segment_offsets[3],
                        std::get<1>(offsets).begin()),
                      raft::device_span<size_t>(counters.data() + j, size_t{1}),
                      loop_stream);
                  }
                }
              }

              edge_partition_hypersparse_key_offset_vectors->push_back(std::move(offsets));
            }
          }
          if (loop_stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }
          if (edge_partition_new_key_buffers) {
            for (size_t j = 0; j < loop_count; ++j) {
              edge_partition_key_buffers[j] = std::move((*edge_partition_new_key_buffers)[j]);
            }
          }
          if (edge_partition_bitmap_buffers) { edge_partition_bitmap_buffers->clear(); }

          auto h_counts = reinterpret_cast<size_t*>(h_staging_buffer.data());
          assert(h_staging_buffer.size() >= loop_count);
          raft::update_host(h_counts, counters.data(), loop_count, handle.get_stream());
          handle.sync_stream();

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream =
              loop_stream_pool_indices
                ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                : handle.get_stream();

            if (nonzero_key_lists[j] && process_local_edges[j]) {
              auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

              auto& keys = edge_partition_key_buffers[j];
              if constexpr (try_bitmap) {
                if (keys.index() == 0) {
                  resize_dataframe_buffer(
                    std::get<0>(keys), key_segment_offsets[3] + h_counts[j], loop_stream);
                } else {
                  resize_dataframe_buffer(
                    std::get<1>(keys), key_segment_offsets[3] + h_counts[j], loop_stream);
                }
              } else {
                resize_dataframe_buffer(keys, key_segment_offsets[3] + h_counts[j], loop_stream);
              }
              // skip shrink_to_fit to cut execution time

              auto& offsets = (*edge_partition_hypersparse_key_offset_vectors)[j];
              if (offsets.index() == 0) {
                std::get<0>(offsets).resize(h_counts[j], loop_stream);
              } else {
                std::get<1>(offsets).resize(h_counts[j], loop_stream);
              }
              // skip shrink_to_fit to cut execution time
            }
          }
        }
      }
    }

    std::conditional_t<GraphViewType::is_multi_gpu && update_major,
                       std::vector<dataframe_buffer_type_t<T>>,
                       std::byte /* dummy */>
      edge_partition_major_output_buffers{};
    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      edge_partition_major_output_buffers.reserve(loop_count);
    }

    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx = i + j;
        auto loop_stream   = loop_stream_pool_indices
                               ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                               : handle.get_stream();

        size_t buffer_size{0};
        if (process_local_edges[j]) {
          if constexpr (use_input_key) {
            auto const& keys = edge_partition_key_buffers[j];
            if constexpr (try_bitmap) {
              if (keys.index() == 0) {
                buffer_size = size_dataframe_buffer(std::get<0>(keys));
              } else {
                buffer_size = size_dataframe_buffer(std::get<1>(keys));
              }
            } else {
              buffer_size = size_dataframe_buffer(keys);
            }
          } else {
            auto edge_partition =
              edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                graph_view.local_edge_partition_view(partition_idx));
            auto const& segment_offsets =
              graph_view.local_edge_partition_segment_offsets(partition_idx);
            // check segment_offsets->size() >= 2 to silence a compiler warning with GCC 14 (if
            // segment_offsets.has_value() is true, segment_offsets->size() should always be larger
            // than 2, so this check shouldn't be necessary otherwise).
            buffer_size =
              segment_offsets
                ? (segment_offsets->size() >= 2
                     ? *(segment_offsets->rbegin() + 1) /* exclude the zero degree segment */
                     : vertex_t{0})
                : edge_partition.major_range_size();
          }
        }
        edge_partition_major_output_buffers.push_back(
          allocate_dataframe_buffer<T>(buffer_size, loop_stream));
      }
    }
    if (loop_stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }

    for (size_t j = 0; j < loop_count; ++j) {
      if constexpr (use_input_key) {
        if (!nonzero_key_lists[j]) { continue; }
      }
      if (!process_local_edges[j]) { continue; }

      auto partition_idx = i + j;

      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(partition_idx));
      auto edge_partition_e_mask =
        edge_mask_view
          ? cuda::std::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, partition_idx)
          : cuda::std::nullopt;
      size_t num_streams_per_loop{1};
      if (stream_pool_indices) {
        assert((*stream_pool_indices).size() >= num_concurrent_loops);
        num_streams_per_loop = (*stream_pool_indices).size() / num_concurrent_loops;
      }
      auto edge_partition_stream_pool_indices =
        stream_pool_indices
          ? std::make_optional<raft::host_span<size_t const>>(
              (*stream_pool_indices).data() + j * num_streams_per_loop, num_streams_per_loop)
          : std::nullopt;

      T major_init{};
      T major_identity_element{};
      if constexpr (update_major) {
        if constexpr (std::is_same_v<ReduceOp,
                                     reduce_op::any<T>>) {  // if any edge has a non-init value,
                                                            // one of the non-init values will
                                                            // be selected.
          major_init             = init;
          major_identity_element = init;
        } else {
          major_init = ReduceOp::identity_element;
          if constexpr (GraphViewType::is_multi_gpu) {
            auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
            auto const minor_comm_rank = minor_comm.get_rank();
            major_init                 = (static_cast<int>(partition_idx) == minor_comm_rank)
                                           ? init
                                           : ReduceOp::identity_element;
          } else {
            major_init = init;
          }
          major_identity_element = ReduceOp::identity_element;
        }
      }

      std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
      if constexpr (use_input_key) {
        if (key_segment_offset_vectors) {
          key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];
          if constexpr (filter_input_key) {
            if (edge_partition_hypersparse_key_offset_vectors) {
              key_segment_offsets->back() =
                size_dataframe_buffer(edge_partition_major_output_buffers[j]);
              *(key_segment_offsets->rbegin() + 1) = key_segment_offsets->back();
            }
          }
        }
      } else {
        auto const& segment_offsets =
          graph_view.local_edge_partition_segment_offsets(partition_idx);
        if (segment_offsets) {
          key_segment_offsets = std::vector<size_t>(segment_offsets->size());
          std::transform(segment_offsets->begin(),
                         segment_offsets->end(),
                         key_segment_offsets->begin(),
                         [](vertex_t offset) { return static_cast<size_t>(offset); });
        }
      }

      edge_partition_src_input_device_view_t edge_partition_src_value_input{};
      edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
      if constexpr (GraphViewType::is_storage_transposed) {
        edge_partition_src_value_input =
          edge_partition_src_input_device_view_t(edge_src_value_input);
        edge_partition_dst_value_input =
          edge_partition_dst_input_device_view_t(edge_dst_value_input, partition_idx);
      } else {
        edge_partition_src_value_input =
          edge_partition_src_input_device_view_t(edge_src_value_input, partition_idx);
        edge_partition_dst_value_input =
          edge_partition_dst_input_device_view_t(edge_dst_value_input);
      }
      auto edge_partition_e_value_input =
        edge_partition_e_input_device_view_t(edge_value_input, partition_idx);

      std::conditional_t<GraphViewType::is_multi_gpu,
                         std::conditional_t<update_major,
                                            dataframe_buffer_iterator_type_t<T>,
                                            edge_partition_minor_output_device_view_t>,
                         VertexValueOutputIterator>
        output_buffer{};
      if constexpr (GraphViewType::is_multi_gpu) {
        if constexpr (update_major) {
          output_buffer = get_dataframe_buffer_begin(edge_partition_major_output_buffers[j]);
        } else {
          output_buffer =
            edge_partition_minor_output_device_view_t(minor_tmp_buffer->mutable_view());
        }
      } else {
        output_buffer = tmp_vertex_value_output_first;
      }

      bool processed{false};
      if constexpr (try_bitmap) {
        auto const& keys = edge_partition_key_buffers[j];
        if (keys.index() == 0) {
          auto edge_partition_key_first = thrust::make_transform_iterator(
            std::get<0>(keys).begin(),
            cuda::proclaim_return_type<vertex_t>(
              [range_first = local_v_list_range_firsts[partition_idx]] __device__(
                uint32_t v_offset) { return range_first + static_cast<vertex_t>(v_offset); }));
          per_v_transform_reduce_e_edge_partition<update_major, GraphViewType>(
            handle,
            edge_partition,
            edge_partition_key_first,
            edge_partition_key_first + std::get<0>(keys).size(),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            output_buffer,
            e_op,
            major_init,
            major_identity_element,
            reduce_op,
            pred_op,
            key_segment_offsets ? std::make_optional<raft::host_span<size_t const>>(
                                    key_segment_offsets->data(), key_segment_offsets->size())
                                : std::nullopt,
            edge_partition_stream_pool_indices);
          processed = true;
        }
      }
      if (!processed) {
        auto edge_partition_key_first = sorted_unique_key_first;
        auto edge_partition_key_last  = sorted_unique_nzd_key_last;
        if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
          auto const& keys = edge_partition_key_buffers[j];
          if constexpr (try_bitmap) {
            edge_partition_key_first = get_dataframe_buffer_begin(std::get<1>(keys));
            edge_partition_key_last  = get_dataframe_buffer_end(std::get<1>(keys));
          } else {
            edge_partition_key_first = get_dataframe_buffer_begin(keys);
            edge_partition_key_last  = get_dataframe_buffer_end(keys);
          }
        }

        per_v_transform_reduce_e_edge_partition<update_major, GraphViewType>(
          handle,
          edge_partition,
          edge_partition_key_first,
          edge_partition_key_last,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          output_buffer,
          e_op,
          major_init,
          major_identity_element,
          reduce_op,
          pred_op,
          key_segment_offsets ? std::make_optional<raft::host_span<size_t const>>(
                                  key_segment_offsets->data(), key_segment_offsets->size())
                              : std::nullopt,
          edge_partition_stream_pool_indices);
      }
    }
    if (stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }

    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      auto const minor_comm_size = minor_comm.get_size();

      if constexpr (use_input_key) {
        edge_partition_key_buffers.clear();
        edge_partition_key_buffers.shrink_to_fit();
      }

      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        size_t tx_buf_size_per_rank{};
        {
          size_t max_size{0};
          for (size_t j = 0; j < loop_count; ++j) {
            auto const& output_buffer = edge_partition_major_output_buffers[j];
            max_size                  = std::max(max_size, size_dataframe_buffer(output_buffer));
          }
          minor_comm.host_allreduce(
            std::addressof(max_size), std::addressof(max_size), size_t{1}, raft::comms::op_t::MAX);
          tx_buf_size_per_rank = raft::round_up_safe(max_size, alignment);
        }

        std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>
          tx_key_output_offsets = rmm::device_uvector<uint32_t>(0, handle.get_stream());
        if (uint32_key_output_offset) {
          resize_dataframe_buffer(std::get<0>(tx_key_output_offsets),
                                  tx_buf_size_per_rank * loop_count,
                                  handle.get_stream());
        } else {
          tx_key_output_offsets =
            rmm::device_uvector<size_t>(tx_buf_size_per_rank * loop_count, handle.get_stream());
        }
        auto tx_values =
          allocate_dataframe_buffer<T>(tx_buf_size_per_rank * loop_count, handle.get_stream());
        thrust::fill(
          handle.get_thrust_policy(), counters.data(), counters.data() + loop_count, size_t{0});
        handle.sync_stream();

        for (size_t j = 0; j < loop_count; ++j) {
          auto loop_stream          = loop_stream_pool_indices
                                        ? handle.get_stream_from_stream_pool((*loop_stream_pool_indices)[j])
                                        : handle.get_stream();
          auto const& output_buffer = edge_partition_major_output_buffers[j];

          if (nonzero_key_lists[j] && process_local_edges[j]) {
            std::optional<
              std::variant<raft::device_span<uint32_t const>, raft::device_span<size_t const>>>
              hypersparse_key_offsets{std::nullopt};
            if constexpr (filter_input_key) {
              if (edge_partition_hypersparse_key_offset_vectors) {
                auto const& offsets = (*edge_partition_hypersparse_key_offset_vectors)[j];

                if (offsets.index() == 0) {
                  assert(uint32_key_output_offset);
                  hypersparse_key_offsets = raft::device_span<uint32_t const>(
                    std::get<0>(offsets).data(), std::get<0>(offsets).size());
                } else {
                  hypersparse_key_offsets = raft::device_span<size_t const>(
                    std::get<1>(offsets).data(), std::get<1>(offsets).size());
                }
              }
            }

            if (uint32_key_output_offset) {
              copy_valid_offset_value_pairs(
                get_dataframe_buffer_begin(output_buffer),
                get_dataframe_buffer_end(output_buffer),
                std::get<0>(tx_key_output_offsets).begin() + tx_buf_size_per_rank * j,
                get_dataframe_buffer_begin(tx_values) + tx_buf_size_per_rank * j,
                raft::device_span<size_t>(counters.data() + j, size_t{1}),
                hypersparse_key_offsets ? std::make_optional<raft::device_span<uint32_t const>>(
                                            std::get<0>(*hypersparse_key_offsets).data(),
                                            std::get<0>(*hypersparse_key_offsets).size())
                                        : std::nullopt,
                init,
                loop_stream);
            } else {
              copy_valid_offset_value_pairs(
                get_dataframe_buffer_begin(output_buffer),
                get_dataframe_buffer_end(output_buffer),
                std::get<1>(tx_key_output_offsets).begin() + tx_buf_size_per_rank * j,
                get_dataframe_buffer_begin(tx_values) + tx_buf_size_per_rank * j,
                raft::device_span<size_t>(counters.data() + j, size_t{1}),
                hypersparse_key_offsets ? std::make_optional<raft::device_span<size_t const>>(
                                            std::get<1>(*hypersparse_key_offsets).data(),
                                            std::get<1>(*hypersparse_key_offsets).size())
                                        : std::nullopt,
                init,
                loop_stream);
            }
          }
        }
        if (loop_stream_pool_indices) { RAFT_CUDA_TRY(cudaDeviceSynchronize()); }
        edge_partition_major_output_buffers.clear();
        if constexpr (filter_input_key) {
          edge_partition_hypersparse_key_offset_vectors = std::nullopt;
        }

        std::vector<size_t> edge_partition_valid_counts(loop_count, 0);
        std::vector<size_t> tx_counts(minor_comm_size, 0);
        std::vector<size_t> rx_counts(minor_comm_size, 0);
        size_t rx_buf_size_per_rank{};
        {
          auto h_counts = reinterpret_cast<size_t*>(h_staging_buffer.data());
          assert(h_staging_buffer.size() >= loop_count);
          raft::update_host(h_counts, counters.data(), loop_count, handle.get_stream());
          handle.sync_stream();
          std::vector<size_t> h_allgathered_valid_counts(minor_comm_size * loop_count);
          std::copy(h_counts,
                    h_counts + loop_count,
                    h_allgathered_valid_counts.begin() + minor_comm_rank * loop_count);
          minor_comm.host_allgather(
            h_allgathered_valid_counts.data(), h_allgathered_valid_counts.data(), loop_count);
          for (size_t j = 0; j < loop_count; ++j) {
            if (nonzero_key_lists[j] && process_local_edges[j]) {
              edge_partition_valid_counts[j] =
                h_allgathered_valid_counts[minor_comm_rank * loop_count + j];
            }
          }
          auto max_size        = std::reduce(h_allgathered_valid_counts.begin(),
                                      h_allgathered_valid_counts.end(),
                                      size_t{0},
                                      [](auto l, auto r) { return std::max(l, r); });
          rx_buf_size_per_rank = raft::round_up_safe(max_size, alignment);
          std::copy(edge_partition_valid_counts.begin(),
                    edge_partition_valid_counts.end(),
                    tx_counts.begin() + i * num_concurrent_loops);
          if (static_cast<size_t>(minor_comm_rank / num_concurrent_loops) ==
              (i / num_concurrent_loops)) {
            auto loop_offset = minor_comm_rank % num_concurrent_loops;
            assert(loop_offset < loop_count);
            for (int j = 0; j < minor_comm_size; ++j) {
              rx_counts[j] = h_allgathered_valid_counts[j * loop_count + loop_offset];
            }
          }
        }

        std::vector<int> ranks(minor_comm_size, 0);
        std::iota(ranks.begin(), ranks.end(), int{0});
        std::vector<size_t> tx_displs(minor_comm_size, 0);
        for (size_t j = 0; j < loop_count; ++j) {
          tx_displs[i * num_concurrent_loops + j] = tx_buf_size_per_rank;
        }
        std::exclusive_scan(tx_displs.begin(), tx_displs.end(), tx_displs.begin(), size_t{0});
        std::vector<size_t> rx_displs(minor_comm_size, 0);
        if (static_cast<size_t>(minor_comm_rank / num_concurrent_loops) ==
            (i / num_concurrent_loops)) {
          for (int j = 0; j < minor_comm_size; ++j) {
            rx_displs[j] = j * rx_buf_size_per_rank;
          }
        }

        std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>
          rx_key_output_offsets = rmm::device_uvector<uint32_t>(0, handle.get_stream());
        if (uint32_key_output_offset) {
          std::get<0>(rx_key_output_offsets)
            .resize(rx_displs.back() + rx_counts.back(), handle.get_stream());
          device_multicast_sendrecv(
            minor_comm,
            std::get<0>(tx_key_output_offsets).begin(),
            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            std::get<0>(rx_key_output_offsets).begin(),
            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            handle.get_stream());
          std::get<0>(tx_key_output_offsets).resize(0, handle.get_stream());
          std::get<0>(tx_key_output_offsets).shrink_to_fit(handle.get_stream());
        } else {
          rx_key_output_offsets =
            rmm::device_uvector<size_t>(rx_displs.back() + rx_counts.back(), handle.get_stream());
          device_multicast_sendrecv(
            minor_comm,
            std::get<1>(tx_key_output_offsets).begin(),
            raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
            raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            std::get<1>(rx_key_output_offsets).begin(),
            raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
            raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
            raft::host_span<int const>(ranks.data(), ranks.size()),
            handle.get_stream());
          std::get<1>(tx_key_output_offsets).resize(0, handle.get_stream());
          std::get<1>(tx_key_output_offsets).shrink_to_fit(handle.get_stream());
        }
        auto rx_values =
          allocate_dataframe_buffer<T>(rx_displs.back() + rx_counts.back(), handle.get_stream());
        device_multicast_sendrecv(minor_comm,
                                  get_dataframe_buffer_begin(tx_values),
                                  raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                                  raft::host_span<size_t const>(tx_displs.data(), tx_displs.size()),
                                  raft::host_span<int const>(ranks.data(), ranks.size()),
                                  get_dataframe_buffer_begin(rx_values),
                                  raft::host_span<size_t const>(rx_counts.data(), rx_counts.size()),
                                  raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
                                  raft::host_span<int const>(ranks.data(), ranks.size()),
                                  handle.get_stream());
        resize_dataframe_buffer(tx_values, 0, handle.get_stream());
        shrink_to_fit_dataframe_buffer(tx_values, handle.get_stream());

        if (static_cast<size_t>(minor_comm_rank / num_concurrent_loops) ==
            (i / num_concurrent_loops)) {
          auto h_rx_counts = reinterpret_cast<size_t*>(h_staging_buffer.data());
          assert(h_staging_buffer.size() >= minor_comm_size);
          std::copy(rx_counts.begin(), rx_counts.end(), h_rx_counts);
          rmm::device_uvector<size_t> d_rx_counts(rx_counts.size(), handle.get_stream());
          raft::update_device(
            d_rx_counts.data(), h_rx_counts, minor_comm_size, handle.get_stream());

          // remove duplicates
          size_t key_output_offset_range_size{0};
          if constexpr (use_input_key) {
            if constexpr (filter_input_key) {
              key_output_offset_range_size = size_dataframe_buffer(tmp_output_indices);
            } else {
              key_output_offset_range_size =
                cuda::std::distance(sorted_unique_key_first, sorted_unique_nzd_key_last);
            }
          } else {
            if (local_vertex_partition_segment_offsets) {
              key_output_offset_range_size = *(local_vertex_partition_segment_offsets.rbegin() + 1);
            } else {
              key_output_offset_range_size = graph_view.local_vertex_partition_range_size();
            }
          }
          rmm::device_uvector<uint32_t> bitmap(packed_bool_size(key_output_offset_range_size) +
                                                 packed_bool_size(size_dataframe_buffer(rx_values)),
                                               handle.get_stream());
          thrust::fill(
            handle.get_thrust_policy(), bitmap.begin(), bitmap.end(), packed_bool_empty_mask());
          raft::device_span<uint32_t> claimed_bitmap(
            bitmap.data(), packed_bool_size(key_output_offset_range_size));
          raft::device_span<uint32_t> keep_bitmap(
            bitmap.data() + packed_bool_size(key_output_offset_range_size),
            packed_bool_size(size_dataframe_buffer(rx_values)));
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(packed_bool_size(size_dataframe_buffer(rx_values))),
            [counts       = raft::device_span<size_t const>(d_rx_counts.data(), d_rx_counts.size()),
             offset_first = uint32_key_output_offset
                              ? static_cast<void*>(std::get<0>(rx_key_output_offsets).data())
                              : static_cast<void*>(std::get<1>(rx_key_output_offsets).data()),
             num_rx_values = size_dataframe_buffer(rx_values),
             claimed_bitmap,
             keep_bitmap,
             rx_buf_size_per_rank,
             uint32_key_output_offset] __device__(size_t i) {
              uint32_t keep_word{0};
              for (size_t j = i * packed_bools_per_word();
                   j < cuda::std::min((i + 1) * packed_bools_per_word(), num_rx_values);
                   ++j) {
                auto rank = j / rx_buf_size_per_rank;
                if ((j % rx_buf_size_per_rank) < counts[rank]) {
                  size_t offset{};
                  if (uint32_key_output_offset) {
                    offset = static_cast<size_t>(*(static_cast<uint32_t*>(offset_first) + j));
                  } else {
                    offset = *(static_cast<size_t*>(offset_first) + j);
                  }
                  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> claimed_word(
                    claimed_bitmap[packed_bool_offset(offset)]);
                  auto old = claimed_word.fetch_or(packed_bool_mask(offset),
                                                   cuda::std::memory_order_relaxed);
                  if ((old & packed_bool_mask(offset)) ==
                      packed_bool_empty_mask()) {  // first to claim
                    keep_word |= packed_bool_mask(j);
                  }
                }
              }
              keep_bitmap[i] = keep_word;
            });
          auto keep_flag_first = thrust::make_transform_iterator(
            thrust::make_counting_iterator(size_t{0}),
            cuda::proclaim_return_type<bool>(
              [keep_bitmap = raft::device_span<uint32_t const>(
                 keep_bitmap.data(), keep_bitmap.size())] __device__(size_t i) {
                auto word = keep_bitmap[packed_bool_offset(i)];
                return ((word & packed_bool_mask(i)) != packed_bool_empty_mask());
              }));
          if (uint32_key_output_offset) {
            thrust::scatter_if(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(rx_values),
              get_dataframe_buffer_begin(rx_values) + size_dataframe_buffer(rx_values),
              get_dataframe_buffer_begin(std::get<0>(rx_key_output_offsets)),
              keep_flag_first,
              tmp_vertex_value_output_first,
              cuda::std::identity{});
          } else {
            thrust::scatter_if(
              handle.get_thrust_policy(),
              get_dataframe_buffer_begin(rx_values),
              get_dataframe_buffer_begin(rx_values) + size_dataframe_buffer(rx_values),
              get_dataframe_buffer_begin(std::get<1>(rx_key_output_offsets)),
              keep_flag_first,
              tmp_vertex_value_output_first,
              cuda::std::identity{});
          }
        }

        handle.sync_stream();  // this is necessary to ensure the above update_device calls to
                               // finish before h_staging_buffer goes out-of-scope
      } else {
        device_group_start(minor_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          bool process = true;
          if constexpr (use_input_key) { process = nonzero_key_lists[j]; }
          if (process) {
            auto partition_idx = i + j;

            device_reduce(minor_comm,
                          get_dataframe_buffer_begin(edge_partition_major_output_buffers[j]),
                          tmp_vertex_value_output_first,
                          size_dataframe_buffer(edge_partition_major_output_buffers[j]),
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(partition_idx),
                          handle.get_stream());
          }
        }
        device_group_end(minor_comm);
        if (loop_stream_pool_indices) { handle.sync_stream(); }
      }
    }
  }

  // 10. communication

  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    auto& comm                 = handle.get_comms();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();

    auto view = minor_tmp_buffer->view();
    if (view.minor_keys()) {  // applying the initial value is deferred to here
      vertex_t max_vertex_partition_size{0};
      for (int i = 0; i < major_comm_size; ++i) {
        auto this_segment_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        max_vertex_partition_size =
          std::max(max_vertex_partition_size,
                   graph_view.vertex_partition_range_size(this_segment_vertex_partition_id));
      }
      auto tx_buffer = allocate_dataframe_buffer<T>(max_vertex_partition_size, handle.get_stream());
      auto tx_buffer_first = get_dataframe_buffer_begin(tx_buffer);
      std::optional<raft::host_span<vertex_t const>> minor_key_offsets{};
      if constexpr (GraphViewType::is_storage_transposed) {
        minor_key_offsets = graph_view.local_sorted_unique_edge_src_vertex_partition_offsets();
      } else {
        minor_key_offsets = graph_view.local_sorted_unique_edge_dst_vertex_partition_offsets();
      }
      for (int i = 0; i < major_comm_size; ++i) {
        auto minor_init = (major_comm_rank == i) ? init : ReduceOp::identity_element;
        auto this_segment_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        thrust::fill(handle.get_thrust_policy(),
                     tx_buffer_first,
                     tx_buffer_first +
                       graph_view.vertex_partition_range_size(this_segment_vertex_partition_id),
                     minor_init);
        auto value_first = thrust::make_transform_iterator(
          view.minor_value_first(),
          cuda::proclaim_return_type<T>(
            [reduce_op, minor_init] __device__(auto val) { return reduce_op(val, minor_init); }));
        thrust::scatter(handle.get_thrust_policy(),
                        value_first + (*minor_key_offsets)[i],
                        value_first + (*minor_key_offsets)[i + 1],
                        thrust::make_transform_iterator(
                          (*(view.minor_keys())).begin() + (*minor_key_offsets)[i],
                          cuda::proclaim_return_type<vertex_t>(
                            [key_first = graph_view.vertex_partition_range_first(
                               this_segment_vertex_partition_id)] __device__(auto key) {
                              return key - key_first;
                            })),
                        tx_buffer_first);
        device_reduce(major_comm,
                      tx_buffer_first,
                      tmp_vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.vertex_partition_range_size(this_segment_vertex_partition_id)),
                      ReduceOp::compatible_raft_comms_op,
                      i,
                      handle.get_stream());
      }
    } else {
      auto first_segment_vertex_partition_id =
        compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(0);
      vertex_t minor_range_first =
        graph_view.vertex_partition_range_first(first_segment_vertex_partition_id);
      for (int i = 0; i < major_comm_size; ++i) {
        auto this_segment_vertex_partition_id =
          compute_local_edge_partition_minor_range_vertex_partition_id_t{
            major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
        auto offset = graph_view.vertex_partition_range_first(this_segment_vertex_partition_id) -
                      minor_range_first;
        device_reduce(major_comm,
                      view.minor_value_first() + offset,
                      tmp_vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.vertex_partition_range_size(this_segment_vertex_partition_id)),
                      ReduceOp::compatible_raft_comms_op,
                      i,
                      handle.get_stream());
      }
    }
  }
}

}  // namespace detail

}  // namespace cugraph
