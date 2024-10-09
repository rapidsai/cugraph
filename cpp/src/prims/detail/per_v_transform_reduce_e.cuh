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

#include "detail/graph_partition_utils.cuh"
#include "prims/detail/multi_stream_utils.cuh"
#include "prims/detail/optional_dataframe_buffer.hpp"
#include "prims/detail/prim_functors.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/reduce_op.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/optional.h>
#include <thrust/scatter.h>
#include <thrust/set_operations.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

int32_t constexpr per_v_transform_reduce_e_kernel_block_size                        = 512;
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
          auto tmp = transform_op(i);
          val      = tmp;
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
      for (edge_t i = 0; i < local_degree; ++i) {
        if (pred_op(i)) {
          auto tmp = transform_op(i);
          if constexpr (std::is_same_v<ReduceOp,
                                       reduce_op::any<result_t>>) {  // init is selected only when
                                                                     // no edges return a valid
                                                                     // value
            val = tmp;
            break;
          } else {
            val = reduce_op(val, tmp);
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
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
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
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
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
    key_count = static_cast<size_t>(thrust::distance(key_first, key_last));
  } else {
    key_count = *(edge_partition.dcs_nzd_vertex_count());
  }

  while (idx < key_count) {
    key_t key{};
    vertex_t major{};
    thrust::optional<vertex_t> major_idx{};
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

      auto call_pred_op = init_pred_op<GraphViewType>(edge_partition,
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
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
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
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
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

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
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

    auto call_pred_op = init_pred_op<GraphViewType>(edge_partition,
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
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
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
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
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

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
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

    auto call_pred_op = init_pred_op<GraphViewType>(edge_partition,
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
                                        std::byte /* dummy */>
      first_valid_lane_id{};
    if constexpr (update_major) {
      reduced_e_op_result =
        (lane_id == 0) ? init : identity_element;  // init == identity_element for reduce_op::any<T>
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        first_valid_lane_id = raft::warp_size();
      }
    }

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
          raft::warp_size();
        for (size_t i = lane_id; i < rounded_up_local_degree; i += raft::warp_size()) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if ((i < static_cast<size_t>(local_degree)) &&
              (*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_lane_id = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                  .Reduce(e_op_result ? lane_id : raft::warp_size(), cub::Min());
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
          thrust::optional<T> e_op_result{thrust::nullopt};
          if (i < static_cast<size_t>(local_degree) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_lane_id = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                  .Reduce(e_op_result ? lane_id : raft::warp_size(), cub::Min());
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
          typename ResultValueOutputIteratorOrWrapper /* wrapper if update_major &&
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
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
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
                       std::byte /* dummy */>
      output_thread_id;

  while (idx < static_cast<size_t>(thrust::distance(key_first, key_last))) {
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

    auto call_pred_op = init_pred_op<GraphViewType>(edge_partition,
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
                                        std::byte /* dummy */>
      first_valid_thread_id{};
    if constexpr (update_major) {
      reduced_e_op_result = threadIdx.x == 0
                              ? init
                              : identity_element;  // init == identity_element for reduce_op::any<T>
      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        first_valid_thread_id = per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size;
      }
    }

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) +
            (per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size - 1)) /
           per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size) *
          per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if ((i < static_cast<size_t>(local_degree)) &&
              (*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result
                        ? threadIdx.x
                        : per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size,
                      cub::Min());
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
          thrust::optional<T> e_op_result{thrust::nullopt};
          if ((i < static_cast<size_t>(local_degree)) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result
                        ? threadIdx.x
                        : per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size,
                      cub::Min());
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

template <typename vertex_t, typename priority_t>
__host__ __device__ priority_t
rank_to_priority(int rank,
                 int root,
                 int subgroup_size /* faster interconnect within a subgroup */,
                 int comm_size,
                 vertex_t offset /* to evenly distribute traffic */)
{
  using cast_t =
    std::conditional_t<std::is_same_v<priority_t, uint8_t>,
                       int16_t,
                       std::conditional_t<std::is_same_v<priority_t, uint16_t>,
                                          int32_t,
                                          int64_t>>;  // to prevent overflow (assuming that
                                                      // comm_size <=
                                                      // std::numeric_limits<priority_t>::max())
  if (rank == root) {                                 // no need for communication (priority 0)
    return priority_t{0};
  } else if (rank / subgroup_size ==
             root / subgroup_size) {  // intra-subgroup communication is sufficient (priorities in
                                      // [1, subgroup_size)
    auto rank_dist =
      static_cast<int>(((static_cast<cast_t>(rank) + subgroup_size) - root) % subgroup_size);
    int modulo = subgroup_size - 1;
    return static_cast<priority_t>(1 + (static_cast<cast_t>(rank_dist - 1) + (offset % modulo)) %
                                         modulo);
  } else {  // inter-subgroup communication is necessary (priorities in [subgroup_size, comm_size)
    auto subgroup_dist =
      static_cast<int>(((static_cast<cast_t>(rank / subgroup_size) + (comm_size / subgroup_size)) -
                        (root / subgroup_size)) %
                       (comm_size / subgroup_size));
    auto intra_subgroup_rank_dist = static_cast<int>(
      ((static_cast<cast_t>(rank % subgroup_size) + subgroup_size) - (root % subgroup_size)) %
      subgroup_size);
    auto rank_dist = subgroup_dist * subgroup_size + intra_subgroup_rank_dist;
    int modulo     = comm_size - subgroup_size;
    return static_cast<priority_t>(
      subgroup_size +
      (static_cast<cast_t>(rank_dist - subgroup_size) + (offset % modulo)) % modulo);
  }
}

template <typename vertex_t, typename priority_t>
__host__ __device__ int priority_to_rank(
  priority_t priority,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  int comm_size,
  vertex_t offset /* to evenly distribute traffict */)
{
  using cast_t =
    std::conditional_t<std::is_same_v<priority_t, uint8_t>,
                       int16_t,
                       std::conditional_t<std::is_same_v<priority_t, uint16_t>,
                                          int32_t,
                                          int64_t>>;  // to prevent overflow (assuming that
                                                      // comm_size <=
                                                      // std::numeric_limits<priority_t>::max())
  if (priority == priority_t{0}) {
    return root;
  } else if (priority < static_cast<priority_t>(subgroup_size)) {
    int modulo     = subgroup_size - 1;
    auto rank_dist = static_cast<int>(
      1 + ((static_cast<cast_t>(priority - 1) + modulo) - (offset % modulo)) % modulo);
    return static_cast<int>((root - (root % subgroup_size)) +
                            ((static_cast<cast_t>(root) + rank_dist) % subgroup_size));
  } else {
    int modulo     = comm_size - subgroup_size;
    auto rank_dist = static_cast<int>(
      subgroup_size +
      ((static_cast<cast_t>(priority) - subgroup_size) + (modulo - (offset % modulo))) % modulo);
    auto subgroup_dist            = rank_dist / subgroup_size;
    auto intra_subgroup_rank_dist = rank_dist % subgroup_size;
    return static_cast<int>(
      ((static_cast<cast_t>((root / subgroup_size) * subgroup_size) +
        subgroup_dist * subgroup_size) +
       (static_cast<cast_t>(root) + intra_subgroup_rank_dist) % subgroup_size) %
      comm_size);
  }
}

template <typename vertex_t, typename priority_t, typename ValueIterator>
rmm::device_uvector<priority_t> compute_priorities(
  raft::comms::comms_t const& comm,
  ValueIterator value_first,
  std::optional<raft::device_span<size_t const>>
    hypersparse_key_offsets,  // we may not have values for the entire "range_size" if
                              // hypersparse_key_offsets.has_value() is true
  size_t contiguous_size,
  size_t range_size,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  typename thrust::iterator_traits<ValueIterator>::value_type init,
  bool ignore_local_values,
  rmm::cuda_stream_view stream_view)
{
  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  // For each vertex, select a comm_rank among the GPUs with a value other than init (if there are
  // more than one, the GPU with (comm_rank == root) has the highest priority, GPUs in the same DGX
  // node should be the next)

  rmm::device_uvector<priority_t> priorities(range_size, stream_view);

  if (ignore_local_values) {
    thrust::fill(rmm::exec_policy_nosync(stream_view),
                 priorities.begin(),
                 priorities.end(),
                 std::numeric_limits<priority_t>::max());
  } else {
    thrust::tabulate(
      rmm::exec_policy_nosync(stream_view),
      priorities.begin(),
      priorities.begin() + contiguous_size,
      [value_first, root, subgroup_size, init, comm_rank, comm_size] __device__(auto offset) {
        auto val = *(value_first + offset);
        return (val != init)
                 ? rank_to_priority<vertex_t, priority_t>(
                     comm_rank, root, subgroup_size, comm_size, static_cast<vertex_t>(offset))
                 : std::numeric_limits<priority_t>::max();  // lowest priority
      });
    if (hypersparse_key_offsets) {
      thrust::fill(rmm::exec_policy_nosync(stream_view),
                   priorities.begin() + contiguous_size,
                   priorities.end(),
                   std::numeric_limits<priority_t>::max());
      auto priority_first = thrust::make_transform_iterator(
        (*hypersparse_key_offsets).begin(),
        cuda::proclaim_return_type<priority_t>(
          [root, subgroup_size, comm_rank, comm_size] __device__(auto offset) {
            return rank_to_priority<vertex_t, priority_t>(
              comm_rank, root, subgroup_size, comm_size, static_cast<vertex_t>(offset));
          }));
      thrust::scatter_if(
        rmm::exec_policy_nosync(stream_view),
        priority_first,
        priority_first + (*hypersparse_key_offsets).size(),
        (*hypersparse_key_offsets).begin(),
        value_first + contiguous_size,
        priorities.begin(),
        is_not_equal_t<typename thrust::iterator_traits<ValueIterator>::value_type>{init});
    }
  }

  return priorities;
}

// return selected ranks if root.
// otherwise, it is sufficient to just return bool flags indiciating whether this rank's values are
// selected or not.
template <typename vertex_t, typename priority_t>
std::variant<rmm::device_uvector<int> /* root */, std::optional<rmm::device_uvector<uint32_t>>>
compute_selected_ranks_from_priorities(
  raft::comms::comms_t const& comm,
  raft::device_span<priority_t const> priorities,
  std::optional<raft::device_span<size_t const>>
    hypersparse_key_offsets,  // we may not have values for the entire "range_size" if
                              // hypersparse_key_offsets.has_value() is true
  size_t contiguous_size,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  bool ignore_local_values,
  rmm::cuda_stream_view stream_view)
{
  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  if (comm_rank == root) {
    rmm::device_uvector<int> selected_ranks(priorities.size(), stream_view);
    auto offset_priority_pair_first =
      thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), priorities.begin());
    thrust::transform(rmm::exec_policy_nosync(stream_view),
                      offset_priority_pair_first,
                      offset_priority_pair_first + priorities.size(),
                      selected_ranks.begin(),
                      [root, subgroup_size, comm_rank, comm_size] __device__(auto pair) {
                        auto offset   = thrust::get<0>(pair);
                        auto priority = thrust::get<1>(pair);
                        auto rank     = (priority == std::numeric_limits<priority_t>::max())
                                          ? comm_size
                                          : priority_to_rank<vertex_t, priority_t>(
                                          priority, root, subgroup_size, comm_size, offset);
                        return rank;
                      });
    return selected_ranks;
  } else {
    std::optional<rmm::device_uvector<uint32_t>> keep_flags{std::nullopt};
    if (!ignore_local_values) {
      keep_flags = rmm::device_uvector<uint32_t>(
        packed_bool_size(hypersparse_key_offsets
                           ? (contiguous_size + (*hypersparse_key_offsets).size())
                           : contiguous_size),
        stream_view);
      thrust::fill(rmm::exec_policy_nosync(stream_view),
                   (*keep_flags).begin(),
                   (*keep_flags).end(),
                   packed_bool_empty_mask());
      auto offset_priority_pair_first =
        thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), priorities.begin());
      thrust::for_each(
        rmm::exec_policy_nosync(stream_view),
        offset_priority_pair_first,
        offset_priority_pair_first + contiguous_size,
        [keep_flags = raft::device_span<uint32_t>((*keep_flags).data(), (*keep_flags).size()),
         root,
         subgroup_size,
         comm_rank,
         comm_size] __device__(auto pair) {
          auto offset   = thrust::get<0>(pair);
          auto priority = thrust::get<1>(pair);
          auto rank     = (priority == std::numeric_limits<priority_t>::max())
                            ? comm_size
                            : priority_to_rank<vertex_t, priority_t>(
                            priority, root, subgroup_size, comm_size, offset);
          if (rank == comm_rank) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
              keep_flags[packed_bool_offset(offset)]);
            word.fetch_or(packed_bool_mask(offset), cuda::std::memory_order_relaxed);
          }
        });
      if (hypersparse_key_offsets) {
        auto pair_first =
          thrust::make_zip_iterator(thrust::make_counting_iterator(size_t{contiguous_size}),
                                    (*hypersparse_key_offsets).begin());
        thrust::for_each(
          rmm::exec_policy_nosync(stream_view),
          pair_first,
          pair_first + (*hypersparse_key_offsets).size(),
          [priorities = raft::device_span<priority_t const>(priorities.data(), priorities.size()),
           keep_flags = raft::device_span<uint32_t>((*keep_flags).data(), (*keep_flags).size()),
           root,
           subgroup_size,
           comm_rank,
           comm_size] __device__(auto pair) {
            auto offset   = thrust::get<1>(pair);
            auto priority = priorities[offset];
            auto rank     = (priority == std::numeric_limits<priority_t>::max())
                              ? comm_size
                              : priority_to_rank<vertex_t, priority_t>(
                              priority, root, subgroup_size, comm_size, offset);
            if (rank == comm_rank) {
              cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
                keep_flags[packed_bool_offset(thrust::get<0>(pair))]);
              word.fetch_or(packed_bool_mask(thrust::get<0>(pair)),
                            cuda::std::memory_order_relaxed);
            }
          });
      }
    }
    return keep_flags;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename OptionalKeyIterator,  // invalid if void*
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
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
  EdgeSrcValueInputWrapper edge_partition_src_value_input,
  EdgeDstValueInputWrapper edge_partition_dst_value_input,
  EdgeValueInputWrapper edge_partition_e_value_input,
  thrust::optional<EdgePartitionEdgeMaskWrapper> edge_partition_e_mask,
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

  if (key_segment_offsets) {
    static_assert(detail::num_sparse_segments_per_vertex_partition == 3);

    if (edge_partition.dcs_nzd_vertex_count()) {
      auto exec_stream =
        edge_partition_stream_pool_indices
          ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[0])
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
      auto exec_stream =
        edge_partition_stream_pool_indices
          ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[1])
          : handle.get_stream();
      raft::grid_1d_thread_t update_grid((*key_segment_offsets)[3] - (*key_segment_offsets)[2],
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
      auto segment_output_buffer = output_buffer;
      if constexpr (update_major) { segment_output_buffer += (*key_segment_offsets)[2]; }
      segment_key_iterator_t segment_key_first{};
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      segment_key_first += (*key_segment_offsets)[2];
      detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          segment_key_first,
          segment_key_first + ((*key_segment_offsets)[3] - (*key_segment_offsets)[2]),
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
      auto exec_stream =
        edge_partition_stream_pool_indices
          ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[2])
          : handle.get_stream();
      raft::grid_1d_warp_t update_grid((*key_segment_offsets)[2] - (*key_segment_offsets)[1],
                                       detail::per_v_transform_reduce_e_kernel_block_size,
                                       handle.get_device_properties().maxGridSize[0]);
      auto segment_output_buffer = output_buffer;
      if constexpr (update_major) { segment_output_buffer += (*key_segment_offsets)[1]; }
      segment_key_iterator_t segment_key_first{};
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      segment_key_first += (*key_segment_offsets)[1];
      detail::per_v_transform_reduce_e_mid_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          segment_key_first,
          segment_key_first + ((*key_segment_offsets)[2] - (*key_segment_offsets)[1]),
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
      auto exec_stream =
        edge_partition_stream_pool_indices
          ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[3])
          : handle.get_stream();
      raft::grid_1d_block_t update_grid(
        (*key_segment_offsets)[1],
        std::is_same_v<ReduceOp, reduce_op::any<T>>
          ? detail::per_v_transform_reduce_e_kernel_high_degree_reduce_any_block_size
          : detail::per_v_transform_reduce_e_kernel_block_size,
        handle.get_device_properties().maxGridSize[0]);
      segment_key_iterator_t segment_key_first{};
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      detail::per_v_transform_reduce_e_high_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
          edge_partition,
          segment_key_first,
          segment_key_first + (*key_segment_offsets)[1],
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
    assert(!edge_partition_stream_pools);
    size_t num_keys{};
    if constexpr (use_input_key) {
      num_keys =
        static_cast<size_t>(thrust::distance(edge_partition_key_first, edge_partition_key_last));
    } else {
      num_keys = static_cast<size_t>(edge_partition.major_range_size());
    }

    if (num_keys > size_t{0}) {
      raft::grid_1d_thread_t update_grid(num_keys,
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
      segment_key_iterator_t segment_key_first{};
      if constexpr (use_input_key) {
        segment_key_first = edge_partition_key_first;
      } else {
        segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
      }
      detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
          edge_partition,
          segment_key_first,
          segment_key_first + num_keys,
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

#define PER_V_PERFORMANCE_MEASUREMENT 1  // FIXME: delete performance logging code

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
#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
#endif
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
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_endpoint_property_device_view_t<
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

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  constexpr bool try_bitmap = GraphViewType::is_multi_gpu &&
                              !std::is_same_v<OptionalKeyIterator, void*> &&
                              std::is_same_v<key_t, vertex_t>;

  [[maybe_unused]] constexpr auto max_segments =
    detail::num_sparse_segments_per_vertex_partition + size_t{1};

  // 1. drop zero degree keys & compute key_segment_offsets

  auto local_vertex_partition_segment_offsets = graph_view.local_vertex_partition_segment_offsets();

  std::conditional_t<use_input_key, std::optional<std::vector<size_t>>, std::byte /* dummy */>
    key_segment_offsets{};
  auto sorted_unique_nzd_key_last = sorted_unique_key_last;
  if constexpr (use_input_key) {
    if (local_vertex_partition_segment_offsets) {
      key_segment_offsets = compute_key_segment_offsets(
        sorted_unique_key_first,
        sorted_unique_nzd_key_last,
        raft::host_span<vertex_t const>((*local_vertex_partition_segment_offsets).data(),
                                        (*local_vertex_partition_segment_offsets).size()),
        graph_view.local_vertex_partition_range_first(),
        handle.get_stream());
      (*key_segment_offsets).back() = *((*key_segment_offsets).rbegin() + 1);
      sorted_unique_nzd_key_last    = sorted_unique_key_first + (*key_segment_offsets).back();
    }
  }

  // 2. initialize vertex value output buffer

  if constexpr (update_major) {  // no vertices in the zero degree segment are visited (otherwise,
                                 // no need to initialize)
    if constexpr (use_input_key) {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first +
                     thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
                   vertex_value_output_first +
                     thrust::distance(sorted_unique_key_first, sorted_unique_key_last),
                   init);
    } else {
      if (local_vertex_partition_segment_offsets) {
        thrust::fill(
          handle.get_thrust_policy(),
          vertex_value_output_first + *((*local_vertex_partition_segment_offsets).rbegin() + 1),
          vertex_value_output_first + *((*local_vertex_partition_segment_offsets).rbegin()),
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
#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1 = std::chrono::steady_clock::now();
#endif

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
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, static_cast<size_t>(minor_comm_rank))
        : thrust::nullopt;

    std::optional<std::vector<size_t>> edge_partition_stream_pool_indices{std::nullopt};
    if (local_vertex_partition_segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      edge_partition_stream_pool_indices = std::vector<size_t>(max_segments);
      std::iota((*edge_partition_stream_pool_indices).begin(),
                (*edge_partition_stream_pool_indices).end(),
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
                              (*key_segment_offsets).data(), (*key_segment_offsets).size())
                          : std::nullopt,
      edge_partition_stream_pool_indices ? std::make_optional<raft::host_span<size_t const>>(
                                             (*edge_partition_stream_pool_indices).data(),
                                             (*edge_partition_stream_pool_indices).size())
                                         : std::nullopt);

    if (edge_partition_stream_pool_indices) {
      handle.sync_stream_pool(*edge_partition_stream_pool_indices);
    }

    auto num_tmp_keys = thrust::count(
      handle.get_thrust_policy(),
      vertex_value_output_first,
      vertex_value_output_first +
        thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
      init);  // we allow false positives (some edge operations may actually return init)

    resize_optional_dataframe_buffer<key_t>(tmp_key_buffer, num_tmp_keys, handle.get_stream());
    resize_optional_dataframe_buffer<size_t>(tmp_output_indices, num_tmp_keys, handle.get_stream());

    auto input_first =
      thrust::make_zip_iterator(sorted_unique_key_first, thrust::make_counting_iterator(size_t{0}));
    thrust::copy_if(
      handle.get_thrust_policy(),
      input_first,
      input_first + thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
      vertex_value_output_first,
      thrust::make_zip_iterator(get_optional_dataframe_buffer_begin<key_t>(tmp_key_buffer),
                                get_optional_dataframe_buffer_begin<size_t>(tmp_output_indices)),
      is_equal_t<T>{init});

    sorted_unique_key_first       = get_optional_dataframe_buffer_begin<key_t>(tmp_key_buffer);
    sorted_unique_nzd_key_last    = get_optional_dataframe_buffer_end<key_t>(tmp_key_buffer);
    tmp_vertex_value_output_first = thrust::make_permutation_iterator(
      vertex_value_output_first, get_optional_dataframe_buffer_begin<size_t>(tmp_output_indices));

    if (key_segment_offsets) {
      key_segment_offsets = compute_key_segment_offsets(
        sorted_unique_key_first,
        sorted_unique_nzd_key_last,
        raft::host_span<vertex_t const>((*local_vertex_partition_segment_offsets).data(),
                                        (*local_vertex_partition_segment_offsets).size()),
        edge_partition.major_range_first(),
        handle.get_stream());
      assert((*key_segment_offsets).back() == *((*key_segment_offsets).rbegin() + 1));
      assert(sorted_uniue_nzd_key_last == sorted_unique_key_first + (*key_segment_offsets).back());
    }
  } else {
    tmp_vertex_value_output_first = vertex_value_output_first;
  }

  /* 4. compute subgroup_size (used to compute priority in device_gatherv) */
#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();
#endif

  [[maybe_unused]] std::conditional_t<GraphViewType::is_multi_gpu && update_major &&
                                        std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                      int,
                                      std::byte /* dummy */>
    subgroup_size{};
  if constexpr (GraphViewType::is_multi_gpu && update_major &&
                std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& comm           = handle.get_comms();
    auto const comm_size = comm.get_size();

    int num_gpus_per_node{};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
    if (comm_size <= num_gpus_per_node) {
      subgroup_size = comm_size;
    } else {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      subgroup_size              = partition_manager::map_major_comm_to_gpu_row_comm
                                     ? std::max(num_gpus_per_node / major_comm_size, int{1})
                                     : std::min(minor_comm_size, num_gpus_per_node);
    }
  }

  // 5. collect local_key_list_sizes & local_v_list_range_firsts & local_v_list_range_lasts &
  // key_segment_offset_vectors

  std::conditional_t<use_input_key, std::vector<size_t>, std::byte /* dummy */>
    local_key_list_sizes{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_v_list_range_firsts{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_v_list_range_lasts{};
  std::conditional_t<use_input_key && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                     std::optional<std::vector<size_t>>,
                     std::byte /* dummy */>
    local_key_list_deg1_sizes{};  // if global degree is 1, any valid local value should be selected
  std::conditional_t<use_input_key,
                     std::optional<std::vector<std::vector<size_t>>>,
                     std::byte /* dummy */>
    key_segment_offset_vectors{};
  if constexpr (use_input_key) {
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      auto const minor_comm_size = minor_comm.get_size();

      size_t num_scalars = 1;  // local_key_list_size
      if constexpr (try_bitmap) {
        num_scalars += 2;  // local_key_list_range_first, local_key_list_range_last
      }
      if (filter_input_key && graph_view.use_dcs()) {
        num_scalars += 1;  // local_key_list_degree_1_size
      }
      if (key_segment_offsets) { num_scalars += (*key_segment_offsets).size(); }

      rmm::device_uvector<size_t> d_aggregate_tmps(minor_comm_size * num_scalars,
                                                   handle.get_stream());
      auto hypersparse_degree_offsets =
        graph_view.local_vertex_partition_hypersparse_degree_offsets();
      thrust::tabulate(
        handle.get_thrust_policy(),
        d_aggregate_tmps.begin() + minor_comm_rank * num_scalars,
        d_aggregate_tmps.begin() + minor_comm_rank * num_scalars + (try_bitmap ? 3 : 1) +
          (filter_input_key && graph_view.use_dcs() ? 1 : 0),
        [sorted_unique_key_first,
         v_list_size = static_cast<size_t>(
           thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last)),
         deg1_v_first = (filter_input_key && graph_view.use_dcs())
                          ? thrust::make_optional(graph_view.local_vertex_partition_range_first() +
                                                  (*local_vertex_partition_segment_offsets)[3] +
                                                  *((*hypersparse_degree_offsets).rbegin() + 1))
                          : thrust::nullopt,
         vertex_partition_range_first =
           graph_view.local_vertex_partition_range_first()] __device__(size_t i) {
          if constexpr (try_bitmap) {
            if (i == 0) {
              return v_list_size;
            } else if (i == 1) {
              vertex_t first{};
              if (v_list_size > 0) {
                first = *sorted_unique_key_first;
              } else {
                first = vertex_partition_range_first;
              }
              assert(static_cast<vertex_t>(static_cast<size_t>(first)) == first);
              return static_cast<size_t>(first);
            }
            if (i == 2) {
              vertex_t last{};
              if (v_list_size > 0) {
                last = *(sorted_unique_key_first + (v_list_size - 1)) + 1;
              } else {
                last = vertex_partition_range_first;
              }
              assert(static_cast<vertex_t>(static_cast<size_t>(last)) == last);
              return static_cast<size_t>(last);
            } else {
              if (deg1_v_first) {
                auto sorted_unique_v_first = thrust::make_transform_iterator(
                  sorted_unique_key_first,
                  cuda::proclaim_return_type<vertex_t>([] __device__(auto key) {
                    return thrust_tuple_get_or_identity<key_t, 0>(key);
                  }));
                return v_list_size - static_cast<size_t>(thrust::distance(
                                       sorted_unique_v_first,
                                       thrust::lower_bound(thrust::seq,
                                                           sorted_unique_v_first,
                                                           sorted_unique_v_first + v_list_size,
                                                           deg1_v_first)));
              } else {
                assert(false);
                return size_t{0};
              }
            }
          } else {
            if (i == 0) {
              return v_list_size;
            } else {
              if (deg1_v_first) {
                auto sorted_unique_v_first = thrust::make_transform_iterator(
                  sorted_unique_key_first,
                  cuda::proclaim_return_type<vertex_t>([] __device__(auto key) {
                    return thrust_tuple_get_or_identity<key_t, 0>(key);
                  }));
                return v_list_size - static_cast<size_t>(thrust::distance(
                                       sorted_unique_v_first,
                                       thrust::lower_bound(thrust::seq,
                                                           sorted_unique_v_first,
                                                           sorted_unique_v_first + v_list_size,
                                                           deg1_v_first)));
              } else {
                assert(false);
                return size_t{0};
              }
            }
          }
        });
      if (key_segment_offsets) {
        raft::update_device(
          d_aggregate_tmps.data() + (minor_comm_rank * num_scalars + (try_bitmap ? 3 : 1) +
                                     (filter_input_key && graph_view.use_dcs() ? 1 : 0)),
          (*key_segment_offsets).data(),
          (*key_segment_offsets).size(),
          handle.get_stream());
      }

      if (minor_comm_size > 1) {
        device_allgather(minor_comm,
                         d_aggregate_tmps.data() + minor_comm_rank * num_scalars,
                         d_aggregate_tmps.data(),
                         num_scalars,
                         handle.get_stream());
      }

      std::vector<size_t> h_aggregate_tmps(d_aggregate_tmps.size());
      raft::update_host(h_aggregate_tmps.data(),
                        d_aggregate_tmps.data(),
                        d_aggregate_tmps.size(),
                        handle.get_stream());
      handle.sync_stream();
      local_key_list_sizes = std::vector<size_t>(minor_comm_size);
      if constexpr (try_bitmap) {
        local_v_list_range_firsts = std::vector<vertex_t>(minor_comm_size);
        local_v_list_range_lasts  = std::vector<vertex_t>(minor_comm_size);
      }
      if constexpr (filter_input_key) {
        if (graph_view.use_dcs()) {
          local_key_list_deg1_sizes = std::vector<size_t>(minor_comm_size);
        }
      }
      if (key_segment_offsets) {
        key_segment_offset_vectors = std::vector<std::vector<size_t>>{};
        (*key_segment_offset_vectors).reserve(minor_comm_size);
      }
      for (int i = 0; i < minor_comm_size; ++i) {
        local_key_list_sizes[i] = h_aggregate_tmps[i * num_scalars];
        if constexpr (try_bitmap) {
          local_v_list_range_firsts[i] =
            static_cast<vertex_t>(h_aggregate_tmps[i * num_scalars + 1]);
          local_v_list_range_lasts[i] =
            static_cast<vertex_t>(h_aggregate_tmps[i * num_scalars + 2]);
        }
        if constexpr (filter_input_key) {
          if (graph_view.use_dcs()) {
            (*local_key_list_deg1_sizes)[i] =
              static_cast<vertex_t>(h_aggregate_tmps[i * num_scalars + (try_bitmap ? 3 : 1)]);
          }
        }
        if (key_segment_offsets) {
          (*key_segment_offset_vectors)
            .emplace_back(h_aggregate_tmps.begin() + i * num_scalars + (try_bitmap ? 3 : 1) +
                            ((filter_input_key && graph_view.use_dcs()) ? 1 : 0),
                          h_aggregate_tmps.begin() + i * num_scalars + (try_bitmap ? 3 : 1) +
                            ((filter_input_key && graph_view.use_dcs()) ? 1 : 0) +
                            (*key_segment_offsets).size());
        }
      }
    } else {
      local_key_list_sizes = std::vector<size_t>{
        static_cast<size_t>(thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last))};
      if (key_segment_offsets) {
        key_segment_offset_vectors       = std::vector<std::vector<size_t>>(1);
        (*key_segment_offset_vectors)[0] = *key_segment_offsets;
      }
    }
  }

  // 6. compute optional bitmap info & compressed vertex list

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

      bool v_compressible{false};
      if constexpr (sizeof(vertex_t) > sizeof(uint32_t)) {
        vertex_t local_v_list_max_range_size{0};
        for (int i = 0; i < minor_comm_size; ++i) {
          auto range_size             = local_v_list_range_lasts[i] - local_v_list_range_firsts[i];
          local_v_list_max_range_size = std::max(range_size, local_v_list_max_range_size);
        }
        if (local_v_list_max_range_size <=
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
      if (avg_fill_ratio > threshold_ratio) {
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
    }
  }

  // 7. set-up stream pool

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    if (local_vertex_partition_segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      size_t tmp_buffer_size_per_loop{0};
      if constexpr (update_major) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();

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
        size_t approx_tmp_size{0};
        if constexpr (filter_input_key) {
          // use tmeporary buffers to store non-zero local degree key offsets in the hypersparse
          // regioon, priorities, selected ranks (or)  flags (non-root), and selected values (and
          // key offsets for the selected values that are in the hypersparse region and have the
          // global degree of 1)
          approx_tmp_size = static_cast<size_t>(
            static_cast<double>(sizeof(size_t)) * 0.25 +
            static_cast<double>(value_size) /
              static_cast<double>(minor_comm_size) /* only one value will be selected */);
        }

        size_t aggregate_major_range_size{};
        if constexpr (use_input_key) {
          aggregate_major_range_size =
            std::reduce(local_key_list_sizes.begin(), local_key_list_sizes.end());
        } else {
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            if constexpr (GraphViewType::is_storage_transposed) {
              aggregate_major_range_size += graph_view.local_edge_partition_dst_range_size(i);
            } else {
              aggregate_major_range_size += graph_view.local_edge_partition_src_range_size(i);
            }
          }
        }
        tmp_buffer_size_per_loop =
          (aggregate_major_range_size / graph_view.number_of_local_edge_partitions()) *
          (key_size + value_size + approx_tmp_size);
      }

      stream_pool_indices = init_stream_pool_indices(
        static_cast<size_t>(static_cast<double>(handle.get_device_properties().totalGlobalMem) *
                            0.2),
        tmp_buffer_size_per_loop,
        graph_view.number_of_local_edge_partitions(),
        max_segments,
        handle.get_stream_pool_size());
      if ((*stream_pool_indices).size() <= 1) { stream_pool_indices = std::nullopt; }
    }
  }

  // 8. set-up temporary buffers

  size_t num_concurrent_loops{1};
  if (stream_pool_indices) {
    assert(((*stream_pool_indices).size() % max_segments) == 0);
    num_concurrent_loops = (*stream_pool_indices).size() / max_segments;
  }

  using minor_tmp_buffer_type = std::conditional_t<GraphViewType::is_storage_transposed,
                                                   edge_src_property_t<GraphViewType, T>,
                                                   edge_dst_property_t<GraphViewType, T>>;
  [[maybe_unused]] std::unique_ptr<minor_tmp_buffer_type> minor_tmp_buffer{};
  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    minor_tmp_buffer = std::make_unique<minor_tmp_buffer_type>(handle, graph_view);
    auto minor_init  = init;
    auto view        = minor_tmp_buffer->view();
    if (view.keys()) {  // defer applying the initial value to the end as minor_tmp_buffer ma not
                        // store values for the entire minor rangey
      minor_init = ReduceOp::identity_element;
    } else {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_rank = major_comm.get_rank();
      minor_init                 = (major_comm_rank == 0) ? init : ReduceOp::identity_element;
    }
    fill_edge_minor_property(handle, graph_view, minor_tmp_buffer->mutable_view(), minor_init);
  }

  using edge_partition_minor_output_device_view_t =
    std::conditional_t<GraphViewType::is_multi_gpu && !update_major,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         decltype(minor_tmp_buffer->mutable_view().value_first())>,
                       void /* dummy */>;

  if constexpr (!GraphViewType::is_multi_gpu || !use_input_key) {
    if (stream_pool_indices) { handle.sync_stream(); }
  }

  // 9. proces local edge partitions

#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time3 = std::chrono::steady_clock::now();
#endif
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i += num_concurrent_loops) {
#if PER_V_PERFORMANCE_MEASUREMENT
    auto subtime0 = std::chrono::steady_clock::now();
    auto subtime1 = std::chrono::steady_clock::now();
    auto subtime2 = std::chrono::steady_clock::now();
#endif
    auto loop_count =
      std::min(num_concurrent_loops, graph_view.number_of_local_edge_partitions() - i);

    std::conditional_t<GraphViewType::is_multi_gpu && use_input_key,
                       std::vector<dataframe_buffer_type_t<key_t>>,
                       std::byte /* dummy */>
      edge_partition_key_buffers{};
    std::conditional_t<filter_input_key,
                       std::optional<std::vector<rmm::device_uvector<size_t>>>,
                       std::byte /* dummy */>
      edge_partition_hypersparse_key_offset_vectors{};  // drop zero local degree keys in the
                                                        // hypersparse region
    std::conditional_t<filter_input_key, std::optional<std::vector<size_t>>, std::byte /* dummy */>
      edge_partition_deg1_hypersparse_key_offset_counts{};
    if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();
      auto const minor_comm_rank = minor_comm.get_rank();

      edge_partition_key_buffers.reserve(loop_count);

      std::conditional_t<std::is_same_v<key_t, vertex_t>,
                         std::optional<std::vector<rmm::device_uvector<uint32_t>>>,
                         std::byte /* dummy */>
        edge_partition_tmp_bcast_buffers{std::nullopt};
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        if (v_list_bitmap || compressed_v_list) {
          edge_partition_tmp_bcast_buffers = std::vector<rmm::device_uvector<uint32_t>>{};
          (*edge_partition_tmp_bcast_buffers).reserve(loop_count);
        }
      }

      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx        = i + j;
        bool use_tmp_bcast_buffer = false;
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          if (edge_partition_tmp_bcast_buffers) {
            (*edge_partition_tmp_bcast_buffers)
              .emplace_back(v_list_bitmap
                              ? packed_bool_size(local_v_list_range_lasts[partition_idx] -
                                                 local_v_list_range_firsts[partition_idx])
                              : local_key_list_sizes[partition_idx],
                            handle.get_stream());
            use_tmp_bcast_buffer = true;
          }
        }
        if (!use_tmp_bcast_buffer) {
          edge_partition_key_buffers.push_back(allocate_dataframe_buffer<key_t>(
            local_key_list_sizes[partition_idx], handle.get_stream()));
        }
      }
#if PER_V_PERFORMANCE_MEASUREMENT
      handle.sync_stream();
      subtime1 = std::chrono::steady_clock::now();
#endif

      device_group_start(minor_comm);
      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx        = i + j;
        bool use_tmp_bcast_buffer = false;
        if constexpr (std::is_same_v<key_t, vertex_t>) {
          if (v_list_bitmap) {
            device_bcast(minor_comm,
                         (*v_list_bitmap).data(),
                         get_dataframe_buffer_begin((*edge_partition_tmp_bcast_buffers)[j]),
                         size_dataframe_buffer((*edge_partition_tmp_bcast_buffers)[j]),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
            use_tmp_bcast_buffer = true;
          } else if (compressed_v_list) {
            device_bcast(minor_comm,
                         (*compressed_v_list).data(),
                         get_dataframe_buffer_begin((*edge_partition_tmp_bcast_buffers)[j]),
                         size_dataframe_buffer((*edge_partition_tmp_bcast_buffers)[j]),
                         static_cast<int>(partition_idx),
                         handle.get_stream());
            use_tmp_bcast_buffer = true;
          }
        }
        if (!use_tmp_bcast_buffer) {
          device_bcast(minor_comm,
                       sorted_unique_key_first,
                       get_dataframe_buffer_begin(edge_partition_key_buffers[j]),
                       local_key_list_sizes[partition_idx],
                       static_cast<int>(partition_idx),
                       handle.get_stream());
        }
      }
      device_group_end(minor_comm);
      if (stream_pool_indices) { handle.sync_stream(); }
#if PER_V_PERFORMANCE_MEASUREMENT
      subtime2 = std::chrono::steady_clock::now();
#endif

      if constexpr (std::is_same_v<key_t, vertex_t>) {
        if (edge_partition_tmp_bcast_buffers) {
          // copy data (in the sparse region first if filter_input_key is true) to
          // edge_partition_key_buffers[j]

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream   = stream_pool_indices
                                   ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                   : handle.get_stream();

            bool process_local_edges{true};
            if constexpr (filter_input_key) {
              process_local_edges = (static_cast<int>(partition_idx) != minor_comm_rank);
            }

            auto keys = allocate_dataframe_buffer<key_t>(
              process_local_edges ? local_key_list_sizes[partition_idx] : size_t{0}, loop_stream);
            if (process_local_edges) {
              auto range_first = local_v_list_range_firsts[partition_idx];
              if (v_list_bitmap) {
                auto range_last = local_v_list_range_lasts[partition_idx];
                if constexpr (filter_input_key) {
                  if (graph_view.use_dcs()) {  // skip copying the hypersparse segment
                    auto edge_partition =
                      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                        graph_view.local_edge_partition_view(partition_idx));
                    range_first =
                      std::min(range_first, *(edge_partition.major_hypersparse_first()));
                    range_last = std::min(range_last, *(edge_partition.major_hypersparse_first()));
                  }
                }
                auto const& rx_tmps = (*edge_partition_tmp_bcast_buffers)[j];
                rmm::device_scalar<size_t> dummy(size_t{0}, loop_stream);
                retrieve_vertex_list_from_bitmap(
                  raft::device_span<uint32_t const>(rx_tmps.data(), rx_tmps.size()),
                  get_dataframe_buffer_begin(keys),
                  raft::device_span<size_t>(dummy.data(), size_t{1}),
                  range_first,
                  range_last,
                  loop_stream);
              } else {
                assert(compressed_v_list);
                auto const& rx_tmps = (*edge_partition_tmp_bcast_buffers)[j];
                auto input_first    = rx_tmps.begin();
                auto input_last     = rx_tmps.end();
                if constexpr (filter_input_key) {
                  if (graph_view.use_dcs()) {  // skip copying the hypersparse segment
                    auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];
                    input_last                      = input_first + key_segment_offsets[3];
                  }
                }
                thrust::transform(
                  rmm::exec_policy_nosync(loop_stream),
                  input_first,
                  input_last,
                  get_dataframe_buffer_begin(keys),
                  cuda::proclaim_return_type<vertex_t>([range_first] __device__(uint32_t v_offset) {
                    return static_cast<vertex_t>(range_first + v_offset);
                  }));
              }
            }
            edge_partition_key_buffers.push_back(std::move(keys));
          }
        }
      }

      if constexpr (filter_input_key) {
        if (graph_view.use_dcs()) {
          edge_partition_hypersparse_key_offset_vectors =
            std::vector<rmm::device_uvector<size_t>>{};
          (*edge_partition_hypersparse_key_offset_vectors).reserve(loop_count);
          edge_partition_deg1_hypersparse_key_offset_counts = std::vector<size_t>(loop_count);

          std::vector<rmm::device_scalar<size_t>> edge_partition_count_scalars{};
          edge_partition_count_scalars.reserve(loop_count);

          std::optional<std::vector<dataframe_buffer_type_t<key_t>>>
            edge_partition_tmp_key_buffers{};
          bool direct_copy = false;  /// directly copy to edge_partition_key_buffers[]
          if constexpr (std::is_same_v<key_t, vertex_t>) {
            if (edge_partition_tmp_bcast_buffers) { direct_copy = true; }
          }
          if (!direct_copy) {  // copy the hypersparse keys to a temporary key buffer first
            edge_partition_tmp_key_buffers = std::vector<dataframe_buffer_type_t<key_t>>{};
            (*edge_partition_tmp_key_buffers).reserve(loop_count);
          }

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream   = stream_pool_indices
                                   ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                   : handle.get_stream();

            auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

            auto process_local_edges = (static_cast<int>(partition_idx) != minor_comm_rank);

            auto& keys   = edge_partition_key_buffers[j];
            auto offsets = rmm::device_uvector<size_t>(
              process_local_edges ? (key_segment_offsets[4] - key_segment_offsets[3]) : vertex_t{0},
              loop_stream);
            rmm::device_scalar<size_t> count(size_t{0}, loop_stream);
            auto tmp_keys = allocate_dataframe_buffer<key_t>(
              edge_partition_tmp_key_buffers ? (key_segment_offsets[4] - key_segment_offsets[3])
                                             : vertex_t{0},
              loop_stream);

            if (process_local_edges) {
              auto edge_partition =
                edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
                  graph_view.local_edge_partition_view(partition_idx));
              auto const& segment_offsets =
                graph_view.local_edge_partition_segment_offsets(partition_idx);

#if 1
              auto segment_bitmap = *(edge_partition.dcs_nzd_range_bitmap());
#else
              // FIXME: we can pre-compute this & store in graph_t
              rmm::device_uvector<uint32_t> segment_bitmap(
                packed_bool_size((*segment_offsets)[4] - (*segment_offsets)[3]), loop_stream);
              thrust::fill(rmm::exec_policy_nosync(loop_stream),
                           segment_bitmap.begin(),
                           segment_bitmap.end(),
                           packed_bool_empty_mask());
              thrust::for_each(
                rmm::exec_policy_nosync(loop_stream),
                *(edge_partition.dcs_nzd_vertices()),
                *(edge_partition.dcs_nzd_vertices()) + *(edge_partition.dcs_nzd_vertex_count()),
                [bitmap = raft::device_span<uint32_t>(segment_bitmap.data(), segment_bitmap.size()),
                 major_hypersparse_first =
                   *(edge_partition.major_hypersparse_first())] __device__(auto major) {
                  auto major_offset = major - major_hypersparse_first;
                  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> word(
                    bitmap[packed_bool_offset(major_offset)]);
                  word.fetch_or(packed_bool_mask(major_offset), cuda::std::memory_order_relaxed);
                });
#endif

              auto range_offset_first = std::min(
                (edge_partition.major_range_first() + (*segment_offsets)[3] >
                 local_v_list_range_firsts[partition_idx])
                  ? ((edge_partition.major_range_first() + (*segment_offsets)[3]) -
                     local_v_list_range_firsts[partition_idx])
                  : vertex_t{0},
                local_v_list_range_lasts[partition_idx] - local_v_list_range_firsts[partition_idx]);

              if constexpr (std::is_same_v<key_t, vertex_t>) {
                if (edge_partition_tmp_bcast_buffers) {
                  auto const& rx_tmps = (*edge_partition_tmp_bcast_buffers)[j];
                  if (v_list_bitmap) {
                    auto range_offset_last =
                      std::min(((edge_partition.major_range_first() + (*segment_offsets)[4]) >
                                local_v_list_range_firsts[partition_idx])
                                 ? ((edge_partition.major_range_first() + (*segment_offsets)[4]) -
                                    local_v_list_range_firsts[partition_idx])
                                 : vertex_t{0},
                               local_v_list_range_lasts[partition_idx] -
                                 local_v_list_range_firsts[partition_idx]);
                    auto count_first = thrust::make_transform_iterator(
                      thrust::make_counting_iterator(packed_bool_offset(range_offset_first)),
                      cuda::proclaim_return_type<vertex_t>(
                        [range_bitmap =
                           raft::device_span<uint32_t const>(rx_tmps.data(), rx_tmps.size()),
                         range_offset_first] __device__(size_t i) {
                          auto word = range_bitmap[i];
                          if (i == packed_bool_offset(range_offset_first)) {
                            word &= ~packed_bool_partial_mask(
                              range_offset_first %
                              packed_bools_per_word());  // exclude the bits in the sparse region
                          }
                          return static_cast<vertex_t>(__popc(word));
                        }));
                    rmm::device_uvector<vertex_t> count_displacements(
                      rx_tmps.size() - packed_bool_offset(range_offset_first), loop_stream);
                    thrust::exclusive_scan(rmm::exec_policy_nosync(loop_stream),
                                           count_first,
                                           count_first + count_displacements.size(),
                                           count_displacements.begin());
                    auto offset_first = thrust::make_transform_iterator(
                      thrust::make_counting_iterator(range_offset_first),
                      cuda::proclaim_return_type<size_t>(
                        [range_bitmap =
                           raft::device_span<uint32_t const>(rx_tmps.data(), rx_tmps.size()),
                         count_displacements = raft::device_span<vertex_t const>(
                           count_displacements.data(), count_displacements.size()),
                         range_offset_first,
                         start_offset = key_segment_offsets[3]] __device__(auto range_offset) {
                          auto word = range_bitmap[packed_bool_offset(range_offset)];
                          if (packed_bool_offset(range_offset) ==
                              packed_bool_offset(range_offset_first)) {
                            word &= ~packed_bool_partial_mask(
                              range_offset_first %
                              packed_bools_per_word());  // exclude the bits in the sparse region
                          }
                          return static_cast<size_t>(
                            start_offset +
                            count_displacements[packed_bool_offset(range_offset) -
                                                packed_bool_offset(range_offset_first)] +
                            __popc(word & packed_bool_partial_mask(range_offset %
                                                                   packed_bools_per_word())));
                        }));
                    auto input_pair_first = thrust::make_zip_iterator(
                      thrust::make_counting_iterator(local_v_list_range_firsts[partition_idx] +
                                                     range_offset_first),
                      offset_first);
                    detail::copy_if_nosync(
                      input_pair_first,
                      input_pair_first + (range_offset_last - range_offset_first),
                      thrust::make_transform_iterator(
                        thrust::make_counting_iterator(range_offset_first),
                        cuda::proclaim_return_type<bool>(
                          [range_bitmap =
                             raft::device_span<uint32_t const>(rx_tmps.data(), rx_tmps.size()),
                           segment_bitmap = raft::device_span<uint32_t const>(
                             segment_bitmap.data(), segment_bitmap.size()),
                           range_first = local_v_list_range_firsts[partition_idx],
                           major_hypersparse_first =
                             *(edge_partition
                                 .major_hypersparse_first())] __device__(auto range_offset) {
                            auto segment_offset =
                              (range_first + range_offset) - major_hypersparse_first;
                            return ((range_bitmap[packed_bool_offset(range_offset)] &
                                     packed_bool_mask(range_offset)) != packed_bool_empty_mask()) &&
                                   ((segment_bitmap[packed_bool_offset(segment_offset)] &
                                     packed_bool_mask(segment_offset)) != packed_bool_empty_mask());
                          })),
                      thrust::make_zip_iterator(
                        get_dataframe_buffer_begin(keys) + key_segment_offsets[3], offsets.begin()),
                      raft::device_span<size_t>(count.data(), size_t{1}),
                      loop_stream);
                  } else {
                    assert(compressed_v_list);
                    auto input_pair_first = thrust::make_zip_iterator(
                      thrust::make_transform_iterator(
                        rx_tmps.begin() + key_segment_offsets[3],
                        cuda::proclaim_return_type<vertex_t>(
                          [range_first =
                             local_v_list_range_firsts[partition_idx]] __device__(auto v_offset) {
                            return static_cast<vertex_t>(range_first + v_offset);
                          })),
                      thrust::make_counting_iterator(static_cast<size_t>(key_segment_offsets[3])));
                    detail::copy_if_nosync(
                      input_pair_first,
                      input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                      thrust::make_transform_iterator(
                        rx_tmps.begin() + key_segment_offsets[3],
                        cuda::proclaim_return_type<bool>(
                          [segment_bitmap = raft::device_span<uint32_t const>(
                             segment_bitmap.data(), segment_bitmap.size()),
                           range_first             = local_v_list_range_firsts[partition_idx],
                           major_hypersparse_first = *(
                             edge_partition.major_hypersparse_first())] __device__(auto v_offset) {
                            auto segment_offset =
                              (range_first + v_offset) - major_hypersparse_first;
                            return ((segment_bitmap[packed_bool_offset(segment_offset)] &
                                     packed_bool_mask(segment_offset)) != packed_bool_empty_mask());
                          })),
                      thrust::make_zip_iterator(
                        get_dataframe_buffer_begin(keys) + key_segment_offsets[3], offsets.begin()),
                      raft::device_span<size_t>(count.data(), size_t{1}),
                      loop_stream);
                  }
                }
              }
              if (edge_partition_tmp_key_buffers) {
                auto input_pair_first =
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(keys),
                                            thrust::make_counting_iterator(size_t{0})) +
                  key_segment_offsets[3];
                detail::copy_if_nosync(
                  input_pair_first,
                  input_pair_first + (key_segment_offsets[4] - key_segment_offsets[3]),
                  thrust::make_transform_iterator(
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
                      })),
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(tmp_keys), offsets.begin()),
                  raft::device_span<size_t>(count.data(), size_t{1}),
                  loop_stream);
              }
            }

            (*edge_partition_hypersparse_key_offset_vectors).push_back(std::move(offsets));
            edge_partition_count_scalars.push_back(std::move(count));
            if (edge_partition_tmp_key_buffers) {
              (*edge_partition_tmp_key_buffers).push_back(std::move(tmp_keys));
            }
          }
          if (edge_partition_tmp_bcast_buffers) { (*edge_partition_tmp_bcast_buffers).clear(); }

          std::vector<size_t> h_counts(loop_count);
          for (size_t j = 0; j < loop_count; ++j) {
            auto loop_stream = stream_pool_indices
                                 ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                 : handle.get_stream();
            h_counts[j]      = edge_partition_count_scalars[j].value(loop_stream);
          }

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream   = stream_pool_indices
                                   ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                   : handle.get_stream();

            auto process_local_edges = (static_cast<int>(partition_idx) != minor_comm_rank);
            if (process_local_edges) {
              auto& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

              if (edge_partition_tmp_key_buffers) {
                auto const& tmp_keys = (*edge_partition_tmp_key_buffers)[j];
                thrust::copy(rmm::exec_policy_nosync(loop_stream),
                             get_dataframe_buffer_begin(tmp_keys),
                             get_dataframe_buffer_begin(tmp_keys) + h_counts[j],
                             get_dataframe_buffer_begin(edge_partition_key_buffers[j]) +
                               key_segment_offsets[3]);
              }
              resize_dataframe_buffer(
                edge_partition_key_buffers[j], key_segment_offsets[3] + h_counts[j], loop_stream);
              (*edge_partition_hypersparse_key_offset_vectors)[j].resize(h_counts[j], loop_stream);
              // skip shrink_to_fit to cut execution time
              (*edge_partition_deg1_hypersparse_key_offset_counts)[j] =
                size_dataframe_buffer((*edge_partition_hypersparse_key_offset_vectors)[j]) -
                static_cast<size_t>(thrust::distance(
                  get_dataframe_buffer_begin((*edge_partition_hypersparse_key_offset_vectors)[j]),
                  thrust::lower_bound(
                    rmm::exec_policy_nosync(loop_stream),
                    get_dataframe_buffer_begin((*edge_partition_hypersparse_key_offset_vectors)[j]),
                    get_dataframe_buffer_end((*edge_partition_hypersparse_key_offset_vectors)[j]),
                    local_key_list_sizes[partition_idx] -
                      (*local_key_list_deg1_sizes)[partition_idx])));
            }
          }
        }
      }
    }
#if PER_V_PERFORMANCE_MEASUREMENT
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
    auto subtime3 = std::chrono::steady_clock::now();
#endif

    std::conditional_t<GraphViewType::is_multi_gpu && update_major,
                       std::vector<dataframe_buffer_type_t<T>>,
                       std::byte /* dummy */>
      edge_partition_major_output_buffers{};
    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      edge_partition_major_output_buffers.reserve(loop_count);
    }
    for (size_t j = 0; j < loop_count; ++j) {
      auto partition_idx = i + j;
      auto loop_stream   = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                             : handle.get_stream();

      if constexpr (GraphViewType::is_multi_gpu && update_major) {
        size_t buffer_size{0};
        if constexpr (use_input_key) {
          buffer_size = size_dataframe_buffer(edge_partition_key_buffers[j]);
        } else {
          auto edge_partition =
            edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
              graph_view.local_edge_partition_view(partition_idx));
          auto const& segment_offsets =
            graph_view.local_edge_partition_segment_offsets(partition_idx);

          buffer_size = segment_offsets
                          ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                          : edge_partition.major_range_size();
        }
        edge_partition_major_output_buffers.push_back(
          allocate_dataframe_buffer<T>(buffer_size, loop_stream));
      }
    }
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if PER_V_PERFORMANCE_MEASUREMENT
    auto subtime4 = std::chrono::steady_clock::now();
#endif

    for (size_t j = 0; j < loop_count; ++j) {
      auto partition_idx = i + j;

      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(partition_idx));
      auto edge_partition_e_mask =
        edge_mask_view
          ? thrust::make_optional<
              detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
              *edge_mask_view, partition_idx)
          : thrust::nullopt;
      auto edge_partition_stream_pool_indices =
        stream_pool_indices ? std::make_optional<raft::host_span<size_t const>>(
                                (*stream_pool_indices).data() + j * max_segments, max_segments)
                            : std::nullopt;

      T major_init{};
      T major_identity_element{};
      if constexpr (update_major) {
        if constexpr (std::is_same_v<ReduceOp,
                                     reduce_op::any<T>>) {  // if any edge has a non-init value, one
                                                            // of the non-init values will be
                                                            // selected.
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

      auto edge_partition_key_first = sorted_unique_key_first;
      auto edge_partition_key_last  = sorted_unique_nzd_key_last;
      if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();
        if (minor_comm_size > 1) {
          edge_partition_key_first = get_dataframe_buffer_begin(edge_partition_key_buffers[j]);
          edge_partition_key_last  = get_dataframe_buffer_end(edge_partition_key_buffers[j]);
        }
      }

      std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
      if constexpr (use_input_key) {
        if (key_segment_offset_vectors) {
          key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];
          if constexpr (filter_input_key) {
            if (edge_partition_hypersparse_key_offset_vectors) {
              (*key_segment_offsets).back() = static_cast<size_t>(
                thrust::distance(edge_partition_key_first, edge_partition_key_last));
              *((*key_segment_offsets).rbegin() + 1) = (*key_segment_offsets).back();
            }
          }
        }
      } else {
        auto const& segment_offsets =
          graph_view.local_edge_partition_segment_offsets(partition_idx);
        if (segment_offsets) {
          key_segment_offsets = std::vector<size_t>((*segment_offsets).size());
          std::transform((*segment_offsets).begin(),
                         (*segment_offsets).end(),
                         (*key_segment_offsets).begin(),
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

      bool process_local_edges = true;
      if constexpr (filter_input_key) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        if (static_cast<int>(partition_idx) == minor_comm_rank) { process_local_edges = false; }
      }

      if (process_local_edges) {
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
                                  (*key_segment_offsets).data(), (*key_segment_offsets).size())
                              : std::nullopt,
          edge_partition_stream_pool_indices);
      }
    }
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if PER_V_PERFORMANCE_MEASUREMENT
    auto subtime5 = std::chrono::steady_clock::now();
#endif

    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      auto const minor_comm_size = minor_comm.get_size();

      if constexpr (use_input_key) {
        if (minor_comm_size > 1) {
          edge_partition_key_buffers.clear();
          edge_partition_key_buffers.shrink_to_fit();
        }
      }

      if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        std::vector<std::variant<rmm::device_uvector<uint8_t>,
                                 rmm::device_uvector<uint16_t>,
                                 rmm::device_uvector<uint32_t>>>
          edge_partition_priorities{};
        edge_partition_priorities.reserve(loop_count);

        std::conditional_t<filter_input_key,
                           std::optional<std::vector<raft::device_span<size_t const>>>,
                           std::byte /* dummy */>
          edge_partition_hypersparse_non_deg1_key_offset_spans{};
        if constexpr (filter_input_key) {
          if (edge_partition_hypersparse_key_offset_vectors) {
            edge_partition_hypersparse_non_deg1_key_offset_spans =
              std::vector<raft::device_span<size_t const>>(loop_count);
          }
        }

        std::vector<size_t> edge_partition_allreduce_sizes(loop_count);
        std::vector<size_t> edge_partition_contiguous_sizes(loop_count);

        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          auto loop_stream   = stream_pool_indices
                                 ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                 : handle.get_stream();

          bool process_local_edges = true;
          if constexpr (filter_input_key) {
            if (static_cast<int>(partition_idx) == minor_comm_rank) { process_local_edges = false; }
          }

          std::optional<raft::device_span<size_t const>> hypersparse_non_deg1_key_offsets{
            std::nullopt};
          if constexpr (filter_input_key) {
            if (edge_partition_hypersparse_key_offset_vectors) {
              hypersparse_non_deg1_key_offsets = raft::device_span<size_t const>(
                (*edge_partition_hypersparse_key_offset_vectors)[j].data(),
                (*edge_partition_hypersparse_key_offset_vectors)[j].size() -
                  (edge_partition_deg1_hypersparse_key_offset_counts
                     ? (*edge_partition_deg1_hypersparse_key_offset_counts)[j]
                     : size_t{0}));
              (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j] =
                *hypersparse_non_deg1_key_offsets;
            }
          }

          auto const& output_buffer = edge_partition_major_output_buffers[j];

          size_t allreduce_size{};
          size_t contiguous_size{};
          if constexpr (filter_input_key) {
            allreduce_size = local_key_list_sizes[partition_idx];
            if (local_key_list_deg1_sizes) {
              allreduce_size -= (*local_key_list_deg1_sizes)[partition_idx];
            }
            if (key_segment_offset_vectors) {
              auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];
              contiguous_size                 = key_segment_offsets[3];
            } else {
              contiguous_size = local_key_list_sizes[partition_idx];
            }
          } else {
            static_assert(!use_input_key);
            auto hypersparse_degree_offsets =
              graph_view.local_edge_partition_hypersparse_degree_offsets(partition_idx);
            allreduce_size = size_dataframe_buffer(output_buffer);
            if (hypersparse_degree_offsets) {
              allreduce_size -= *((*hypersparse_degree_offsets).rbegin()) -
                                *((*hypersparse_degree_offsets).rbegin() + 1);
            }
            contiguous_size = size_dtaframe_buffer(output_buffer);
          }
          edge_partition_allreduce_sizes[j]  = allreduce_size;
          edge_partition_contiguous_sizes[j] = contiguous_size;

          std::variant<rmm::device_uvector<uint8_t>,
                       rmm::device_uvector<uint16_t>,
                       rmm::device_uvector<uint32_t>>
            priorities = rmm::device_uvector<uint8_t>(0, loop_stream);
          if (minor_comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority == uint8_t
            priorities = compute_priorities<vertex_t, uint8_t>(
              minor_comm,
              get_dataframe_buffer_begin(output_buffer),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              allreduce_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              init,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
          } else if (minor_comm_size <=
                     std::numeric_limits<uint16_t>::max()) {  // priority == uint16_t
            priorities = compute_priorities<vertex_t, uint16_t>(
              minor_comm,
              get_dataframe_buffer_begin(output_buffer),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              allreduce_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              init,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
          } else {  // priority == uint32_t
            priorities = compute_priorities<vertex_t, uint32_t>(
              minor_comm,
              get_dataframe_buffer_begin(output_buffer),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              allreduce_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              init,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
          }
          edge_partition_priorities.push_back(std::move(priorities));
        }
        if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if PER_V_PERFORMANCE_MEASUREMENT
        auto subtime6 = std::chrono::steady_clock::now();
#endif

        device_group_start(minor_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          auto& priorities = edge_partition_priorities[j];
          if (minor_comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority == uint8_t
            device_allreduce(minor_comm,
                             std::get<0>(priorities).data(),
                             std::get<0>(priorities).data(),
                             std::get<0>(priorities).size(),
                             raft::comms::op_t::MIN,
                             handle.get_stream());
          } else if (minor_comm_size <=
                     std::numeric_limits<uint16_t>::max()) {  // priority == uint16_t
            CUGRAPH_FAIL(
              "unimplemented.");  // currently, raft does not support allreduce on uint16_t.
          } else {                // priority == uint32_t
            device_allreduce(minor_comm,
                             std::get<2>(priorities).data(),
                             std::get<2>(priorities).data(),
                             std::get<2>(priorities).size(),
                             raft::comms::op_t::MIN,
                             handle.get_stream());
          }
        }
        device_group_end(minor_comm);
        if (stream_pool_indices) { handle.sync_stream(); }
#if PER_V_PERFORMANCE_MEASUREMENT
        auto subtime7 = std::chrono::steady_clock::now();
#endif

        std::vector<
          std::variant<rmm::device_uvector<int>, std::optional<rmm::device_uvector<uint32_t>>>>
          edge_partition_selected_ranks_or_flags{};
        edge_partition_selected_ranks_or_flags.reserve(loop_count);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          auto loop_stream   = stream_pool_indices
                                 ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                 : handle.get_stream();

          bool process_local_edges = true;
          if constexpr (filter_input_key) {
            if (static_cast<int>(partition_idx) == minor_comm_rank) { process_local_edges = false; }
          }

          auto const& output_buffer = edge_partition_major_output_buffers[j];
          std::optional<raft::device_span<size_t const>> hypersparse_non_deg1_key_offsets{
            std::nullopt};
          if constexpr (filter_input_key) {
            if (edge_partition_hypersparse_key_offset_vectors) {
              hypersparse_non_deg1_key_offsets =
                (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j];
            }
          }

          auto contiguous_size = edge_partition_contiguous_sizes[j];

          auto& priorities = edge_partition_priorities[j];
          std::variant<rmm::device_uvector<int>, std::optional<rmm::device_uvector<uint32_t>>>
            selected_ranks_or_flags = rmm::device_uvector<int>(0, loop_stream);
          if (minor_comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority == uint8_t
            selected_ranks_or_flags = compute_selected_ranks_from_priorities<vertex_t, uint8_t>(
              minor_comm,
              raft::device_span<uint8_t const>(std::get<0>(priorities).data(),
                                               std::get<0>(priorities).size()),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
            std::get<0>(priorities).resize(0, loop_stream);
            std::get<0>(priorities).shrink_to_fit(loop_stream);
          } else if (minor_comm_size <=
                     std::numeric_limits<uint16_t>::max()) {  // priority == uint16_t
            selected_ranks_or_flags = compute_selected_ranks_from_priorities<vertex_t, uint16_t>(
              minor_comm,
              raft::device_span<uint16_t const>(std::get<1>(priorities).data(),
                                                std::get<1>(priorities).size()),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
            std::get<1>(priorities).resize(0, loop_stream);
            std::get<1>(priorities).shrink_to_fit(loop_stream);
          } else {  // priority_t == uint32_t
            selected_ranks_or_flags = compute_selected_ranks_from_priorities<vertex_t, uint32_t>(
              minor_comm,
              raft::device_span<uint32_t const>(std::get<2>(priorities).data(),
                                                std::get<2>(priorities).size()),
              hypersparse_non_deg1_key_offsets,
              contiguous_size,
              static_cast<int>(partition_idx),
              subgroup_size,
              process_local_edges ? false : true /* ignore_local_values */,
              loop_stream);
            std::get<2>(priorities).resize(0, loop_stream);
            std::get<2>(priorities).shrink_to_fit(loop_stream);
          }
          edge_partition_selected_ranks_or_flags.push_back(std::move(selected_ranks_or_flags));
        }
#if PER_V_PERFORMANCE_MEASUREMENT
        if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
        auto subtime8 = std::chrono::steady_clock::now();
#endif

        std::vector<dataframe_buffer_type_t<T>> edge_partition_values{};
        edge_partition_values.reserve(loop_count);
        std::vector<rmm::device_scalar<size_t>> edge_partition_count_scalars{};
        edge_partition_count_scalars.reserve(loop_count);

        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          auto loop_stream   = stream_pool_indices
                                 ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                 : handle.get_stream();

          bool process_local_edges = true;
          if constexpr (filter_input_key) {
            if (static_cast<int>(partition_idx) == minor_comm_rank) { process_local_edges = false; }
          }

          auto& output_buffer = edge_partition_major_output_buffers[j];

          auto values =
            allocate_dataframe_buffer<T>(size_dataframe_buffer(output_buffer), loop_stream);
          rmm::device_scalar<size_t> count(size_t{0}, loop_stream);
          if (minor_comm_rank == static_cast<int>(partition_idx)) {
            if (process_local_edges) {
              assert(!use_input_key);
              assert(edge_partition_selected_ranks_or_flags[j].index() == 0);
              auto const& selected_ranks = std::get<0>(edge_partition_selected_ranks_or_flags[j]);
              copy_if_nosync(
                get_dataframe_buffer_begin(output_buffer),
                get_dataframe_buffer_begin(output_buffer) + edge_partition_allreduce_sizes[j],
                thrust::make_transform_iterator(
                  selected_ranks.begin(),
                  cuda::proclaim_return_type<bool>(
                    [minor_comm_rank] __device__(auto rank) { return rank == minor_comm_rank; })),
                get_dataframe_buffer_begin(values),
                raft::device_span<size_t>(count.data(), size_t{1}),
                loop_stream);
            }
          } else {
            assert(edge_partition_selected_ranks_or_flags[j].index() == 1);
            auto& keep_flags = std::get<1>(edge_partition_selected_ranks_or_flags[j]);
            size_t input_end_offset{};
            if constexpr (filter_input_key) {
              input_end_offset =
                edge_partition_contiguous_sizes[j] +
                (edge_partition_hypersparse_non_deg1_key_offset_spans
                   ? (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j].size()
                   : size_t{0});
            } else {
              input_end_offset = edge_partition_allreduce_sizes[j];
            }
            copy_if_nosync(
              get_dataframe_buffer_begin(output_buffer),
              get_dataframe_buffer_begin(output_buffer) + input_end_offset,
              thrust::make_transform_iterator(
                thrust::make_counting_iterator(size_t{0}),
                cuda::proclaim_return_type<bool>(
                  [keep_flags = raft::device_span<uint32_t const>(
                     (*keep_flags).data(), (*keep_flags).size())] __device__(size_t offset) {
                    auto word = keep_flags[packed_bool_offset(offset)];
                    return ((word & packed_bool_mask(offset)) != packed_bool_empty_mask());
                  })),
              get_dataframe_buffer_begin(values),
              raft::device_span<size_t>(count.data(), size_t{1}),
              loop_stream);
            (*keep_flags).resize(0, loop_stream);
            (*keep_flags).shrink_to_fit(loop_stream);
          }

          edge_partition_values.push_back(std::move(values));
          edge_partition_count_scalars.push_back(std::move(count));
        }

        std::vector<size_t> copy_sizes(loop_count);
        for (size_t j = 0; j < loop_count; ++j) {
          auto loop_stream = stream_pool_indices
                               ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                               : handle.get_stream();
          copy_sizes[j]    = edge_partition_count_scalars[j].value(loop_stream);
        }

        std::optional<std::variant<std::vector<rmm::device_uvector<uint32_t>>,
                                   std::vector<rmm::device_uvector<size_t>>>>
          edge_partition_deg1_hypersparse_output_offset_vectors{};

        if (graph_view.use_dcs()) {
          size_t max_output_range_size{0};
          if constexpr (filter_input_key) {
            max_output_range_size = std::reduce(local_key_list_sizes.begin(),
                                                local_key_list_sizes.end(),
                                                size_t{0},
                                                [](auto l, auto r) { return std::max(l, r); });
          } else {
            for (size_t j = 0; j < loop_count; ++j) {
              auto& output_buffer = edge_partition_major_output_buffers[j];
              max_output_range_size =
                std::max(size_dataframe_buffer(output_buffer), max_output_range_size);
            }
          }
          if (max_output_range_size < static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            edge_partition_deg1_hypersparse_output_offset_vectors =
              std::vector<rmm::device_uvector<uint32_t>>{};
            std::get<0>(*edge_partition_deg1_hypersparse_output_offset_vectors).reserve(loop_count);
          } else {
            edge_partition_deg1_hypersparse_output_offset_vectors =
              std::vector<rmm::device_uvector<size_t>>{};
            std::get<1>(*edge_partition_deg1_hypersparse_output_offset_vectors).reserve(loop_count);
          }

          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto loop_stream   = stream_pool_indices
                                   ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                   : handle.get_stream();

            bool process_local_edges = true;
            if constexpr (filter_input_key) {
              if (static_cast<int>(partition_idx) == minor_comm_rank) {
                process_local_edges = false;
              }
            }

            auto& output_buffer = edge_partition_major_output_buffers[j];
            auto& values        = edge_partition_values[j];
            auto& count         = edge_partition_count_scalars[j];

            size_t output_offset_buf_size{0};
            if constexpr (filter_input_key) {
              output_offset_buf_size = (*edge_partition_deg1_hypersparse_key_offset_counts)[j];
            } else {
              assert(!use_input_key);
              output_offset_buf_size =
                size_dataframe_buffer(output_buffer) - edge_partition_allreduce_sizes[j];
            }

            std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>
              output_offsets = rmm::device_uvector<uint32_t>(0, loop_stream);
            if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
              std::get<0>(output_offsets).resize(output_offset_buf_size, loop_stream);
            } else {
              output_offsets = rmm::device_uvector<size_t>(output_offset_buf_size, loop_stream);
            }

            size_t input_start_offset{};
            if constexpr (filter_input_key) {
              input_start_offset =
                edge_partition_contiguous_sizes[j] +
                (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j].size();
            } else {
              static_assert(!use_input_key);
              input_start_offset = edge_partition_allreduce_sizes[j];
            }
            auto flag_first = thrust::make_transform_iterator(
              get_dataframe_buffer_begin(output_buffer) + input_start_offset,
              cuda::proclaim_return_type<bool>(
                [init] __device__(auto val) { return val != init; }));

            if constexpr (filter_input_key) {
              assert(static_cast<int>(partition_idx) != minor_comm_rank);
              auto& hypersparse_key_offsets = (*edge_partition_hypersparse_key_offset_vectors)[j];
              if (output_offsets.index() == 0) {
                auto input_pair_first = thrust::make_zip_iterator(
                  get_dataframe_buffer_begin(output_buffer) + input_start_offset,
                  thrust::make_transform_iterator(
                    hypersparse_key_offsets.begin() +
                      (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j].size(),
                    typecast_t<size_t, uint32_t>{}));
                copy_if_nosync(
                  input_pair_first,
                  input_pair_first + (*edge_partition_deg1_hypersparse_key_offset_counts)[j],
                  flag_first,
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(values) + copy_sizes[j],
                                            std::get<0>(output_offsets).begin()),
                  raft::device_span<size_t>(count.data(), size_t{1}),
                  loop_stream);
              } else {
                auto input_pair_first = thrust::make_zip_iterator(
                  get_dataframe_buffer_begin(output_buffer) + input_start_offset,
                  hypersparse_key_offsets.begin() +
                    (*edge_partition_hypersparse_non_deg1_key_offset_spans)[j].size());
                copy_if_nosync(
                  input_pair_first,
                  input_pair_first + (*edge_partition_deg1_hypersparse_key_offset_counts)[j],
                  flag_first,
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(values) + copy_sizes[j],
                                            std::get<1>(output_offsets).begin()),
                  raft::device_span<size_t>(count.data(), size_t{1}),
                  loop_stream);
              }
              hypersparse_key_offsets.resize(0, loop_stream);
              hypersparse_key_offsets.shrink_to_fit(loop_stream);
            } else {
              static_assert(!use_input_key);
              assert(process_local_edges);
              if (output_offsets.index() == 0) {
                auto input_pair_first =
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(output_buffer),
                                            thrust::make_counting_iterator(uint32_t{0}));
                copy_if_nosync(
                  input_pair_first + input_start_offset,
                  input_pair_first + size_dataframe_buffer(output_buffer),
                  flag_first,
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(values) + copy_sizes[j],
                                            std::get<0>(output_offsets).begin()),
                  raft::device_span<size_t>(count.data(), size_t{1}),
                  loop_stream);
              } else {
                auto input_pair_first =
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(output_buffer),
                                            thrust::make_counting_iterator(size_t{0}));
                copy_if_nosync(
                  input_pair_first + input_start_offset,
                  input_pair_first + size_dataframe_buffer(output_buffer),
                  flag_first,
                  thrust::make_zip_iterator(get_dataframe_buffer_begin(values) + copy_sizes[j],
                                            std::get<1>(output_offsets).begin()),
                  raft::device_span<size_t>(count.data(), size_t{1}),
                  loop_stream);
              }
            }

            if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
              std::get<0>(*edge_partition_deg1_hypersparse_output_offset_vectors)
                .push_back(std::move(std::get<0>(output_offsets)));
            } else {
              assert((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 1);
              std::get<1>(*edge_partition_deg1_hypersparse_output_offset_vectors)
                .push_back(std::move(std::get<1>(output_offsets)));
            }

            resize_dataframe_buffer(output_buffer, 0, loop_stream);
            shrink_to_fit_dataframe_buffer(output_buffer, loop_stream);
          }

          for (size_t j = 0; j < loop_count; ++j) {
            auto loop_stream    = stream_pool_indices
                                    ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                                    : handle.get_stream();
            auto deg1_copy_size = edge_partition_count_scalars[j].value(loop_stream);
            copy_sizes[j] += deg1_copy_size;
            if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
              std::get<0>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j].resize(
                deg1_copy_size, loop_stream);
            } else {
              assert((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 1);
              std::get<1>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j].resize(
                deg1_copy_size, loop_stream);
            }
            // skip shrink_to_fit() to cut execution time
          }
        }

        for (size_t j = 0; j < loop_count; ++j) {
          auto loop_stream = stream_pool_indices
                               ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                               : handle.get_stream();
          resize_dataframe_buffer(edge_partition_values[j], copy_sizes[j], loop_stream);
          // skip shrink_to_fit() to cut execution time
        }
        if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if PER_V_PERFORMANCE_MEASUREMENT
        auto subtime9 = std::chrono::steady_clock::now();
#endif

        std::optional<std::vector<size_t>> rx_value_sizes{};
        std::optional<std::vector<size_t>> rx_value_displs{};
        std::optional<dataframe_buffer_type_t<T>> rx_values{};

        std::optional<std::vector<size_t>> rx_offset_sizes{};
        std::optional<std::vector<size_t>> rx_offset_displs{};
        std::optional<std::variant<rmm::device_uvector<uint32_t>, rmm::device_uvector<size_t>>>
          rx_offsets{};
        {
          auto size_per_rank = loop_count * (graph_view.use_dcs() ? 2 : 1);
          rmm::device_uvector<size_t> d_aggregate_buffer_sizes(minor_comm_size * size_per_rank,
                                                               handle.get_stream());
          std::vector<size_t> h_buffer_sizes(size_per_rank);
          for (size_t j = 0; j < loop_count; ++j) {
            h_buffer_sizes[j] = size_dataframe_buffer(edge_partition_values[j]);
            if (graph_view.use_dcs()) {
              if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
                h_buffer_sizes[loop_count + j] =
                  std::get<0>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j].size();
              } else {
                assert((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 1);
                h_buffer_sizes[loop_count + j] =
                  std::get<1>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j].size();
              }
            }
          }
          raft::update_device(d_aggregate_buffer_sizes.data() + minor_comm_rank * size_per_rank,
                              h_buffer_sizes.data(),
                              h_buffer_sizes.size(),
                              handle.get_stream());
          device_allgather(minor_comm,
                           d_aggregate_buffer_sizes.data() + minor_comm_rank * size_per_rank,
                           d_aggregate_buffer_sizes.data(),
                           size_per_rank,
                           handle.get_stream());
          if (static_cast<size_t>(minor_comm_rank / num_concurrent_loops) ==
              (i / num_concurrent_loops)) {
            std::vector<size_t> h_aggregate_buffer_sizes(d_aggregate_buffer_sizes.size());
            raft::update_host(h_aggregate_buffer_sizes.data(),
                              d_aggregate_buffer_sizes.data(),
                              d_aggregate_buffer_sizes.size(),
                              handle.get_stream());
            handle.sync_stream();
            auto j          = static_cast<size_t>(minor_comm_rank % num_concurrent_loops);
            rx_value_sizes  = std::vector<size_t>(minor_comm_size);
            rx_value_displs = std::vector<size_t>(minor_comm_size);
            if (graph_view.use_dcs()) {
              rx_offset_sizes  = std::vector<size_t>(minor_comm_size);
              rx_offset_displs = std::vector<size_t>(minor_comm_size);
            }
            for (int k = 0; k < minor_comm_size; ++k) {
              (*rx_value_sizes)[k] = h_aggregate_buffer_sizes[k * size_per_rank + j];
              if (graph_view.use_dcs()) {
                (*rx_offset_sizes)[k] =
                  h_aggregate_buffer_sizes[k * size_per_rank + loop_count + j];
              }
            }
            std::exclusive_scan((*rx_value_sizes).begin(),
                                (*rx_value_sizes).end(),
                                (*rx_value_displs).begin(),
                                size_t{0});
            std::exclusive_scan((*rx_offset_sizes).begin(),
                                (*rx_offset_sizes).end(),
                                (*rx_offset_displs).begin(),
                                size_t{0});
            rx_values = allocate_dataframe_buffer<T>(
              (*rx_value_displs).back() + (*rx_value_sizes).back(), handle.get_stream());
            if (graph_view.use_dcs()) {
              if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
                rx_offsets = rmm::device_uvector<uint32_t>(
                  (*rx_offset_displs).back() + (*rx_offset_sizes).back(), handle.get_stream());
              } else {
                assert((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 1);
                rx_offsets = rmm::device_uvector<size_t>(
                  (*rx_offset_displs).back() + (*rx_offset_sizes).back(), handle.get_stream());
              }
            }
          }
        }
#if PER_V_PERFORMANCE_MEASUREMENT
        handle.sync_stream();
        auto subtime10 = std::chrono::steady_clock::now();
#endif

        device_group_start(minor_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;
          auto& values       = edge_partition_values[j];

          if (minor_comm_rank == static_cast<int>(partition_idx)) {
            device_gatherv(minor_comm,
                           get_dataframe_buffer_begin(values),
                           get_dataframe_buffer_begin(*rx_values),
                           values.size(),
                           *rx_value_sizes,
                           *rx_value_displs,
                           static_cast<int>(partition_idx),
                           handle.get_stream());
          } else {
            device_gatherv(minor_comm,
                           get_dataframe_buffer_begin(values),
                           dataframe_buffer_iterator_type_t<T>{},
                           values.size(),
                           std::vector<size_t>{},
                           std::vector<size_t>{},
                           static_cast<int>(partition_idx),
                           handle.get_stream());
          }
        }
        device_group_end(minor_comm);
        if (graph_view.use_dcs()) {
          device_group_start(minor_comm);
          for (size_t j = 0; j < loop_count; ++j) {
            auto partition_idx = i + j;
            auto& values       = edge_partition_values[j];

            if ((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 0) {
              auto& offsets =
                std::get<0>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j];
              if (minor_comm_rank == static_cast<int>(partition_idx)) {
                device_gatherv(minor_comm,
                               offsets.data(),
                               std::get<0>(*rx_offsets).data(),
                               offsets.size(),
                               *rx_offset_sizes,
                               *rx_offset_displs,
                               static_cast<int>(partition_idx),
                               handle.get_stream());
              } else {
                device_gatherv(minor_comm,
                               offsets.data(),
                               static_cast<uint32_t*>(nullptr),
                               offsets.size(),
                               std::vector<size_t>{},
                               std::vector<size_t>{},
                               static_cast<int>(partition_idx),
                               handle.get_stream());
              }
            } else {
              assert((*edge_partition_deg1_hypersparse_output_offset_vectors).index() == 1);
              auto& offsets =
                std::get<1>(*edge_partition_deg1_hypersparse_output_offset_vectors)[j];
              if (minor_comm_rank == static_cast<int>(partition_idx)) {
                device_gatherv(minor_comm,
                               offsets.data(),
                               std::get<1>(*rx_offsets).data(),
                               offsets.size(),
                               *rx_offset_sizes,
                               *rx_offset_displs,
                               static_cast<int>(partition_idx),
                               handle.get_stream());
              } else {
                device_gatherv(minor_comm,
                               offsets.data(),
                               static_cast<size_t*>(nullptr),
                               offsets.size(),
                               std::vector<size_t>{},
                               std::vector<size_t>{},
                               static_cast<int>(partition_idx),
                               handle.get_stream());
              }
            }
          }
          device_group_end(minor_comm);
        }
        handle.sync_stream();  // this is required before edge_partition_values.clear();
        edge_partition_values.clear();
        if (stream_pool_indices) {
          handle.sync_stream_pool(*stream_pool_indices);
        }  // to ensure that memory is freed
#if PER_V_PERFORMANCE_MEASUREMENT
        auto subtime11 = std::chrono::steady_clock::now();
#endif

        if (rx_values && (size_dataframe_buffer(*rx_values) > 0)) {
          auto j             = static_cast<size_t>(minor_comm_rank % num_concurrent_loops);
          auto partition_idx = i + j;

          auto& selected_ranks = std::get<0>(edge_partition_selected_ranks_or_flags[j]);
          auto old_size        = selected_ranks.size();
          size_t output_range_size{};
          if constexpr (filter_input_key) {
            output_range_size = local_key_list_sizes[partition_idx];
          } else {
            auto const& segment_offsets = graph_view.local_vertex_partition_segment_offsets();
            output_range_size =
              segment_offsets
                ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                : graph_view.local_vertex_partition_range_size();
          }
          selected_ranks.resize(output_range_size, handle.get_stream());
          thrust::fill(handle.get_thrust_policy(),
                       selected_ranks.begin() + old_size,
                       selected_ranks.end(),
                       minor_comm_size);
          if (rx_offsets) {
            rmm::device_uvector<size_t> lasts((*rx_offset_displs).size(), handle.get_stream());
            raft::update_device(lasts.data(),
                                (*rx_offset_displs).data() + 1,
                                (*rx_offset_displs).size() - 1,
                                handle.get_stream());
            auto num_elements = (*rx_offset_displs).back() + (*rx_offset_sizes).back();
            lasts.set_element_async(lasts.size() - 1, num_elements, handle.get_stream());

            if ((*rx_offsets).index() == 0) {
              auto& offsets = std::get<0>(*rx_offsets);
              thrust::for_each(
                handle.get_thrust_policy(),
                thrust::make_counting_iterator(size_t{0}),
                thrust::make_counting_iterator(offsets.size()),
                [offsets        = raft::device_span<uint32_t const>(offsets.data(), offsets.size()),
                 lasts          = raft::device_span<size_t const>(lasts.data(), lasts.size()),
                 selected_ranks = raft::device_span<int>(
                   selected_ranks.data(), selected_ranks.size())] __device__(auto i) {
                  auto minor_comm_rank       = static_cast<int>(thrust::distance(
                    lasts.begin(),
                    thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), i)));
                  selected_ranks[offsets[i]] = minor_comm_rank;
                });
              offsets.resize(0, handle.get_stream());
              offsets.shrink_to_fit(handle.get_stream());
            } else {
              assert((*rx_offsets).index() == 1);
              auto& offsets = std::get<1>(*rx_offsets);
              thrust::for_each(
                handle.get_thrust_policy(),
                thrust::make_counting_iterator(size_t{0}),
                thrust::make_counting_iterator(offsets.size()),
                [offsets        = raft::device_span<size_t const>(offsets.data(), offsets.size()),
                 lasts          = raft::device_span<size_t const>(lasts.data(), lasts.size()),
                 selected_ranks = raft::device_span<int>(
                   selected_ranks.data(), selected_ranks.size())] __device__(auto i) {
                  auto minor_comm_rank       = static_cast<int>(thrust::distance(
                    lasts.begin(),
                    thrust::upper_bound(thrust::seq, lasts.begin(), lasts.end(), i)));
                  selected_ranks[offsets[i]] = minor_comm_rank;
                });
              offsets.resize(0, handle.get_stream());
              offsets.shrink_to_fit(handle.get_stream());
            }
          }

          // FIXME: we may use 8 bit ranks to further cut sort time
          if (selected_ranks.size() <= std::numeric_limits<uint32_t>::max()) {
            rmm::device_uvector<uint32_t> rx_positions(selected_ranks.size(), handle.get_stream());
            thrust::sequence(
              handle.get_thrust_policy(), rx_positions.begin(), rx_positions.end(), uint32_t{0});
            thrust::stable_sort_by_key(handle.get_thrust_policy(),
                                       selected_ranks.begin(),
                                       selected_ranks.end(),
                                       rx_positions.begin());
            // selected_ranks[] == minor_comm_size if no GPU in minor_comm has a non-init value
            rx_positions.resize((*rx_value_displs).back() + (*rx_value_sizes).back(),
                                handle.get_stream());
            thrust::scatter(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(*rx_values),
                            get_dataframe_buffer_end(*rx_values),
                            rx_positions.begin(),
                            tmp_vertex_value_output_first);
          } else {
            rmm::device_uvector<size_t> rx_positions(selected_ranks.size(), handle.get_stream());
            thrust::sequence(
              handle.get_thrust_policy(), rx_positions.begin(), rx_positions.end(), size_t{0});
            thrust::stable_sort_by_key(handle.get_thrust_policy(),
                                       selected_ranks.begin(),
                                       selected_ranks.end(),
                                       rx_positions.begin());
            // selected_ranks[] == minor_comm_size if no GPU in minor_comm has a non-init value
            rx_positions.resize((*rx_value_displs).back() + (*rx_value_sizes).back(),
                                handle.get_stream());
            thrust::scatter(handle.get_thrust_policy(),
                            get_dataframe_buffer_begin(*rx_values),
                            get_dataframe_buffer_end(*rx_values),
                            rx_positions.begin(),
                            tmp_vertex_value_output_first);
          }
        }
        handle.sync_stream();
#if PER_V_PERFORMANCE_MEASUREMENT
        auto subtime12                         = std::chrono::steady_clock::now();
        std::chrono::duration<double> subdur0  = subtime1 - subtime0;
        std::chrono::duration<double> subdur1  = subtime2 - subtime1;
        std::chrono::duration<double> subdur2  = subtime3 - subtime2;
        std::chrono::duration<double> subdur3  = subtime4 - subtime3;
        std::chrono::duration<double> subdur4  = subtime5 - subtime4;
        std::chrono::duration<double> subdur5  = subtime6 - subtime5;
        std::chrono::duration<double> subdur6  = subtime7 - subtime6;
        std::chrono::duration<double> subdur7  = subtime8 - subtime7;
        std::chrono::duration<double> subdur8  = subtime9 - subtime8;
        std::chrono::duration<double> subdur9  = subtime10 - subtime9;
        std::chrono::duration<double> subdur10 = subtime11 - subtime10;
        std::chrono::duration<double> subdur11 = subtime12 - subtime11;
        std::cerr << "sub (per_v) took (" << subdur0.count() << "," << subdur1.count() << ","
                  << subdur2.count() << "," << subdur3.count() << "," << subdur4.count() << ","
                  << subdur5.count() << "," << subdur6.count() << "," << subdur7.count() << ","
                  << subdur8.count() << "," << subdur9.count() << "," << subdur10.count() << ","
                  << subdur11.count() << ")" << std::endl;
#endif
      } else {
        device_group_start(minor_comm);
        for (size_t j = 0; j < loop_count; ++j) {
          auto partition_idx = i + j;

          device_reduce(minor_comm,
                        get_dataframe_buffer_begin(edge_partition_major_output_buffers[j]),
                        tmp_vertex_value_output_first,
                        size_dataframe_buffer(edge_partition_major_output_buffers[j]),
                        ReduceOp::compatible_raft_comms_op,
                        static_cast<int>(partition_idx),
                        handle.get_stream());
        }
        device_group_end(minor_comm);
        if (stream_pool_indices) { handle.sync_stream(); }
      }
    }
  }
#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time4 = std::chrono::steady_clock::now();
#endif

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
    if (view.keys()) {  // applying the initial value is deferred to here
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
          view.value_first(),
          cuda::proclaim_return_type<T>(
            [reduce_op, minor_init] __device__(auto val) { return reduce_op(val, minor_init); }));
        thrust::scatter(handle.get_thrust_policy(),
                        value_first + (*minor_key_offsets)[i],
                        value_first + (*minor_key_offsets)[i + 1],
                        thrust::make_transform_iterator(
                          (*(view.keys())).begin() + (*minor_key_offsets)[i],
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
                      view.value_first() + offset,
                      tmp_vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.vertex_partition_range_size(this_segment_vertex_partition_id)),
                      ReduceOp::compatible_raft_comms_op,
                      i,
                      handle.get_stream());
      }
    }
  }
#if PER_V_PERFORMANCE_MEASUREMENT
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time5                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::chrono::duration<double> dur1 = time2 - time1;
  std::chrono::duration<double> dur2 = time3 - time2;
  std::chrono::duration<double> dur3 = time4 - time3;
  std::chrono::duration<double> dur4 = time5 - time4;
  std::cerr << "detail::per_v (pre, filter, post, ep, comm) took (" << dur0.count() << ","
            << dur1.count() << "," << dur2.count() << "," << dur3.count() << "," << dur4.count()
            << ") num_concurrent_loops=" << num_concurrent_loops << std::endl;
#endif
}

}  // namespace detail

}  // namespace cugraph
