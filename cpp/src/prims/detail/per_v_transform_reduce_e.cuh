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
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

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
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>

#include <numeric>
#include <type_traits>
#include <utility>

namespace cugraph {

namespace detail {

// FIXME: on A6000 we got better performance with 128, need to tune on H100 (possibly due to wasting
// less computing power on processing high degree vertices, we may use different values for
// different kernels for exhaustive tuning)
int32_t constexpr per_v_transform_reduce_e_kernel_block_size = 512;

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
    per_v_transform_reduce_e_kernel_block_size>;
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
        first_valid_thread_id = per_v_transform_reduce_e_kernel_block_size;
      }
    }

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (per_v_transform_reduce_e_kernel_block_size - 1)) /
           per_v_transform_reduce_e_kernel_block_size) *
          per_v_transform_reduce_e_kernel_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if ((i < static_cast<size_t>(local_degree)) &&
              (*edge_partition_e_mask).get(edge_offset + i) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result ? threadIdx.x : per_v_transform_reduce_e_kernel_block_size,
                      cub::Min());
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          first_valid_thread_id = output_thread_id;
          if (threadIdx.x == first_valid_thread_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_thread_id != per_v_transform_reduce_e_kernel_block_size) { break; }
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
          ((static_cast<size_t>(local_degree) + (per_v_transform_reduce_e_kernel_block_size - 1)) /
           per_v_transform_reduce_e_kernel_block_size) *
          per_v_transform_reduce_e_kernel_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if ((i < static_cast<size_t>(local_degree)) && call_pred_op(i)) {
            e_op_result = call_e_op(i);
          }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result ? threadIdx.x : per_v_transform_reduce_e_kernel_block_size,
                      cub::Min());
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          first_valid_thread_id = output_thread_id;
          if (threadIdx.x == first_valid_thread_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_thread_id != per_v_transform_reduce_e_kernel_block_size) { break; }
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
        if (threadIdx.x == ((first_valid_thread_id == per_v_transform_reduce_e_kernel_block_size)
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

template <typename vertex_t>
__host__ __device__ int rank_to_priority(
  int rank,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  int comm_size,
  vertex_t offset /* to evenly distribute traffic */)
{
  if (rank == root) {  // no need for communication (priority 0)
    return int{0};
  } else if (rank / subgroup_size ==
             root / subgroup_size) {  // intra-subgroup communication is sufficient (priorities in
                                      // [1, subgroup_size)
    int modulo     = subgroup_size - 1;
    auto rank_dist = (rank + subgroup_size - root) % subgroup_size;
    return 1 + ((rank_dist - 1) + (offset % modulo)) % modulo;
  } else {  // inter-subgroup communication is necessary (priorities in [subgroup_size, comm_size)
    int modulo = comm_size - subgroup_size;
    auto subgroup_dist =
      ((rank / subgroup_size) + (comm_size / subgroup_size) - (root / subgroup_size)) %
      (comm_size / subgroup_size);
    auto intra_subgroup_rank_dist =
      ((rank % subgroup_size) + subgroup_size - (root % subgroup_size)) % subgroup_size;
    return subgroup_size +
           ((subgroup_dist * subgroup_size + intra_subgroup_rank_dist - subgroup_size) +
            (offset % modulo)) %
             modulo;
  }
}

template <typename vertex_t>
__host__ __device__ int priority_to_rank(
  int priority,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  int comm_size,
  vertex_t offset /* to evenly distribute traffict */)
{
  if (priority == int{0}) {
    return root;
  } else if (priority < subgroup_size) {
    int modulo     = subgroup_size - 1;
    auto rank_dist = 1 + (priority - 1 + modulo - (offset % modulo)) % modulo;
    return (root + rank_dist) % subgroup_size;
  } else {
    int modulo = comm_size - subgroup_size;
    auto rank_dist =
      subgroup_size + (priority - subgroup_size + modulo - (offset % modulo)) % modulo;
    auto subgroup_dist            = rank_dist / subgroup_size;
    auto intra_subgroup_rank_dist = rank_dist % subgroup_size;
    return ((root / subgroup_size + subgroup_dist) % (comm_size / subgroup_size)) * subgroup_size +
           (root % subgroup_size + intra_subgroup_rank_dist) % subgroup_size;
  }
}

template <typename vertex_t, typename priority_t, typename ValueIterator>
std::optional<rmm::device_uvector<bool>> compute_keep_flags(
  raft::comms::comms_t const& comm,
  ValueIterator value_first,
  ValueIterator value_last,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  typename thrust::iterator_traits<ValueIterator>::value_type init,
  bool ignore_local_values,
  rmm::cuda_stream_view stream_view)
{
  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  // For each vertex, select a comm_rank among the GPUs with a value other than init (if there are
  // more than one, the GPU with (comm_rank == root) has the highest priority, the GPUs in the same
  // DGX node should be the next)

  rmm::device_uvector<priority_t> priorities(thrust::distance(value_first, value_last),
                                             stream_view);
  if (ignore_local_values) {
    thrust::fill(rmm::exec_policy(stream_view),
                 priorities.begin(),
                 priorities.end(),
                 std::numeric_limits<priority_t>::max());
  } else {
    thrust::tabulate(
      rmm::exec_policy(stream_view),
      priorities.begin(),
      priorities.end(),
      [value_first, root, subgroup_size, init, comm_rank, comm_size] __device__(auto offset) {
        auto val = *(value_first + offset);
        return (val != init)
                 ? rank_to_priority(
                     comm_rank, root, subgroup_size, comm_size, static_cast<vertex_t>(offset))
                 : std::numeric_limits<priority_t>::max();  // lowest priority
      });
  }
  device_allreduce(comm,
                   priorities.data(),
                   priorities.data(),
                   priorities.size(),
                   raft::comms::op_t::MIN,
                   stream_view);

  std::optional<rmm::device_uvector<bool>> keep_flags{std::nullopt};
  if (!ignore_local_values) {
    keep_flags = rmm::device_uvector<bool>(priorities.size(), stream_view);
    auto offset_priority_pair_first =
      thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), priorities.begin());
    thrust::transform(rmm::exec_policy(stream_view),
                      offset_priority_pair_first,
                      offset_priority_pair_first + priorities.size(),
                      (*keep_flags).begin(),
                      [root, subgroup_size, comm_rank, comm_size] __device__(auto pair) {
                        auto offset   = thrust::get<0>(pair);
                        auto priority = thrust::get<1>(pair);
                        auto rank =
                          priority_to_rank(priority, root, subgroup_size, comm_size, offset);
                        return (rank == comm_rank);
                      });
  }

  return keep_flags;
}

template <typename vertex_t, typename ValueIterator>
std::tuple<rmm::device_uvector<vertex_t>,
           dataframe_buffer_type_t<typename thrust::iterator_traits<ValueIterator>::value_type>>
gather_offset_value_pairs(raft::comms::comms_t const& comm,
                          ValueIterator value_first,
                          ValueIterator value_last,
                          int root,
                          int subgroup_size /* faster interconnect within a subgroup */,
                          typename thrust::iterator_traits<ValueIterator>::value_type init,
                          bool ignore_local_values,  // no valid value in [value_first, value_last)
                          rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  std::optional<rmm::device_uvector<bool>> keep_flags{std::nullopt};
  if (comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority == uint8_t
    keep_flags = compute_keep_flags<vertex_t, uint8_t>(
      comm, value_first, value_last, root, subgroup_size, init, ignore_local_values, stream_view);
  }
#if 0  // FIXME: this should be enabled (currently, raft does not support allreduce on uint16_t).
  else if (comm_size <= std::numeric_limits<uint16_t>::max()) {  // priority == uint16_t
    keep_flags = compute_keep_flags<vertex_t, uint16_t>(
      comm, value_first, value_last, root, subgroup_size, init, stream_view);
  }
#endif
  else {  // priority_t == uint32_t
    keep_flags = compute_keep_flags<vertex_t, uint32_t>(
      comm, value_first, value_last, root, subgroup_size, init, ignore_local_values, stream_view);
  }

  rmm::device_uvector<vertex_t> offsets(0, stream_view);
  auto values = allocate_dataframe_buffer<value_t>(0, stream_view);
  if (keep_flags) {
    auto copy_size = thrust::count(
      rmm::exec_policy(stream_view), (*keep_flags).begin(), (*keep_flags).end(), true);

    offsets.resize(copy_size, stream_view);
    resize_dataframe_buffer(values, copy_size, stream_view);
    auto offset_value_pair_first =
      thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), value_first);
    thrust::copy_if(rmm::exec_policy(stream_view),
                    offset_value_pair_first,
                    offset_value_pair_first + (*keep_flags).size(),
                    (*keep_flags).begin(),
                    thrust::make_zip_iterator(offsets.begin(), get_dataframe_buffer_begin(values)),
                    thrust::identity<bool>{});
  }

  auto rx_sizes = host_scalar_gather(comm, offsets.size(), root, stream_view);
  std::vector<size_t> rx_displs{};
  if (comm_rank == root) {
    rx_displs.resize(rx_sizes.size());
    std::exclusive_scan(rx_sizes.begin(), rx_sizes.end(), rx_displs.begin(), size_t{0});
  }

  // FIXME: calling the following two device_gatherv within device_group_start() and
  // device_group_end() improves performance (approx. 5%)
  // FIXME: or we can implement this in All-to-All after iteration over every edge partition
  // FIXME: we may consdier optionally sending offsets in bitmaps
  rmm::device_uvector<vertex_t> rx_offsets(
    comm_rank == root ? (rx_displs.back() + rx_sizes.back()) : size_t{0}, stream_view);
  device_gatherv(comm,
                 offsets.begin(),
                 rx_offsets.begin(),
                 offsets.size(),
                 rx_sizes,
                 rx_displs,
                 root,
                 stream_view);
  offsets.resize(0, stream_view);
  offsets.shrink_to_fit(stream_view);

  auto rx_values = allocate_dataframe_buffer<value_t>(
    comm_rank == root ? (rx_displs.back() + rx_sizes.back()) : size_t{0}, stream_view);
  device_gatherv(comm,
                 get_dataframe_buffer_begin(values),
                 get_dataframe_buffer_begin(rx_values),
                 values.size(),
                 rx_sizes,
                 rx_displs,
                 root,
                 stream_view);
  resize_dataframe_buffer(values, 0, stream_view);
  shrink_to_fit_dataframe_buffer(values, stream_view);

  return std::make_tuple(std::move(rx_offsets), std::move(rx_values));
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
  std::optional<std::vector<size_t>> const& key_segment_offsets,
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
        thrust::fill(rmm::exec_policy(exec_stream),
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
          segment_key_last += (*key_segment_offsets)[4];
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
      raft::grid_1d_block_t update_grid((*key_segment_offsets)[1],
                                        detail::per_v_transform_reduce_e_kernel_block_size,
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

#define PER_V_PERFORMANCE_MEASUREMENT 0

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
#if PER_V_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time0 = std::chrono::steady_clock::now();
#endif
  constexpr bool update_major  = (incoming == GraphViewType::is_storage_transposed);
  constexpr bool use_input_key = !std::is_same_v<OptionalKeyIterator, void*>;
  constexpr bool filter_input_key =
    GraphViewType::is_multi_gpu && update_major && use_input_key &&
    std::is_same_v<ReduceOp,
                   reduce_op::any<T>>;  // if GraphViewType::is_multi_gpu && update_major &&
                                        // std::is_same_v<ReduceOp, reduce_op::any<T>>, for any
                                        // vertex in the frontier, we need to visit only local edges
                                        // if we find any valid local edge (FIXME: this is
                                        // applicable even when use_input_key is false).

  static_assert(update_major || !use_input_key);
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

  // 1. exclude zero degree keys

  std::optional<std::vector<vertex_t>> local_vertex_partition_segment_offsets{std::nullopt};
  {
    size_t partition_idx = 0;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      partition_idx              = static_cast<size_t>(minor_comm_rank);
    }
    local_vertex_partition_segment_offsets =
      graph_view.local_edge_partition_segment_offsets(partition_idx);
  }

  auto sorted_unique_nzd_key_last = sorted_unique_key_last;
  if constexpr (use_input_key) {
    if (local_vertex_partition_segment_offsets) {
      auto sorted_uniue_nzd_key_last =
        compute_key_lower_bound(sorted_unique_key_first,
                                sorted_unique_key_last,
                                graph_view.local_vertex_partition_range_first() +
                                  *((*local_vertex_partition_segment_offsets).rbegin() + 1),
                                handle.get_stream());
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

  // 3. filter input keys

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

    std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
    if (local_vertex_partition_segment_offsets) {
      if constexpr (use_input_key) {
        key_segment_offsets = compute_key_segment_offsets(
          sorted_unique_key_first,
          sorted_unique_nzd_key_last,
          raft::host_span<vertex_t const>((*local_vertex_partition_segment_offsets).data(),
                                          (*local_vertex_partition_segment_offsets).size()),
          edge_partition.major_range_first(),
          handle.get_stream());
      } else {
        key_segment_offsets = std::vector<size_t>((*local_vertex_partition_segment_offsets).size());
        std::transform((*local_vertex_partition_segment_offsets).begin(),
                       (*local_vertex_partition_segment_offsets).end(),
                       (*key_segment_offsets).begin(),
                       [](vertex_t offset) { return static_cast<size_t>(offset); });
      }
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
      key_segment_offsets,
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
  } else {
    tmp_vertex_value_output_first = vertex_value_output_first;
  }

  /* 4. compute subgroup_size (used to compute priority in device_gatherv) */

  [[maybe_unused]] std::conditional_t<GraphViewType::is_multi_gpu && update_major &&
                                        std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                      int,
                                      std::byte /* dummy */>
    subgroup_size{};
  if constexpr (GraphViewType::is_multi_gpu && update_major &&
                std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    int num_gpus_per_node{};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
    subgroup_size = partition_manager::map_major_comm_to_gpu_row_comm
                      ? std::max(num_gpus_per_node / minor_comm_size, int{1})
                      : std::min(minor_comm_size, num_gpus_per_node);
  }

  // 5. compute optional bitmap info

  std::
    conditional_t<try_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      key_list_bitmap{};
  if constexpr (try_bitmap) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    if (minor_comm_size > 1) {
      size_t bool_size =
        local_vertex_partition_segment_offsets
          ? *((*local_vertex_partition_segment_offsets).rbegin() + 1)
          : graph_view
              .local_vertex_partition_range_size();  // FIXME: if filtered, we can reduce bool_size

      // FIXME: *sorted_unique_nzd_key_last - *sorted_unique_key_first could be smaller than
      // bool_size by a non-negligible amount
      key_list_bitmap =
        compute_vertex_list_bitmap_info(sorted_unique_key_first,
                                        sorted_unique_nzd_key_last,
                                        graph_view.local_vertex_partition_range_first(),
                                        graph_view.local_vertex_partition_range_first() + bool_size,
                                        handle.get_stream());
    }
  }

  // 6. collect local_key_list_sizes & use_bitmap_flags

  std::conditional_t<use_input_key, std::vector<size_t>, std::byte /* dummy */>
    local_key_list_sizes{};
  std::conditional_t<try_bitmap, std::vector<bool>, std::byte /* dummy */> use_bitmap_flags{};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    if constexpr (use_input_key) {
      local_key_list_sizes = host_scalar_allgather(
        minor_comm,
        static_cast<size_t>(thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last)),
        handle.get_stream());
    }
    if constexpr (try_bitmap) {
      auto tmp_flags = host_scalar_allgather(
        minor_comm, key_list_bitmap ? uint8_t{1} : uint8_t{0}, handle.get_stream());
      use_bitmap_flags.resize(tmp_flags.size());
      std::transform(tmp_flags.begin(), tmp_flags.end(), use_bitmap_flags.begin(), [](auto flag) {
        return flag == uint8_t{1};
      });
    }
  } else {
    if constexpr (use_input_key) {
      local_key_list_sizes = std::vector<size_t>{
        static_cast<size_t>(thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last))};
    }
  }

  // 7. set-up stream pool

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    if (local_vertex_partition_segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // memory footprint vs parallelism trade-off
      // peak memory requirement per loop is
      // update_major ? (use_input_key ? aggregate key list size : V) / comm_size * sizeof(T) : 0
      // and limit memory requirement to (E / comm_size) * sizeof(vertex_t)
      // FIXME: what about offsets & values?

      size_t num_streams =
        std::min(static_cast<size_t>(minor_comm_size) * max_segments,
                 raft::round_down_safe(handle.get_stream_pool_size(), max_segments));
      if constexpr (update_major) {
        size_t value_size{0};
        if constexpr (is_thrust_tuple_of_arithmetic<T>::value) {
          auto elem_sizes = compute_thrust_tuple_element_sizes<T>{}();
          value_size      = std::reduce(elem_sizes.begin(), elem_sizes.end());
        } else {
          value_size = sizeof(T);
        }
        size_t key_size{0};
        if constexpr (use_input_key) {
          if constexpr (std::is_same_v<key_t, vertex_t>) {
            key_size = sizeof(vertex_t);
          } else {
            key_size = sizeof(thrust::tuple_element<0, key_t>::type) +
                       sizeof(thrust::tuple_element<1, key_t>::type);
          }
        }

        auto num_edges = graph_view.compute_number_of_edges(handle);

        size_t aggregate_major_range_size{};
        if constexpr (use_input_key) {
          aggregate_major_range_size =
            host_scalar_allreduce(handle.get_comms(),
                                  static_cast<size_t>(thrust::distance(sorted_unique_key_first,
                                                                       sorted_unique_nzd_key_last)),
                                  raft::comms::op_t::SUM,
                                  handle.get_stream());
        } else {
          aggregate_major_range_size = graph_view.number_of_vertices();
        }
        num_streams = std::min(
          static_cast<size_t>(
            (aggregate_major_range_size > 0
               ? (static_cast<double>(num_edges) / static_cast<double>(aggregate_major_range_size))
               : double{0}) *
            (static_cast<double>(sizeof(vertex_t)) / static_cast<double>(value_size + key_size))) *
            max_segments,
          num_streams);
      }

      if (num_streams >= max_segments) {
        assert((num_streams % max_segments) == 0);
        stream_pool_indices = std::vector<size_t>(num_streams);
        std::iota((*stream_pool_indices).begin(), (*stream_pool_indices).end(), size_t{0});
      }
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

  std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                     std::vector<rmm::device_uvector<vertex_t>>,
                     std::byte /* dummy */>
    offset_vectors{};
  std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                     std::vector<dataframe_buffer_type_t<T>>,
                     std::byte /* dummy */>
    value_vectors{};
  if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto capacity = graph_view.local_edge_partition_segment_offsets(0) ? max_segments : 1;
    offset_vectors.reserve(capacity);
    value_vectors.reserve(capacity);

    for (size_t i = 0; i < capacity; ++i) {
      offset_vectors.emplace_back(0, handle.get_stream());
      value_vectors.emplace_back(0, handle.get_stream());
    }
  }

  if (stream_pool_indices) { handle.sync_stream(); }

  // 9. proces local edge partitions

#if PER_V_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time1 = std::chrono::steady_clock::now();
#endif
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i += num_concurrent_loops) {
    auto loop_count =
      std::min(num_concurrent_loops, graph_view.number_of_local_edge_partitions() - i);

    std::conditional_t<GraphViewType::is_multi_gpu && use_input_key,
                       std::vector<dataframe_buffer_type_t<T>>,
                       std::byte /* dummy */>
      edge_partition_key_buffers{};
    if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
      edge_partition_key_buffers.reserve(loop_count);
    }
    std::vector<std::optional<std::vector<size_t>>> key_segment_offset_vectors{};
    key_segment_offset_vectors.reserve(loop_count);
    std::conditional_t<GraphViewType::is_multi_gpu && update_major,
                       std::vector<dataframe_buffer_type_t<T>>,
                       std::byte /* dummy */>
      major_output_buffers{};
    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      major_output_buffers.reserve(loop_count);
    }
    for (size_t j = 0; j < loop_count; ++j) {
      auto partition_idx = i * num_concurrent_loops + j;
      auto loop_stream   = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                             : handle.get_stream();

      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(partition_idx));
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);

      auto edge_partition_key_first = sorted_unique_key_first;
      auto edge_partition_key_last  = sorted_unique_nzd_key_last;
      if constexpr (GraphViewType::is_multi_gpu && use_input_key) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size     = minor_comm.get_size();
        auto edge_partition_key_buffer = allocate_dataframe_buffer<key_t>(
          minor_comm_size > 1 ? local_key_list_sizes[partition_idx] : size_t{0}, loop_stream);
        if (minor_comm_size > 1) {
          auto const minor_comm_rank = minor_comm.get_rank();

          if constexpr (try_bitmap) {
            std::variant<raft::device_span<uint32_t const>, decltype(sorted_unique_key_first)>
              v_list{};
            if (use_bitmap_flags[partition_idx]) {
              v_list = (static_cast<int>(partition_idx) == minor_comm_rank)
                         ? raft::device_span<uint32_t const>((*key_list_bitmap).data(),
                                                             (*key_list_bitmap).size())
                         : raft::device_span<uint32_t const>(static_cast<uint32_t const*>(nullptr),
                                                             size_t{0});
            } else {
              v_list = sorted_unique_key_first;
            }
            auto bool_size = segment_offsets ? *((*segment_offsets).rbegin() + 1)
                                             : edge_partition.major_range_size();
            device_bcast_vertex_list(minor_comm,
                                     v_list,
                                     get_dataframe_buffer_begin(edge_partition_key_buffer),
                                     edge_partition.major_range_first(),
                                     edge_partition.major_range_first() + bool_size,
                                     local_key_list_sizes[partition_idx],
                                     static_cast<int>(partition_idx),
                                     loop_stream);
          } else {
            device_bcast(minor_comm,
                         sorted_unique_key_first,
                         get_dataframe_buffer_begin(edge_partition_key_buffer),
                         local_key_list_sizes[partition_idx],
                         static_cast<int>(partition_idx),
                         loop_stream);
          }
        }
        edge_partition_key_buffers.push_back(std::move(edge_partition_key_buffer));
        edge_partition_key_first = get_dataframe_buffer_begin(edge_partition_key_buffers[j]);
        edge_partition_key_last  = get_dataframe_buffer_end(edge_partition_key_buffers[j]);
      }

      std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
      if (segment_offsets) {
        if constexpr (use_input_key) {
          key_segment_offsets = compute_key_segment_offsets(
            edge_partition_key_first,
            edge_partition_key_last,
            raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
            edge_partition.major_range_first(),
            loop_stream);
        } else {
          key_segment_offsets = std::vector<size_t>((*segment_offsets).size());
          std::transform((*segment_offsets).begin(),
                         (*segment_offsets).end(),
                         (*key_segment_offsets).begin(),
                         [](vertex_t offset) { return static_cast<size_t>(offset); });
        }
      }
      key_segment_offset_vectors.push_back(std::move(key_segment_offsets));

      if constexpr (GraphViewType::is_multi_gpu && update_major) {
        size_t buffer_size{0};
        if constexpr (use_input_key) {
          buffer_size = local_key_list_sizes[partition_idx];
        } else {
          buffer_size = segment_offsets
                          ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                          : edge_partition.major_range_size();
        }
        major_output_buffers.push_back(allocate_dataframe_buffer<T>(buffer_size, loop_stream));
      }
    }
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }

    for (size_t j = 0; j < loop_count; ++j) {
      auto partition_idx = i * num_concurrent_loops + j;

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

      auto const& key_segment_offsets = key_segment_offset_vectors[j];

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
          output_buffer = get_dataframe_buffer_begin(major_output_buffers[j]);
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
          key_segment_offsets,
          edge_partition_stream_pool_indices);
      }

      if constexpr (GraphViewType::is_multi_gpu && update_major) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        auto const minor_comm_size = minor_comm.get_size();

        if (key_segment_offsets && edge_partition_stream_pool_indices) {
          if (edge_partition.dcs_nzd_vertex_count()) {
            if ((*key_segment_offsets)[4] - (*key_segment_offsets)[3] > 0) {
              auto segment_stream =
                handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[0]);
              auto segment_offset = (*key_segment_offsets)[3];
              auto segment_size   = (*key_segment_offsets)[4] - (*key_segment_offsets)[3];
              if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
                auto [offsets, values] = gather_offset_value_pairs<vertex_t>(
                  minor_comm,
                  output_buffer + segment_offset,
                  output_buffer + (segment_offset + segment_size),
                  static_cast<int>(partition_idx),
                  subgroup_size,
                  init,
                  process_local_edges ? false : true,
                  segment_stream);
                if (static_cast<int>(partition_idx) == minor_comm_rank) {
                  offset_vectors[3] = std::move(offsets);
                  value_vectors[3]  = std::move(values);
                }
              } else {
                device_reduce(minor_comm,
                              output_buffer + segment_offset,
                              tmp_vertex_value_output_first + segment_offset,
                              segment_size,
                              ReduceOp::compatible_raft_comms_op,
                              static_cast<int>(partition_idx),
                              segment_stream);
              }
            }
          }
          if ((*key_segment_offsets)[3] - (*key_segment_offsets)[2] > 0) {
            auto segment_stream =
              handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[1]);
            auto segment_offset = (*key_segment_offsets)[2];
            auto segment_size   = (*key_segment_offsets)[3] - (*key_segment_offsets)[2];
            if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
              auto [offsets, values] =
                gather_offset_value_pairs<vertex_t>(minor_comm,
                                                    output_buffer + segment_offset,
                                                    output_buffer + (segment_offset + segment_size),
                                                    static_cast<int>(partition_idx),
                                                    subgroup_size,
                                                    init,
                                                    process_local_edges ? false : true,
                                                    segment_stream);
              if (static_cast<int>(partition_idx) == minor_comm_rank) {
                offset_vectors[2] = std::move(offsets);
                value_vectors[2]  = std::move(values);
              }
            } else {
              device_reduce(minor_comm,
                            output_buffer + segment_offset,
                            tmp_vertex_value_output_first + segment_offset,
                            segment_size,
                            ReduceOp::compatible_raft_comms_op,
                            static_cast<int>(partition_idx),
                            segment_stream);
            }
          }
          if ((*key_segment_offsets)[2] - (*key_segment_offsets)[1] > 0) {
            auto segment_stream =
              handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[2]);
            auto segment_offset = (*key_segment_offsets)[1];
            auto segment_size   = (*key_segment_offsets)[2] - (*key_segment_offsets)[1];
            if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
              auto [offsets, values] =
                gather_offset_value_pairs<vertex_t>(minor_comm,
                                                    output_buffer + segment_offset,
                                                    output_buffer + (segment_offset + segment_size),
                                                    static_cast<int>(partition_idx),
                                                    subgroup_size,
                                                    init,
                                                    process_local_edges ? false : true,
                                                    segment_stream);
              if (static_cast<int>(partition_idx) == minor_comm_rank) {
                offset_vectors[1] = std::move(offsets);
                value_vectors[1]  = std::move(values);
              }
            } else {
              device_reduce(minor_comm,
                            output_buffer + segment_offset,
                            tmp_vertex_value_output_first + segment_offset,
                            segment_size,
                            ReduceOp::compatible_raft_comms_op,
                            static_cast<int>(partition_idx),
                            segment_stream);
            }
          }
          if ((*key_segment_offsets)[1] > 0) {
            auto segment_stream =
              handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[3]);
            auto segment_size = (*key_segment_offsets)[1];
            if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
              auto [offsets, values] =
                gather_offset_value_pairs<vertex_t>(minor_comm,
                                                    output_buffer,
                                                    output_buffer + segment_size,
                                                    static_cast<int>(partition_idx),
                                                    subgroup_size,
                                                    init,
                                                    process_local_edges ? false : true,
                                                    segment_stream);
              if (static_cast<int>(partition_idx) == minor_comm_rank) {
                offset_vectors[0] = std::move(offsets);
                value_vectors[0]  = std::move(values);
              }
            } else {
              device_reduce(minor_comm,
                            output_buffer,
                            tmp_vertex_value_output_first,
                            segment_size,
                            ReduceOp::compatible_raft_comms_op,
                            static_cast<int>(partition_idx),
                            segment_stream);
            }
          }
        } else {
          size_t reduction_size{};
          if constexpr (use_input_key) {
            reduction_size = static_cast<size_t>(
              thrust::distance(edge_partition_key_first, edge_partition_key_last));
          } else {
            reduction_size = static_cast<size_t>(
              key_segment_offsets
                ? *((*key_segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                : edge_partition.major_range_size());
          }
          if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
            auto [offsets, values] =
              gather_offset_value_pairs<vertex_t>(minor_comm,
                                                  output_buffer,
                                                  output_buffer + reduction_size,
                                                  static_cast<int>(partition_idx),
                                                  subgroup_size,
                                                  init,
                                                  process_local_edges ? false : true,
                                                  handle.get_stream());
            if (static_cast<int>(partition_idx) == minor_comm_rank) {
              offset_vectors[0] = std::move(offsets);
              value_vectors[0]  = std::move(values);
            }
          } else {
            device_reduce(minor_comm,
                          output_buffer,
                          tmp_vertex_value_output_first,
                          reduction_size,
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(partition_idx),
                          handle.get_stream());
          }
        }
      }
    }
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
  }
#if PER_V_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time2 = std::chrono::steady_clock::now();
#endif

  // 10. scatter

  if constexpr (GraphViewType::is_multi_gpu && update_major &&
                std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();

    auto segment_offsets =
      graph_view.local_edge_partition_segment_offsets(static_cast<size_t>(minor_comm_rank));

    std::optional<std::vector<size_t>> edge_partition_stream_pool_indices{std::nullopt};
    if (segment_offsets && (handle.get_stream_pool_size() >= max_segments)) {
      edge_partition_stream_pool_indices = std::vector<size_t>(max_segments);
      std::iota((*edge_partition_stream_pool_indices).begin(),
                (*edge_partition_stream_pool_indices).end(),
                size_t{0});
    }

    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(static_cast<size_t>(minor_comm_rank)));

    std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
    if (segment_offsets) {
      if constexpr (use_input_key) {
        key_segment_offsets = compute_key_segment_offsets(
          sorted_unique_key_first,
          sorted_unique_nzd_key_last,
          raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
          edge_partition.major_range_first(),
          handle.get_stream());
      } else {
        key_segment_offsets = std::vector<size_t>((*segment_offsets).size());
        std::transform((*segment_offsets).begin(),
                       (*segment_offsets).end(),
                       (*key_segment_offsets).begin(),
                       [](vertex_t offset) { return static_cast<size_t>(offset); });
      }
    }

    if (key_segment_offsets && edge_partition_stream_pool_indices) {
      if (edge_partition.dcs_nzd_vertex_count()) {
        if ((*key_segment_offsets)[4] - (*key_segment_offsets)[3] > 0) {
          auto segment_stream =
            handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[0]);
          auto segment_offset = (*key_segment_offsets)[3];
          thrust::scatter(rmm::exec_policy(segment_stream),
                          get_dataframe_buffer_begin(value_vectors[3]),
                          get_dataframe_buffer_end(value_vectors[3]),
                          offset_vectors[3].begin(),
                          tmp_vertex_value_output_first + segment_offset);
        }
      }
      if ((*key_segment_offsets)[3] - (*key_segment_offsets)[2] > 0) {
        auto segment_stream =
          handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[1]);
        auto segment_offset = (*key_segment_offsets)[2];
        thrust::scatter(rmm::exec_policy(segment_stream),
                        get_dataframe_buffer_begin(value_vectors[2]),
                        get_dataframe_buffer_end(value_vectors[2]),
                        offset_vectors[2].begin(),
                        tmp_vertex_value_output_first + segment_offset);
      }
      if ((*key_segment_offsets)[2] - (*key_segment_offsets)[1] > 0) {
        auto segment_stream =
          handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[2]);
        auto segment_offset = (*key_segment_offsets)[1];
        thrust::scatter(rmm::exec_policy(segment_stream),
                        get_dataframe_buffer_begin(value_vectors[1]),
                        get_dataframe_buffer_end(value_vectors[1]),
                        offset_vectors[1].begin(),
                        tmp_vertex_value_output_first + segment_offset);
      }
      if ((*key_segment_offsets)[1] > 0) {
        auto segment_stream =
          handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[3]);
        thrust::scatter(rmm::exec_policy(segment_stream),
                        get_dataframe_buffer_begin(value_vectors[0]),
                        get_dataframe_buffer_end(value_vectors[0]),
                        offset_vectors[0].begin(),
                        tmp_vertex_value_output_first);
      }
    } else {
      thrust::scatter(handle.get_thrust_policy(),
                      get_dataframe_buffer_begin(value_vectors[0]),
                      get_dataframe_buffer_end(value_vectors[0]),
                      offset_vectors[0].begin(),
                      tmp_vertex_value_output_first);
    }
  }
#if PER_V_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time3 = std::chrono::steady_clock::now();
#endif

  // 11. communication

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
#if PER_V_PERFORMANCE_MEASUREMENT  // FIXME: delete
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto time4                         = std::chrono::steady_clock::now();
  std::chrono::duration<double> dur0 = time1 - time0;
  std::chrono::duration<double> dur1 = time2 - time1;
  std::chrono::duration<double> dur2 = time3 - time2;
  std::chrono::duration<double> dur3 = time4 - time3;
  std::cout << "\t\tdetail::per_v took (" << dur0.count() << "," << dur1.count() << ","
            << dur2.count() << ")" << std::endl;
#endif
}

}  // namespace detail

}  // namespace cugraph
