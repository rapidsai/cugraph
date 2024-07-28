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
#include "prims/pred_op.cuh"
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
    if constexpr (std::is_same_v<PredOp, pred_op::const_true<edge_t>>) {
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
  ReduceOp reduce_op)
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

      if (edge_partition_e_mask) {
        update_result_value_output<update_major>(
          edge_partition,
          indices,
          local_degree,
          call_e_op,
          init,
          reduce_op,
          [&edge_partition_e_mask, edge_offset] __device__(edge_t i) {
            return (*edge_partition_e_mask).get(edge_offset + i);
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
                                                 pred_op::const_true<edge_t>{},
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
  ReduceOp reduce_op)
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

    if (edge_partition_e_mask) {
      update_result_value_output<update_major>(
        edge_partition,
        indices,
        local_degree,
        call_e_op,
        init,
        reduce_op,
        [&edge_partition_e_mask, edge_offset] __device__(edge_t i) {
          return (*edge_partition_e_mask).get(edge_offset + i);
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
                                               pred_op::const_true<edge_t>{},
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
  T identity_element /* relevant only if update_major == true && !std::is_same_v<ReduceOp,
                        reduce_op::any<T>> */
  ,
  ReduceOp reduce_op)
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

    [[maybe_unused]] std::conditional_t<update_major, T, std::byte /* dummy */>
      reduced_e_op_result{};
    [[maybe_unused]] std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                        int32_t,
                                        std::byte /* dummy */>
      first_valid_lane_id{};
    if constexpr (update_major) { reduced_e_op_result = (lane_id == 0) ? init : identity_element; }
    if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
      first_valid_lane_id = raft::warp_size();
    }

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
          raft::warp_size();
        for (size_t i = lane_id; i < rounded_up_local_degree; i += raft::warp_size()) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if (i < static_cast<size_t>(local_degree) &&
              (*edge_partition_e_mask).get(edge_offset + i)) {
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
          if ((*edge_partition_e_mask).get(edge_offset + i)) {
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
          if (i < static_cast<size_t>(local_degree)) { e_op_result = call_e_op(i); }
          first_valid_lane_id = WarpReduce(temp_storage[threadIdx.x / raft::warp_size()])
                                  .Reduce(e_op_result ? lane_id : raft::warp_size(), cub::Min());
          first_valid_lane_id = __shfl_sync(raft::warp_full_mask(), first_valid_lane_id, int{0});
          if (lane_id == first_valid_lane_id) { reduced_e_op_result = *e_op_result; }
          if (first_valid_lane_id != raft::warp_size()) { break; }
        }
      } else {
        for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
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
  T identity_element /* relevant only if update_major == true && !std::is_same_v<ReduceOp,
                        reduce_op::any<T>> */
  ,
  ReduceOp reduce_op)
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

    [[maybe_unused]] std::conditional_t<update_major, T, std::byte /* dummy */>
      reduced_e_op_result{};
    [[maybe_unused]] std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                        int32_t,
                                        std::byte /* dummy */>
      first_valid_thread_id{};
    if constexpr (update_major) {
      reduced_e_op_result = threadIdx.x == 0 ? init : identity_element;
    }
    if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
      first_valid_thread_id = per_v_transform_reduce_e_kernel_block_size;
    }

    if (edge_partition_e_mask) {
      if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
        auto rounded_up_local_degree =
          ((static_cast<size_t>(local_degree) + (per_v_transform_reduce_e_kernel_block_size - 1)) /
           per_v_transform_reduce_e_kernel_block_size) *
          per_v_transform_reduce_e_kernel_block_size;
        for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
          thrust::optional<T> e_op_result{thrust::nullopt};
          if (i < static_cast<size_t>(local_degree) &&
              ((*edge_partition_e_mask).get_(edge_offset + i))) {
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
          if ((*edge_partition_e_mask).get(edge_offset + i)) {
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
          if (i < static_cast<size_t>(local_degree)) { e_op_result = call_e_op(i); }
          first_valid_thread_id =
            BlockReduce(temp_storage)
              .Reduce(e_op_result ? threadIdx.x : per_v_transform_reduce_e_kernel_block_size,
                      cub::Min());
          if (threadIdx.x == 0) { output_thread_id = first_valid_thread_id; }
          __syncthreads();
          if (threadIdx.x == output_thread_id) { reduced_e_op_result = *e_op_result; }
          if (output_thread_id != per_v_transform_reduce_e_kernel_block_size) { break; }
        }
      } else {
        for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
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
    int modulo = subgroup_size - 1;
    return int{1} + static_cast<int>((static_cast<size_t>(rank) + offset) % modulo);
  } else {  // inter-subgroup communication is necessary (priorities in [subgroup_size, comm_size)
    int modulo = comm_size - subgroup_size;
    return subgroup_size + static_cast<int>((static_cast<size_t>(rank) + offset) % modulo);
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
    int modulo = subgroup_size - int{1};
    return static_cast<int>(
      (static_cast<size_t>(priority - int{1}) + (modulo - static_cast<int>(offset % modulo))) %
      modulo);
  } else {
    int modulo = comm_size - subgroup_size;
    return static_cast<int>((static_cast<size_t>(priority - subgroup_size) +
                             (modulo - static_cast<int>(offset % modulo))) %
                            modulo);
  }
}

template <typename vertex_t, typename priority_t, typename ValueIterator>
rmm::device_uvector<bool> compute_keep_flags(
  raft::comms::comms_t const& comm,
  ValueIterator value_first,
  ValueIterator value_last,
  int root,
  int subgroup_size /* faster interconnect within a subgroup */,
  typename thrust::iterator_traits<ValueIterator>::value_type init,
  rmm::cuda_stream_view stream_view)
{
  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  // For each vertex, select a comm_rank among the GPUs with a value other than init (if there are
  // more than one, the GPU with (comm_rank == root) has the highest priority, the GPUs in the same
  // DGX node should be the next)

  rmm::device_uvector<priority_t> priorities(thrust::distance(value_first, value_last),
                                             stream_view);
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
  device_allreduce(comm,
                   priorities.data(),
                   priorities.data(),
                   priorities.size(),
                   raft::comms::op_t::MIN,
                   root,
                   stream_view);

  rmm::device_uvector<bool> keep_flags(priorities.size());
  auto offset_priority_pair_first =
    thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), priorities.begin());
  thrust::transform(rmm::exec_policy(stream_view),
                    offset_priority_pair_first,
                    offset_priority_pair_first + priorities.size(),
                    keep_flags.begin(),
                    [root, subgroup_size, comm_rank, comm_size] __device__(auto pair) {
                      auto offset   = thrust::get<0>(pair);
                      auto priority = thrust::get<1>(pair);
                      auto rank =
                        priority_to_rank(priority, root, subgroup_size, comm_size, offset);
                      return (rank == comm_rank);
                    });

  return keep_flags;
}

template <typename vertex_t, typename ValueIterator>
std::tuple<rmm::device_uvector<vertex_t>,
           dataframe_buffer_type_t<typename thrust::iterator_traits<ValueIterator>::value_type>>
compute_offset_value_pairs(raft::comms::comms_t const& comm,
                           ValueIterator value_first,
                           ValueIterator value_last,
                           int root,
                           int subgroup_size /* faster interconnect within a subgroup */,
                           typename thrust::iterator_traits<ValueIterator>::value_type init,
                           rmm::cuda_stream_view stream_view)
{
  using value_t = typename thrust::iterator_traits<ValueIterator>::value_type;

  auto const comm_rank = comm.get_rank();
  auto const comm_size = comm.get_size();

  rmm::device_uvector<bool> keep_flags(0, stream_view);
  if (comm_size <= std::numeric_limits<uint8_t>::max()) {  // priority == uint8_t
    keep_flags = compute_keep_flags<vertex_t, uint8_t>(
      comm, value_first, value_last, root, subgroup_size, init, stream_view);
  } else if (comm_size <= std::numeric_limits<uint16_t>::max()) {  // priority == uint16_t
    keep_flags = compute_keep_flags<vertex_t, uint16_t>(
      comm, value_first, value_last, root, subgroup_size, init, stream_view);
  } else {  // priority_t == uint32_t
    keep_flags = compute_keep_flags<vertex_t, uint32_t>(
      comm, value_first, value_last, root, subgroup_size, init, stream_view);
  }

  auto copy_size = thrust::count_if(
    rmm::exec_policy(stream_view), keep_flags.begin(), keep_flags.end(), thrust::identity<bool>{});

  rmm::device_uvector<vertex_t> offsets(copy_size, stream_view);
  auto values = allocate_dataframe_buffer<value_t>(copy_size, stream_view);
  auto offset_value_pair_first =
    thrust::make_zip_iterator(thrust::make_counting_iterator(vertex_t{0}), value_first);
  thrust::copy_if(rmm::exec_policy(stream_view),
                  offset_value_pair_first,
                  offset_value_pair_first + keep_flags.size(),
                  keep_flags.begin(),
                  thrust::make_zip_iterator(offsets.begin(), dataframe_buffer_begin(values)),
                  thrust::identity<bool>{});

  return std::make_tuple(std::move(offsets), std::move(values));
}

template <typename vertex_t, typename value_t, typename VertexValueOutputIterator>
void gather_offset_value_pairs_and_update_vertex_value_output(
  raft::comms::comms_t const& comm,
  rmm::device_uvector<vertex_t>&& offsets,
  dataframe_buffer_type_t<value_t>&& values,
  VertexValueOutputIterator vertex_value_output_first,
  int root,
  rmm::cuda_stream_view stream_view)
{
  auto const comm_rank = comm.get_rank();

  auto rx_sizes = host_scalar_gather(comm, offsets.size(), root, stream_view);
  std::vector<size_t> rx_displs{};
  rmm::device_uvector<vertex_t> rx_offsets(0, stream_view);
  if (comm_rank == root) {
    rx_displs.resize(rx_sizes.size());
    std::exclusive_scan(rx_sizes.begin(), rx_sizes.end(), rx_displs.begin(), size_t{0});
    rx_offsets.resize(rx_displs.back() + rx_sizes.back(), stream_view);
  }

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

  auto rx_values = allocate_dataframe_buffer<value_t>(rx_offsets.size(), stream_view);
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

  if (comm_rank == root) {
    thrust::scatter(rmm::exec_policy(stream_view),
                    get_dataframe_buffer_begin(rx_values),
                    get_dataframe_buffer_end(rx_values),
                    rx_offsets.begin(),
                    vertex_value_output_first);
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
                              VertexValueOutputIterator vertex_value_output_first)
{
  constexpr bool update_major  = (incoming == GraphViewType::is_storage_transposed);
  constexpr bool use_input_key = !std::is_same_v<OptionalKeyIterator, void*>;

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

  constexpr bool use_bitmap = GraphViewType::is_multi_gpu &&
                              !std::is_same_v<OptionalKeyIterator, void*> &&
                              std::is_same_v<key_t, vertex_t>;

  [[maybe_unused]] constexpr auto max_segments =
    detail::num_sparse_segments_per_vertex_partition + size_t{1};

  // 1. prepare key list

  auto sorted_unique_nzd_key_last = sorted_unique_key_last;
  if constexpr (use_input_key) {
    size_t partition_idx = 0;
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      partition_idx              = static_cast<size_t>(minor_comm_rank);
    }
    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);
    if (segment_offsets) {
      auto sorted_uniue_nzd_key_last = compute_key_lower_bound(
        sorted_unique_key_first,
        sorted_unique_key_last,
        graph_view.local_vertex_partition_range_first() + ((*segment_offsets).rbegin() + 1),
        handle.get_stream());
    }
  }

  std::conditional_t<use_input_key, std::vector<size_t>, std::byte /* dummy */>
    local_key_list_sizes{};
  if constexpr (use_input_key) {
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      local_key_list_sizes = host_scalar_allgather(
        minor_comm,
        static_cast<size_t>(thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last)),
        handle.get_stream());
    } else {
      local_key_list_sizes = std::vector<size_t>{
        static_cast<size_t>(thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last))};
    }
  }

  std::
    conditional_t<use_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      key_list_bitmap{};
  std::conditional_t<use_bitmap, std::vector<bool>, std::byte /* dummy */> use_bitmap_flags{};
  if constexpr (use_bitmap) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto segment_offsets =
      graph_view.local_edge_partition_segment_offsets(static_cast<size_t>(minor_comm_rank));
    size_t bool_size = segment_offsets ? *((*segment_offsets).rbegin() + 1)
                                       : graph_view.local_vertex_partition_range_size();

    std::tie(key_list_bitmap, use_bitmap_flags) =
      compute_vertex_list_bitmap_info(minor_comm,
                                      sorted_unique_key_first,
                                      sorted_unique_nzd_key_last,
                                      graph_view.local_vertex_partition_range_first(),
                                      graph_view.local_vertex_partition_range_first() + bool_size,
                                      handle.get_stream());
  }

  // 2. compute subgroup_size, set-up temporary buffers & stream pool, and initialize

  [[maybe_unused]] std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                                      int,
                                      std::byte /* dummy */>
    subgroup_size{};
  if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    int num_gpus_per_node{};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
    subgroup_size = partition_manager::map_major_comm_to_gpu_row_comm
                      ? std::max(num_gpus_per_node / minor_comm_size, int{1})
                      : std::min(minor_comm_size, num_gpus_per_node);
  }

  using minor_tmp_buffer_type = std::conditional_t<GraphViewType::is_storage_transposed,
                                                   edge_src_property_t<GraphViewType, T>,
                                                   edge_dst_property_t<GraphViewType, T>>;
  [[maybe_unused]] std::unique_ptr<minor_tmp_buffer_type> minor_tmp_buffer{};
  if constexpr (GraphViewType::is_multi_gpu && !update_major) {
    minor_tmp_buffer = std::make_unique<minor_tmp_buffer_type>(handle, graph_view);
  }

  using edge_partition_minor_output_device_view_t =
    std::conditional_t<GraphViewType::is_multi_gpu && !update_major,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         decltype(minor_tmp_buffer->mutable_view().value_first())>,
                       void /* dummy */>;

  if constexpr (update_major) {  // no vertices in the zero degree segment are visited
    if constexpr (use_input_key) {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first +
                     thrust::distance(sorted_unique_key_first, sorted_unique_nzd_key_last),
                   vertex_value_output_first +
                     thrust::distance(sorted_unique_key_first, sorted_unique_key_last),
                   init);
    } else {
      size_t partition_idx = 0;
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        partition_idx              = static_cast<size_t>(minor_comm_rank);
      }
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);
      if (segment_offsets) {
        thrust::fill(handle.get_thrust_policy(),
                     vertex_value_output_first + *((*segment_offsets).rbegin() + 1),
                     vertex_value_output_first + *((*segment_offsets).rbegin()),
                     init);
      }
    }
  } else {
    if constexpr (GraphViewType::is_multi_gpu) {
      auto minor_init = init;
      auto view       = minor_tmp_buffer->view();
      if (view.keys()) {  // defer applying the initial value to the end as minor_tmp_buffer may
                          // not store values for the entire minor range
        minor_init = ReduceOp::identity_element;
      } else {
        auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_rank = major_comm.get_rank();
        minor_init                 = (major_comm_rank == 0) ? init : ReduceOp::identity_element;
      }
      fill_edge_minor_property(handle, graph_view, minor_tmp_buffer->mutable_view(), minor_init);
    } else {
      thrust::fill(handle.get_thrust_policy(),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.local_vertex_partition_range_size(),
                   init);
    }
  }

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    if ((graph_view.local_edge_partition_segment_offsets(0)) &&
        (handle.get_stream_pool_size() >= max_segments)) {
      for (size_t i = 1; i < graph_view.number_of_local_edge_partitions(); ++i) {
        assert(graph_view.local_edge_partition_segment_offsets(i));
      }

      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      // memory footprint vs parallelism trade-off
      // peak memory requirement per loop is
      // update_major ? (use_input_key ? aggregate key list size : V) / comm_size * sizeof(T) : 0
      // and limit memory requirement to (E / comm_size) * sizeof(vertex_t)
      // FIXME: should we consider edge_partition_key_buffer as well?

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
        handle.sync_stream();
      }
    }
  }

  std::vector<dataframe_buffer_type_t<T>> major_tmp_buffers{};
  if constexpr (GraphViewType::is_multi_gpu && update_major) {
    std::vector<size_t> major_tmp_buffer_sizes(graph_view.number_of_local_edge_partitions(),
                                               size_t{0});
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      if constexpr (use_input_key) {
        major_tmp_buffer_sizes = local_key_list_sizes;
      } else {
        auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
        if (segment_offsets) {
          major_tmp_buffer_sizes[i] =
            *((*segment_offsets).rbegin() + 1);  // exclude the zero degree segment
        } else {
          if constexpr (GraphViewType::is_storage_transposed) {
            major_tmp_buffer_sizes[i] = graph_view.local_edge_partition_dst_range_size(i);
          } else {
            major_tmp_buffer_sizes[i] = graph_view.local_edge_partition_src_range_size(i);
          }
        }
      }
    }
    if (stream_pool_indices) {
      auto num_concurrent_loops = (*stream_pool_indices).size() / max_segments;
      major_tmp_buffers.reserve(num_concurrent_loops);
      for (size_t i = 0; i < num_concurrent_loops; ++i) {
        size_t max_size{0};
        for (size_t j = i; j < graph_view.number_of_local_edge_partitions();
             j += num_concurrent_loops) {
          max_size = std::max(major_tmp_buffer_sizes[j], max_size);
        }
        major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(max_size, handle.get_stream()));
      }
    } else {
      major_tmp_buffers.reserve(1);
      major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(
        *std::max_element(major_tmp_buffer_sizes.begin(), major_tmp_buffer_sizes.end()),
        handle.get_stream()));
    }
  } else {  // dummy
    major_tmp_buffers.reserve(1);
    major_tmp_buffers.push_back(allocate_dataframe_buffer<T>(size_t{0}, handle.get_stream()));
  }

  std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                     std::vector<rmm::device_uvector<vertex_t>>,
                     std::byte /* dummy */>
    offset_vectors{};
  std::conditional_t<update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>,
                     std::vector<dataframe_buffer_type_t<T>>,
                     std::byte /* dummy */>
    value_vectors{};
  if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto capacity = graph_view.number_of_local_edge_partitions() *
                    (graph_view.local_edge_partition_segment_offsets(0) ? max_segments : 1);
    offset_vectors.reserve(capacity);
    value_vectors.reserve(capacity);

    for (size_t i = 0; i < capacity; ++i) {
      offset_vectors.emplace_back(0, handle.get_stream());
      value_vectors.emplace_back(0, handle.get_stream());
    }
  }

  if (stream_pool_indices) { handle.sync_stream(); }

  // 3. proces local edge partitions

  auto edge_mask_view = graph_view.edge_mask_view();

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    auto major_init = ReduceOp::identity_element;
    if constexpr (update_major) {
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
          major_init = init;  // init is selected only when no edges return a valid value
        } else {
          major_init = (static_cast<int>(i) == minor_comm_rank) ? init : ReduceOp::identity_element;
        }
      } else {
        major_init = init;
      }
    }

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    auto loop_stream =
      stream_pool_indices
        ? handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size())
        : handle.get_stream();

    auto edge_partition_key_first  = sorted_unique_key_first;
    auto edge_partition_key_last   = sorted_unique_nzd_key_last;
    auto edge_partition_key_buffer = allocate_optional_dataframe_buffer<
      std::conditional_t<GraphViewType::is_multi_gpu && use_input_key, key_t, void>>(0,
                                                                                     loop_stream);
    std::conditional_t<use_input_key, std::optional<std::vector<size_t>>, std::byte /* dummy */>
      key_segment_offsets{};
    if constexpr (use_input_key) {
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();
        if (minor_comm_size > 1) {
          auto const minor_comm_rank = minor_comm.get_rank();

          resize_optional_dataframe_buffer(
            edge_partition_key_buffer, local_key_list_sizes[i], loop_stream);

          if constexpr (use_bitmap) {
            std::variant<raft::device_span<uint32_t const>, decltype(sorted_unique_key_first)>
              v_list{};
            if (use_bitmap_flags[i]) {
              v_list = raft::device_span<uint32_t const>((*key_list_bitmap).data(),
                                                         (*key_list_bitmap).size());
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
                                     static_cast<size_t>(thrust::distance(
                                       sorted_unique_key_first, sorted_unique_nzd_key_last)),
                                     static_cast<int>(i),
                                     loop_stream);
          } else {
            device_bcast(minor_comm,
                         sorted_unique_key_first,
                         get_dataframe_buffer_begin(edge_partition_key_buffer),
                         local_key_list_sizes[i],
                         static_cast<int>(i),
                         loop_stream);
          }

          edge_partition_key_first = get_dataframe_buffer_begin(edge_partition_key_buffer);
          edge_partition_key_last  = get_dataframe_buffer_end(edge_partition_key_buffer);
        }
      }
      if (segment_offsets) {
        key_segment_offsets = compute_key_segment_offsets(
          edge_partition_key_first,
          edge_partition_key_last,
          raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
          edge_partition.major_range_first(),
          graph_view.use_dcs(),
          loop_stream);
      } else {
        key_segment_offsets = std::nullopt;
      }
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(loop_stream));

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

    auto major_buffer_first =
      get_dataframe_buffer_begin(major_tmp_buffers[i % major_tmp_buffers.size()]);

    std::conditional_t<GraphViewType::is_multi_gpu,
                       std::conditional_t<update_major,
                                          decltype(major_buffer_first),
                                          edge_partition_minor_output_device_view_t>,
                       VertexValueOutputIterator>
      output_buffer{};
    if constexpr (GraphViewType::is_multi_gpu) {
      if constexpr (update_major) {
        output_buffer = major_buffer_first;
      } else {
        output_buffer = edge_partition_minor_output_device_view_t(minor_tmp_buffer->mutable_view());
      }
    } else {
      output_buffer = vertex_value_output_first;
    }

    using segment_key_iterator_t =
      std::conditional_t<use_input_key,
                         decltype(edge_partition_key_first),
                         decltype(thrust::make_counting_iterator(vertex_t{0}))>;

    if (segment_offsets) {
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);

      std::vector<size_t> h_offsets{};
      if constexpr (use_input_key) {
        h_offsets = (*key_segment_offsets);
      } else {
        h_offsets.resize((*segment_offsets).size());
        std::transform((*segment_offsets).begin(),
                       (*segment_offsets).end(),
                       h_offsets.begin(),
                       [](vertex_t offset) { return static_cast<size_t>(offset); });
      }

      // FIXME: we may further improve performance by 1) individually tuning block sizes for
      // different segments; and 2) adding one more segment for very high degree vertices and
      // running segmented reduction
      if (edge_partition.dcs_nzd_vertex_count()) {
        auto exec_stream =
          stream_pool_indices
            ? handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size())
            : handle.get_stream();

        if constexpr (update_major && !use_input_key) {  // this is necessary as we don't visit
                                                         // every vertex in the hypersparse segment
          thrust::fill(rmm::exec_policy(exec_stream),
                       output_buffer + h_offsets[3],
                       output_buffer + h_offsets[4],
                       major_init);
        }

        auto segment_size = use_input_key
                              ? (h_offsets[4] - h_offsets[3])
                              : static_cast<size_t>(*(edge_partition.dcs_nzd_vertex_count()));
        if (segment_size > 0) {
          raft::grid_1d_thread_t update_grid(segment_size,
                                             detail::per_v_transform_reduce_e_kernel_block_size,
                                             handle.get_device_properties().maxGridSize[0]);
          auto segment_output_buffer = output_buffer;
          if constexpr (update_major) { segment_output_buffer += h_offsets[3]; }
          auto segment_key_first = edge_partition_key_first;
          auto segment_key_last  = edge_partition_key_last;
          if constexpr (use_input_key) {
            segment_key_first += h_offsets[3];
            segment_key_last += h_offsets[4];
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
              reduce_op);
        }
      }
      if (h_offsets[3] - h_offsets[2]) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 1) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_thread_t update_grid(h_offsets[3] - h_offsets[2],
                                           detail::per_v_transform_reduce_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        auto segment_output_buffer = output_buffer;
        if constexpr (update_major) { segment_output_buffer += h_offsets[2]; }
        segment_key_iterator_t segment_key_first{};
        if constexpr (use_input_key) {
          segment_key_first = edge_partition_key_first;
        } else {
          segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
        }
        segment_key_first += h_offsets[2];
        auto num_keys = h_offsets[3] - h_offsets[2];
        detail::per_v_transform_reduce_e_low_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            segment_key_first,
            segment_key_first + (h_offsets[3] - h_offsets[2]),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            segment_output_buffer,
            e_op,
            major_init,
            reduce_op);
      }
      if (h_offsets[2] - h_offsets[1] > 0) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 2) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_warp_t update_grid(h_offsets[2] - h_offsets[1],
                                         detail::per_v_transform_reduce_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        auto segment_output_buffer = output_buffer;
        if constexpr (update_major) { segment_output_buffer += h_offsets[1]; }
        segment_key_iterator_t segment_key_first{};
        if constexpr (use_input_key) {
          segment_key_first = edge_partition_key_first;
        } else {
          segment_key_first = thrust::make_counting_iterator(edge_partition.major_range_first());
        }
        segment_key_first += h_offsets[1];
        detail::per_v_transform_reduce_e_mid_degree<update_major, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
            edge_partition,
            segment_key_first,
            segment_key_first + (h_offsets[2] - h_offsets[1]),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            segment_output_buffer,
            e_op,
            major_init,
            ReduceOp::identity_element,
            reduce_op);
      }
      if (h_offsets[1] > 0) {
        auto exec_stream = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((i * max_segments + 3) %
                                                                  (*stream_pool_indices).size())
                             : handle.get_stream();
        raft::grid_1d_block_t update_grid(h_offsets[1],
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
            segment_key_first + h_offsets[1],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            output_buffer,
            e_op,
            major_init,
            ReduceOp::identity_element,
            reduce_op);
      }
    } else {
      size_t num_keys{};
      if constexpr (use_input_key) {
        num_keys =
          static_cast<size_t>(thrust::distance(edge_partition_key_first, edge_partition_key_last));
      } else {
        num_keys = static_cast<size_t>(edge_partition.major_range_size());
      }

      if (edge_partition.major_range_size() > 0) {
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
            reduce_op);
      }
    }

    if constexpr (GraphViewType::is_multi_gpu && update_major) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      auto const minor_comm_size = minor_comm.get_size();

      if (segment_offsets && stream_pool_indices) {
        if ((*segment_offsets)[4] - (*segment_offsets)[3] > 0) {
          auto segment_stream =
            handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[3];
          auto segment_size   = (*segment_offsets)[4] - (*segment_offsets)[3];
          if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
            auto [offsets, values] = compute_offset_value_pairs<vertex_t>(
              minor_comm,
              major_buffer_first + segment_offset,
              major_buffer_first + (segment_offset + segment_size),
              static_cast<int>(i),
              subgroup_size,
              init,
              segment_stream);
            offset_vectors[i * max_segments + 3] = std::move(offsets);
            value_vectors[i * max_segments + 3]  = std::move(values);
          } else {
            device_reduce(minor_comm,
                          major_buffer_first + segment_offset,
                          vertex_value_output_first + segment_offset,
                          segment_size,
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(i),
                          segment_stream);
          }
        }
        if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 1) %
                                                                   (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[2];
          auto segment_size   = (*segment_offsets)[3] - (*segment_offsets)[2];
          if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
            auto [offsets, values] = compute_offset_value_pairs<vertex_t>(
              minor_comm,
              major_buffer_first + segment_offset,
              major_buffer_first + (segment_offset + segment_size),
              static_cast<int>(i),
              subgroup_size,
              init,
              segment_stream);
            offset_vectors[i * max_segments + 2] = std::move(offsets);
            value_vectors[i * max_segments + 2]  = std::move(values);
          } else {
            device_reduce(minor_comm,
                          major_buffer_first + segment_offset,
                          vertex_value_output_first + segment_offset,
                          segment_size,
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(i),
                          segment_stream);
          }
        }
        if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 2) %
                                                                   (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[1];
          auto segment_size   = (*segment_offsets)[2] - (*segment_offsets)[1];
          if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
            auto [offsets, values] = compute_offset_value_pairs<vertex_t>(
              minor_comm,
              major_buffer_first + segment_offset,
              major_buffer_first + (segment_offset + segment_size),
              static_cast<int>(i),
              subgroup_size,
              init,
              segment_stream);
            offset_vectors[i * max_segments + 1] = std::move(offsets);
            value_vectors[i * max_segments + 1]  = std::move(values);
          } else {
            device_reduce(minor_comm,
                          major_buffer_first + segment_offset,
                          vertex_value_output_first + segment_offset,
                          segment_size,
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(i),
                          segment_stream);
          }
        }
        if ((*segment_offsets)[1] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 3) %
                                                                   (*stream_pool_indices).size());
          auto segment_size   = (*segment_offsets)[1];
          if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
            auto [offsets, values] =
              compute_offset_value_pairs<vertex_t>(minor_comm,
                                                   major_buffer_first,
                                                   major_buffer_first + segment_size,
                                                   static_cast<int>(i),
                                                   subgroup_size,
                                                   init,
                                                   segment_stream);
            offset_vectors[i * max_segments] = std::move(offsets);
            value_vectors[i * max_segments]  = std::move(values);
          } else {
            device_reduce(minor_comm,
                          major_buffer_first,
                          vertex_value_output_first,
                          segment_size,
                          ReduceOp::compatible_raft_comms_op,
                          static_cast<int>(i),
                          segment_stream);
          }
        }
      } else {
        size_t reduction_size = static_cast<size_t>(
          segment_offsets ? *((*segment_offsets).rbegin() + 1) /* exclude the zero degree segment */
                          : edge_partition.major_range_size());
        if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {
          auto [offsets, values] =
            compute_offset_value_pairs<vertex_t>(minor_comm,
                                                 major_buffer_first,
                                                 major_buffer_first + reduction_size,
                                                 static_cast<int>(i),
                                                 subgroup_size,
                                                 init,
                                                 handle.get_stream());
          offset_vectors[i] = std::move(offsets);
          value_vectors[i]  = std::move(values);
        } else {
          device_reduce(minor_comm,
                        major_buffer_first,
                        vertex_value_output_first,
                        reduction_size,
                        ReduceOp::compatible_raft_comms_op,
                        static_cast<int>(i),
                        handle.get_stream());
        }
      }
    }

    if (stream_pool_indices && ((i + 1) % major_tmp_buffers.size() == 0)) {
      handle.sync_stream_pool(
        *stream_pool_indices);  // to prevent buffer over-write (this can happen as
                                // *segment_offsets do not necessarily coincide in different edge
                                // partitions).
    }
  }

  if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }

  // 4. communication

  if constexpr (update_major && std::is_same_v<ReduceOp, reduce_op::any<T>>) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();

    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);

      if (segment_offsets && stream_pool_indices) {
        if ((*segment_offsets)[4] - (*segment_offsets)[3] > 0) {
          auto segment_stream =
            handle.get_stream_from_stream_pool((i * max_segments) % (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[3];
          gather_offset_value_pairs_and_update_vertex_value_output(
            minor_comm,
            std::move(offset_vectors[i * max_segments + 3]),
            std::move(value_vectors[i * max_segments + 3]),
            vertex_value_output_first + segment_offset,
            static_cast<int>(i),
            segment_stream);
        }
        if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 1) %
                                                                   (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[2];
          gather_offset_value_pairs_and_update_vertex_value_output(
            minor_comm,
            std::move(offset_vectors[i * max_segments + 2]),
            std::move(value_vectors[i * max_segments + 2]),
            vertex_value_output_first + segment_offset,
            static_cast<int>(i),
            segment_stream);
        }
        if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 2) %
                                                                   (*stream_pool_indices).size());
          auto segment_offset = (*segment_offsets)[1];
          gather_offset_value_pairs_and_update_vertex_value_output(
            minor_comm,
            std::move(offset_vectors[i * max_segments + 1]),
            std::move(value_vectors[i * max_segments + 1]),
            vertex_value_output_first + segment_offset,
            static_cast<int>(i),
            segment_stream);
        }
        if ((*segment_offsets)[1] > 0) {
          auto segment_stream = handle.get_stream_from_stream_pool((i * max_segments + 3) %
                                                                   (*stream_pool_indices).size());
          gather_offset_value_pairs_and_update_vertex_value_output(
            minor_comm,
            std::move(offset_vectors[i * max_segments]),
            std::move(value_vectors[i * max_segments]),
            vertex_value_output_first,
            static_cast<int>(i),
            segment_stream);
        }
      } else {
        gather_offset_value_pairs_and_update_vertex_value_output(minor_comm,
                                                                 std::move(offset_vectors[i]),
                                                                 std::move(value_vectors[i]),
                                                                 vertex_value_output_first,
                                                                 static_cast<int>(i),
                                                                 handle.get_stream());
      }
    }
  }

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
                      vertex_value_output_first,
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
                      vertex_value_output_first,
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

/**
 * @brief Iterate over every vertex's incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transform_reduce.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to
 * fill the wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_incoming_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = true;

  detail::per_v_transform_reduce_e<incoming>(handle,
                                             graph_view,
                                             static_cast<void*>(nullptr),
                                             static_cast<void*>(nullptr),
                                             edge_src_value_input,
                                             edge_dst_value_input,
                                             edge_value_input,
                                             e_op,
                                             init,
                                             reduce_op,
                                             vertex_value_output_first);
}

/**
 * @brief For each (tagged-)vertex in the input (tagged-)vertex list, iterate over the incoming
 * edges to update (tagged-)vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to update
 * (tagged-)vertex properties.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be reduced with the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 incoming edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the (tagged-)vertex property variables for
 * the first (inclusive) (tagged-)vertex in @p key_list. `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p key_list.size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_incoming_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       KeyBucketType const& key_list,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  static_assert(GraphViewType::is_storage_transposed);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = true;

  detail::per_v_transform_reduce_e<incoming>(handle,
                                             graph_view,
                                             key_list.begin(),
                                             key_list.end(),
                                             edge_src_value_input,
                                             edge_dst_value_input,
                                             edge_value_input,
                                             e_op,
                                             init,
                                             reduce_op,
                                             vertex_value_output_first);
}

/**
 * @brief Iterate over every vertex's outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.local_vertex_partition_range_size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_outgoing_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = false;

  detail::per_v_transform_reduce_e<incoming>(handle,
                                             graph_view,
                                             static_cast<void*>(nullptr),
                                             static_cast<void*>(nullptr),
                                             edge_src_value_input,
                                             edge_dst_value_input,
                                             edge_value_input,
                                             e_op,
                                             init,
                                             reduce_op,
                                             vertex_value_output_first);
}

/**
 * @brief For each (tagged-)vertex in the input (tagged-)vertex list, iterate over the outgoing
 * edges to update (tagged-)vertex properties.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam KeyBucketType Type of the key bucket class which abstracts the current (tagged-)vertex
 * list.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam T Type of the initial value for per-vertex reduction.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param key_list KeyBucketType class object to store the (tagged-)vertex list to update
 * (tagged-)vertex properties.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced.
 * @param init Initial value to be reduced with the reduced @p e_op return values for each vertex.
 * If @p reduce_op is cugraph::reduce_op::any, init value is never selected except for the
 * (tagged-)vertices with 0 outgoing edges.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param vertex_value_output_first Iterator pointing to the (tagged-)vertex property variables for
 * the first (inclusive) (tagged-)vertex in @p key_list. `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p key_list.size().
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename GraphViewType,
          typename KeyBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T,
          typename VertexValueOutputIterator>
void per_v_transform_reduce_outgoing_e(raft::handle_t const& handle,
                                       GraphViewType const& graph_view,
                                       KeyBucketType const& key_list,
                                       EdgeSrcValueInputWrapper edge_src_value_input,
                                       EdgeDstValueInputWrapper edge_dst_value_input,
                                       EdgeValueInputWrapper edge_value_input,
                                       EdgeOp e_op,
                                       T init,
                                       ReduceOp reduce_op,
                                       VertexValueOutputIterator vertex_value_output_first,
                                       bool do_expensive_check = false)
{
  static_assert(!GraphViewType::is_storage_transposed);
  static_assert(KeyBucketType::is_sorted_unique);

  if (do_expensive_check) {
    // currently, nothing to do
  }

  constexpr bool incoming = false;

  detail::per_v_transform_reduce_e<incoming>(handle,
                                             graph_view,
                                             key_list.begin(),
                                             key_list.end(),
                                             edge_src_value_input,
                                             edge_dst_value_input,
                                             edge_value_input,
                                             e_op,
                                             init,
                                             reduce_op,
                                             vertex_value_output_first);
}

}  // namespace cugraph
