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

#include "prims/detail/multi_stream_utils.cuh"
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

  __shared__ edge_t
    warp_local_degree_inclusive_sums[extract_transform_v_frontier_e_kernel_block_size];
  __shared__ edge_t warp_key_local_edge_offsets[extract_transform_v_frontier_e_kernel_block_size];

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

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
    auto this_warp_inclusive_sum_last = this_warp_inclusive_sum_first + (max_key_idx - min_key_idx);

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

    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
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

        warp_push_buffer_elements(
          buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
      }
    } else {
      for (size_t i = lane_id; i < rounded_up_local_out_degree; i += raft::warp_size()) {
        e_op_result_t e_op_result{thrust::nullopt};
        if (i < static_cast<size_t>(local_out_degree)) { e_op_result = call_e_op(i); }

        warp_push_buffer_elements(
          buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename GraphViewType,
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

        warp_push_buffer_elements(
          buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
      }
    } else {
      for (size_t i = threadIdx.x; i < rounded_up_local_out_degree; i += blockDim.x) {
        e_op_result_t e_op_result{thrust::nullopt};
        if (i < static_cast<size_t>(local_out_degree)) { e_op_result = call_e_op(i); }

        warp_push_buffer_elements(
          buffer_key_output_first, buffer_value_output_first, buffer_idx, lane_id, e_op_result);
      }
    }

    idx += gridDim.x;
  }
}

#define EXTRACT_PERFORMANCE_MEASUREMENT 1

template <bool incoming,  // iterate over incoming edges (incoming == true) or outgoing edges
                          // (incoming == false)
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

  constexpr bool try_bitmap = GraphViewType::is_multi_gpu && std::is_same_v<key_t, vertex_t> &&
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

  [[maybe_unused]] constexpr auto max_segments =
    detail::num_sparse_segments_per_vertex_partition + size_t{1};

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

  std::optional<std::vector<size_t>> key_segment_offsets{std::nullopt};
  {  // drop zero degree vertices & compute key_segment_offsets
    size_t partition_idx{0};
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      partition_idx    = static_cast<size_t>(minor_comm.get_rank());
    }
    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(partition_idx);
    if (segment_offsets) {
      key_segment_offsets = compute_key_segment_offsets(
        frontier_key_first,
        frontier_key_last,
        raft::host_span<vertex_t const>((*segment_offsets).data(), (*segment_offsets).size()),
        graph_view.local_vertex_partition_range_first(),
        handle.get_stream());
      (*key_segment_offsets).back() = *((*key_segment_offsets).rbegin() + 1);
      frontier_key_last             = frontier_key_first + (*key_segment_offsets).back();
    }
  }

  // 2. compute local max_pushes

  size_t local_max_pushes{};
  {
    size_t partition_idx{};
    if constexpr (GraphViewType::is_multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      partition_idx              = static_cast<size_t>(minor_comm_rank);
    }
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(partition_idx));
    auto frontier_major_first =
      thrust_tuple_get_or_identity<decltype(frontier_key_first), 0>(frontier_key_first);
    auto frontier_major_last =
      thrust_tuple_get_or_identity<decltype(frontier_key_last), 0>(frontier_key_last);
    // for an edge-masked graph, we can pass edge mask to compute tighter bound (at the expense of
    // additional computing)
    local_max_pushes = edge_partition.compute_number_of_edges(
      frontier_major_first, frontier_major_last, handle.get_stream());
  }

  // 3. communication over minor_comm

  std::vector<size_t> local_frontier_sizes{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_frontier_range_firsts{};
  std::conditional_t<try_bitmap, std::vector<vertex_t>, std::byte /* dummy */>
    local_frontier_range_lasts{};
  std::optional<std::vector<std::vector<size_t>>> key_segment_offset_vectors{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    local_frontier_sizes       = host_scalar_allgather(
      minor_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
    if constexpr (try_bitmap) {
      std::array<vertex_t, 2> v_list_range = {vertex_t{0}, vertex_t{0}};
      auto v_list_size =
        static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last));
      if (v_list_size > 0) {
        rmm::device_uvector<vertex_t> tmps(2, handle.get_stream());
        thrust::tabulate(handle.get_thrust_policy(),
                         tmps.begin(),
                         tmps.end(),
                         [frontier_key_first, v_list_size] __device__(size_t i) {
                           return (i == 0) ? *frontier_key_first
                                           : (*(frontier_key_first + (v_list_size - 1)) + 1);
                         });
        raft::update_host(v_list_range.data(), tmps.data(), 2, handle.get_stream());
        handle.sync_stream();
      }
      local_frontier_range_firsts =
        host_scalar_allgather(minor_comm, v_list_range[0], handle.get_stream());
      local_frontier_range_lasts =
        host_scalar_allgather(minor_comm, v_list_range[1], handle.get_stream());
    }
    if (key_segment_offsets) {
      rmm::device_uvector<size_t> d_key_segment_offsets((*key_segment_offsets).size(),
                                                        handle.get_stream());
      raft::update_device(d_key_segment_offsets.data(),
                          (*key_segment_offsets).data(),
                          (*key_segment_offsets).size(),
                          handle.get_stream());
      rmm::device_uvector<size_t> d_aggregate_key_segment_offsets(
        minor_comm_size * d_key_segment_offsets.size(), handle.get_stream());
      std::vector<size_t> rx_counts(minor_comm_size, d_key_segment_offsets.size());
      std::vector<size_t> rx_displacements(minor_comm_size);
      std::exclusive_scan(rx_counts.begin(), rx_counts.end(), rx_displacements.begin(), size_t{0});
      device_allgatherv(minor_comm,
                        d_key_segment_offsets.data(),
                        d_aggregate_key_segment_offsets.data(),
                        rx_counts,
                        rx_displacements,
                        handle.get_stream());
      std::vector<size_t> h_aggregate_key_segment_offsets(d_aggregate_key_segment_offsets.size());
      raft::update_host(h_aggregate_key_segment_offsets.data(),
                        d_aggregate_key_segment_offsets.data(),
                        d_aggregate_key_segment_offsets.size(),
                        handle.get_stream());
      handle.sync_stream();
      key_segment_offset_vectors = std::vector<std::vector<size_t>>(minor_comm_size);
      for (int i = 0; i < minor_comm_size; ++i) {
        (*key_segment_offset_vectors)[i] = std::vector<size_t>(
          h_aggregate_key_segment_offsets.begin() + i * (*key_segment_offsets).size(),
          h_aggregate_key_segment_offsets.begin() + (i + 1) * (*key_segment_offsets).size());
      }
    }
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
    if (key_segment_offsets) {
      key_segment_offset_vectors       = std::vector<std::vector<size_t>>(1);
      (*key_segment_offset_vectors)[0] = *key_segment_offsets;
    }
  }

  // update frontier bitmap (used to reduce broadcast bandwidth size)

  std::
    conditional_t<try_bitmap, std::optional<rmm::device_uvector<uint32_t>>, std::byte /* dummy */>
      frontier_bitmap{};
  if constexpr (try_bitmap) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();
    if (minor_comm_size > 1) {
      auto const minor_comm_rank = minor_comm.get_rank();
      double avg_fill_ratio{0.0};
      for (int i = 0; i < minor_comm_size; ++i) {
        auto num_keys   = static_cast<double>(local_frontier_sizes[i]);
        auto range_size = local_frontier_range_lasts[i] - local_frontier_range_firsts[i];
        avg_fill_ratio +=
          (range_size > 0) ? (num_keys / static_cast<double>(range_size)) : double{0.0};
      }
      avg_fill_ratio /= static_cast<double>(minor_comm_size);

      constexpr double threshold_ratio =
        8.0 /* tuning parameter */ / static_cast<double>(sizeof(vertex_t) * 8);
      if (avg_fill_ratio > threshold_ratio) {
        frontier_bitmap =
          compute_vertex_list_bitmap_info(frontier_key_first,
                                          frontier_key_last,
                                          local_frontier_range_firsts[minor_comm_rank],
                                          local_frontier_range_lasts[minor_comm_rank],
                                          handle.get_stream());
      }
    }
  }

  // set-up stream ppol

  std::optional<std::vector<size_t>> stream_pool_indices{std::nullopt};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto partition_idx         = static_cast<size_t>(minor_comm_rank);

    if (graph_view.local_edge_partition_segment_offsets(partition_idx) &&
        (handle.get_stream_pool_size() >= max_segments)) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto max_tmp_buffer_size = static_cast<size_t>(
        static_cast<double>(handle.get_device_properties().totalGlobalMem) * 0.05);

      auto aggregate_major_range_size = host_scalar_allreduce(
        comm,
        static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
        raft::comms::op_t::SUM,
        handle.get_stream());
      auto aggregate_max_pushes = host_scalar_allreduce(
        comm,
        local_max_pushes,
        raft::comms::op_t::SUM,
        handle.get_stream());  // this is approximate as we only consider local edges for
                               // [frontier_key_first, frontier_key_last), note that neighbor lists
                               // are partitioned if minor_comm_size > 1

      size_t key_size{0};
      if constexpr (std::is_arithmetic_v<key_t>) {
        key_size = sizeof(key_t);
      } else {
        key_size = cugraph::sum_thrust_tuple_element_sizes<key_t>();
      }
      size_t output_key_size{0};
      if constexpr (!std::is_same_v<output_key_t, void>) {
        if constexpr (std::is_arithmetic_v<output_key_t>) {
          output_key_size = sizeof(output_key_t);
        } else {
          output_key_size = cugraph::sum_thrust_tuple_element_sizes<output_key_t>();
        }
      }
      size_t output_value_size{0};
      if constexpr (!std::is_same_v<output_value_t, void>) {
        if constexpr (std::is_arithmetic_v<output_value_t>) {
          output_value_size = sizeof(output_value_t);
        } else {
          output_value_size = cugraph::sum_thrust_tuple_element_sizes<output_value_t>();
        }
      }
      auto approx_tmp_buffer_size_per_edge_partition =
        (aggregate_major_range_size / comm_size) * key_size +
        (aggregate_max_pushes / comm_size) * (output_key_size + output_value_size);

      stream_pool_indices = init_stream_pool_indices(max_tmp_buffer_size,
                                                     approx_tmp_buffer_size_per_edge_partition,
                                                     graph_view.number_of_local_edge_partitions(),
                                                     max_segments,
                                                     handle.get_stream_pool_size());
      if ((*stream_pool_indices).size() <= 1) { stream_pool_indices = std::nullopt; }
    }
  }

  size_t num_concurrent_loops{1};
  if (stream_pool_indices) {
    assert(((*stream_pool_indices).size() % max_segments) == 0);
    num_concurrent_loops = (*stream_pool_indices).size() / max_segments;
  }

  if (stream_pool_indices) { handle.sync_stream(); }

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
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i += num_concurrent_loops) {
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime0 = std::chrono::steady_clock::now();
#endif
    auto loop_count =
      std::min(num_concurrent_loops, graph_view.number_of_local_edge_partitions() - i);

    std::conditional_t<GraphViewType::is_multi_gpu,
                       std::vector<dataframe_buffer_type_t<key_t>>,
                       std::byte /* dummy */>
      edge_partition_key_buffers{};
    if constexpr (GraphViewType::is_multi_gpu) {
      edge_partition_key_buffers.reserve(loop_count);
      for (size_t j = 0; j < loop_count; ++j) {
        auto partition_idx = i + j;
        auto loop_stream   = stream_pool_indices
                               ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                               : handle.get_stream();

        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        auto const minor_comm_size = minor_comm.get_size();

        auto edge_partition_key_buffer = allocate_dataframe_buffer<key_t>(
          minor_comm_size > 1 ? local_frontier_sizes[partition_idx] : size_t{0}, loop_stream);
        if (size_dataframe_buffer(edge_partition_key_buffer) > 0) {
          auto edge_partition =
            edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
              graph_view.local_edge_partition_view(partition_idx));

          if constexpr (try_bitmap) {
            std::variant<raft::device_span<uint32_t const>, decltype(frontier_key_first)> v_list{};
            if (frontier_bitmap) {
              v_list = (static_cast<int>(partition_idx) == minor_comm_rank)
                         ? raft::device_span<uint32_t const>((*frontier_bitmap).data(),
                                                             (*frontier_bitmap).size())
                         : raft::device_span<uint32_t const>(static_cast<uint32_t const*>(nullptr),
                                                             size_t{0});
            } else {
              v_list = frontier_key_first;
            }
            device_bcast_vertex_list(minor_comm,
                                     v_list,
                                     get_dataframe_buffer_begin(edge_partition_key_buffer),
                                     local_frontier_range_firsts[partition_idx],
                                     local_frontier_range_lasts[partition_idx],
                                     local_frontier_sizes[partition_idx],
                                     static_cast<int>(partition_idx),
                                     loop_stream);
          } else {
            device_bcast(minor_comm,
                         frontier_key_first,
                         get_dataframe_buffer_begin(edge_partition_key_buffer),
                         local_frontier_sizes[partition_idx],
                         static_cast<int>(partition_idx),
                         loop_stream);
          }
        }
        edge_partition_key_buffers.push_back(std::move(edge_partition_key_buffer));
      }
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime1 = std::chrono::steady_clock::now();
#endif
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
    auto subtime2 = std::chrono::steady_clock::now();
#endif

    std::vector<optional_dataframe_buffer_type_t<output_key_t>> output_key_buffers{};
    output_key_buffers.reserve(loop_count);
    std::vector<optional_dataframe_buffer_type_t<output_value_t>> output_value_buffers{};
    output_value_buffers.reserve(loop_count);
    std::vector<rmm::device_scalar<size_t>> output_buffer_idx_scalars{};
    output_buffer_idx_scalars.reserve(loop_count);
    for (size_t j = 0; j < loop_count; ++j) {
      auto partition_idx = i + j;
      auto loop_stream   = stream_pool_indices
                             ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                             : handle.get_stream();

      auto edge_partition =
        edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
          graph_view.local_edge_partition_view(partition_idx));

      auto edge_partition_max_pushes = local_max_pushes;
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_rank = minor_comm.get_rank();
        auto const minor_comm_size = minor_comm.get_size();
        if (minor_comm_size > 1) {
          if (static_cast<int>(partition_idx) != minor_comm_rank) {
            auto edge_partition_frontier_key_first =
              get_dataframe_buffer_begin(edge_partition_key_buffers[j]);
            auto edge_partition_frontier_key_last =
              get_dataframe_buffer_end(edge_partition_key_buffers[j]);
            auto edge_partition_frontier_major_first =
              thrust_tuple_get_or_identity<decltype(edge_partition_frontier_key_first), 0>(
                edge_partition_frontier_key_first);
            auto edge_partition_frontier_major_last =
              thrust_tuple_get_or_identity<decltype(edge_partition_frontier_key_last), 0>(
                edge_partition_frontier_key_last);
            edge_partition_max_pushes = edge_partition.compute_number_of_edges(
              edge_partition_frontier_major_first, edge_partition_frontier_major_last, loop_stream);
            // FIXME: compute_number_of_edges() implicitly synchronizes to copy the results to host
            edge_partition_max_pushes = edge_partition.compute_number_of_edges(
              edge_partition_frontier_major_first, edge_partition_frontier_major_last, loop_stream);
          }
        }
      }

      output_key_buffers.push_back(
        allocate_optional_dataframe_buffer<output_key_t>(edge_partition_max_pushes, loop_stream));
      output_value_buffers.push_back(
        allocate_optional_dataframe_buffer<output_value_t>(edge_partition_max_pushes, loop_stream));
      output_buffer_idx_scalars.push_back(rmm::device_scalar<size_t>(size_t{0}, loop_stream));
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime3 = std::chrono::steady_clock::now();
#endif
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
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

      auto edge_partition_frontier_key_first = frontier_key_first;
      auto edge_partition_frontier_key_last  = frontier_key_last;
      if constexpr (GraphViewType::is_multi_gpu) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();
        if (minor_comm_size > 1) {
          edge_partition_frontier_key_first =
            get_dataframe_buffer_begin(edge_partition_key_buffers[j]);
          edge_partition_frontier_key_last =
            get_dataframe_buffer_end(edge_partition_key_buffers[j]);
        }
      }

      auto& tmp_key_buffer   = output_key_buffers[j];
      auto& tmp_value_buffer = output_value_buffers[j];
      auto& tmp_buffer_idx   = output_buffer_idx_scalars[j];

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

      if (key_segment_offset_vectors) {
        auto const& key_segment_offsets = (*key_segment_offset_vectors)[partition_idx];

        if (key_segment_offsets[1] > 0) {
          auto exec_stream =
            edge_partition_stream_pool_indices
              ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[0])
              : handle.get_stream();
          raft::grid_1d_block_t update_grid(key_segment_offsets[1],
                                            extract_transform_v_frontier_e_kernel_block_size,
                                            handle.get_device_properties().maxGridSize[0]);
          extract_transform_v_frontier_e_high_degree<GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
              edge_partition,
              edge_partition_frontier_key_first,
              edge_partition_frontier_key_first + key_segment_offsets[1],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_e_mask,
              get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
              get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
              tmp_buffer_idx.data(),
              e_op);
        }
        if (key_segment_offsets[2] - key_segment_offsets[1] > 0) {
          auto exec_stream =
            edge_partition_stream_pool_indices
              ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[1])
              : handle.get_stream();
          raft::grid_1d_warp_t update_grid(key_segment_offsets[2] - key_segment_offsets[1],
                                           extract_transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
          extract_transform_v_frontier_e_mid_degree<GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
              edge_partition,
              edge_partition_frontier_key_first + key_segment_offsets[1],
              edge_partition_frontier_key_first + key_segment_offsets[2],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_e_mask,
              get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
              get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
              tmp_buffer_idx.data(),
              e_op);
        }
        if (key_segment_offsets[3] - key_segment_offsets[2] > 0) {
          auto exec_stream =
            edge_partition_stream_pool_indices
              ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[2])
              : handle.get_stream();
          raft::grid_1d_thread_t update_grid(key_segment_offsets[3] - key_segment_offsets[2],
                                             extract_transform_v_frontier_e_kernel_block_size,
                                             handle.get_device_properties().maxGridSize[0]);
          extract_transform_v_frontier_e_hypersparse_or_low_degree<false, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
              edge_partition,
              edge_partition_frontier_key_first + key_segment_offsets[2],
              edge_partition_frontier_key_first + key_segment_offsets[3],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_e_mask,
              get_optional_dataframe_buffer_begin<output_key_t>(tmp_key_buffer),
              get_optional_dataframe_buffer_begin<output_value_t>(tmp_value_buffer),
              tmp_buffer_idx.data(),
              e_op);
        }
        if (edge_partition.dcs_nzd_vertex_count() &&
            (key_segment_offsets[4] - key_segment_offsets[3] > 0)) {
          auto exec_stream =
            edge_partition_stream_pool_indices
              ? handle.get_stream_from_stream_pool((*edge_partition_stream_pool_indices)[3])
              : handle.get_stream();
          raft::grid_1d_thread_t update_grid(key_segment_offsets[4] - key_segment_offsets[3],
                                             extract_transform_v_frontier_e_kernel_block_size,
                                             handle.get_device_properties().maxGridSize[0]);
          extract_transform_v_frontier_e_hypersparse_or_low_degree<true, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, exec_stream>>>(
              edge_partition,
              edge_partition_frontier_key_first + key_segment_offsets[3],
              edge_partition_frontier_key_first + key_segment_offsets[4],
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
        assert(!edge_partition_stream_pool_indices);
        if (local_frontier_sizes[partition_idx] > 0) {
          raft::grid_1d_thread_t update_grid(local_frontier_sizes[partition_idx],
                                             extract_transform_v_frontier_e_kernel_block_size,
                                             handle.get_device_properties().maxGridSize[0]);

          extract_transform_v_frontier_e_hypersparse_or_low_degree<false, GraphViewType>
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
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime5 = std::chrono::steady_clock::now();
#endif
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime6 = std::chrono::steady_clock::now();
#endif

    std::vector<size_t> tmp_buffer_sizes(loop_count);
    for (size_t j = 0; j < loop_count; ++j) {
      auto loop_stream = stream_pool_indices
                           ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                           : handle.get_stream();

      auto& tmp_buffer_idx = output_buffer_idx_scalars[j];
      // FIXME: tmp_buffer_idx.value() implicitly synchronizes to copy the results to host
      tmp_buffer_sizes[j] = tmp_buffer_idx.value(loop_stream);
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime7 = std::chrono::steady_clock::now();
#endif
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
    auto subtime8 = std::chrono::steady_clock::now();
#endif

    for (size_t j = 0; j < loop_count; ++j) {
      auto loop_stream = stream_pool_indices
                           ? handle.get_stream_from_stream_pool((*stream_pool_indices)[j])
                           : handle.get_stream();

      auto tmp_buffer_size = tmp_buffer_sizes[j];
      if (tmp_buffer_size > 0) {
        auto& tmp_key_buffer   = output_key_buffers[j];
        auto& tmp_value_buffer = output_value_buffers[j];

        resize_optional_dataframe_buffer<output_key_t>(
          tmp_key_buffer, tmp_buffer_size, loop_stream);
        shrink_to_fit_optional_dataframe_buffer<output_key_t>(tmp_key_buffer, loop_stream);

        resize_optional_dataframe_buffer<output_value_t>(
          tmp_value_buffer, tmp_buffer_size, loop_stream);
        shrink_to_fit_optional_dataframe_buffer<output_value_t>(tmp_value_buffer, loop_stream);

        key_buffers.push_back(std::move(tmp_key_buffer));
        value_buffers.push_back(std::move(tmp_value_buffer));
      }
    }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime9 = std::chrono::steady_clock::now();
#endif
    if (stream_pool_indices) { handle.sync_stream_pool(*stream_pool_indices); }
#if EXTRACT_PERFORMANCE_MEASUREMENT  // FIXME: delete
    auto subtime10                        = std::chrono::steady_clock::now();
    std::chrono::duration<double> subdur0 = subtime1 - subtime0;
    std::chrono::duration<double> subdur1 = subtime2 - subtime1;
    std::chrono::duration<double> subdur2 = subtime3 - subtime2;
    std::chrono::duration<double> subdur3 = subtime4 - subtime3;
    std::chrono::duration<double> subdur4 = subtime5 - subtime4;
    std::chrono::duration<double> subdur5 = subtime6 - subtime5;
    std::chrono::duration<double> subdur6 = subtime7 - subtime6;
    std::chrono::duration<double> subdur7 = subtime8 - subtime7;
    std::chrono::duration<double> subdur8 = subtime9 - subtime8;
    std::chrono::duration<double> subdur9 = subtime10 - subtime9;
    std::cerr << "sub (extract) took (" << subdur0.count() << "," << subdur1.count() << ","
              << subdur2.count() << "," << subdur3.count() << "," << subdur4.count() << ","
              << subdur5.count() << "," << subdur6.count() << "," << subdur7.count() << ","
              << subdur8.count() << "," << subdur9.count() << ")" << std::endl;
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
    /* nothing to do */
  } else if (key_buffers.size() == 1) {
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
    // FIXME: this copy can be performed in multiple streams
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
  std::cerr << "\t\t"
            << "detail::extract (pre,fill,concat) took (" << dur0.count() << "," << dur1.count()
            << "," << dur2.count() << ")" << std::endl;
#endif

  return std::make_tuple(std::move(key_buffer), std::move(value_buffer));
}

}  // namespace detail

}  // namespace cugraph
