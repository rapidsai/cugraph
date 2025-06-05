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

#include "prims/detail/partition_v_frontier.cuh"
#include "prims/property_op_utils.cuh"

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_v_frontier_e_kernel_block_size = 128;

template <typename key_t,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__device__ void transform_v_frontier_e_update_buffer_element(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>& edge_partition,
  key_t key,
  typename GraphViewType::vertex_type major_offset,
  typename GraphViewType::vertex_type minor,
  typename GraphViewType::edge_type edge_offset,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgeOp e_op,
  ValueIterator value_iter)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
  std::conditional_t<GraphViewType::is_storage_transposed, vertex_t, key_t> key_or_src{};
  std::conditional_t<GraphViewType::is_storage_transposed, key_t, vertex_t> key_or_dst{};
  if constexpr (GraphViewType::is_storage_transposed) {
    key_or_src = minor;
    key_or_dst = key;
  } else {
    key_or_src = key;
    key_or_dst = minor;
  }
  auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
  auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

  *value_iter = e_op(key_or_src,
                     key_or_dst,
                     edge_partition_src_value_input.get(src_offset),
                     edge_partition_dst_value_input.get(dst_offset),
                     edge_partition_e_value_input.get(edge_offset));
}

template <bool hypersparse,
          typename GraphViewType,
          typename KeyIterator,
          typename IndexIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ static void transform_v_frontier_e_hypersparse_or_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator edge_partition_frontier_key_first,
  IndexIterator edge_partition_frontier_key_index_first,
  IndexIterator edge_partition_frontier_key_index_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask,
  raft::device_span<size_t const> edge_partition_frontier_local_degree_offsets,
  EdgeOp e_op,
  ValueIterator value_first)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(cuda::std::distance(edge_partition_frontier_key_index_first,
                                                       edge_partition_frontier_key_index_last))) {
    auto key_idx      = *(edge_partition_frontier_key_index_first + idx);
    auto key          = *(edge_partition_frontier_key_first + key_idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    if constexpr (hypersparse) {
      auto major_idx = edge_partition.major_idx_from_major_nocheck(major);
      if (major_idx) {
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(*major_idx);
      } else {
        local_degree = edge_t{0};
      }
    } else {
      thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    }
    auto this_key_value_first = value_first + edge_partition_frontier_local_degree_offsets[key_idx];
    if (edge_partition_e_mask) {
      edge_t counter{0};
      for (edge_t i = 0; i < local_degree; ++i) {
        if ((*edge_partition_e_mask).get(edge_offset + i)) {
          transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
            edge_partition,
            key,
            major_offset,
            indices[i],
            edge_offset + i,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            e_op,
            this_key_value_first + counter);
          ++counter;
        }
      }
    } else {
      for (edge_t i = 0; i < local_degree; ++i) {
        transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
          edge_partition,
          key,
          major_offset,
          indices[i],
          edge_offset + i,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          this_key_value_first + i);
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename IndexIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ static void transform_v_frontier_e_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator edge_partition_frontier_key_first,
  IndexIterator edge_partition_frontier_key_index_first,
  IndexIterator edge_partition_frontier_key_index_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask,
  raft::device_span<size_t const> edge_partition_frontier_local_degree_offsets,
  EdgeOp e_op,
  ValueIterator value_first)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_v_frontier_e_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  size_t idx         = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(cuda::std::distance(edge_partition_frontier_key_index_first,
                                                       edge_partition_frontier_key_index_last))) {
    auto key_idx      = *(edge_partition_frontier_key_index_first + idx);
    auto key          = *(edge_partition_frontier_key_first + key_idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    auto this_key_value_first = value_first + edge_partition_frontier_local_degree_offsets[key_idx];
    if (edge_partition_e_mask) {
      auto rounded_up_local_degree =
        ((static_cast<size_t>(local_degree) + (raft::warp_size() - 1)) / raft::warp_size()) *
        raft::warp_size();
      edge_t base_offset{0};
      for (edge_t i = lane_id; i < rounded_up_local_degree; i += raft::warp_size()) {
        auto valid  = (i < local_degree) && (*edge_partition_e_mask).get(edge_offset + i);
        auto ballot = __ballot_sync(raft::warp_full_mask(), valid ? uint32_t{1} : uint32_t{0});
        if (valid) {
          auto intra_warp_offset = __popc(ballot & ~(raft::warp_full_mask() << lane_id));
          transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
            edge_partition,
            key,
            major_offset,
            indices[i],
            edge_offset + i,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            e_op,
            this_key_value_first + base_offset + intra_warp_offset);
        }
        base_offset += __popc(ballot);
      }
    } else {
      for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
        transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
          edge_partition,
          key,
          major_offset,
          indices[i],
          edge_offset + i,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          this_key_value_first + i);
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename GraphViewType,
          typename KeyIterator,
          typename IndexIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionEdgeMaskWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ static void transform_v_frontier_e_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  KeyIterator edge_partition_frontier_key_first,
  IndexIterator edge_partition_frontier_key_index_first,
  IndexIterator edge_partition_frontier_key_index_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask,
  raft::device_span<size_t const> edge_partition_frontier_local_degree_offsets,
  EdgeOp e_op,
  ValueIterator value_first)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockScan = cub::BlockScan<edge_t, transform_v_frontier_e_kernel_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __shared__ edge_t increment;

  while (idx < static_cast<size_t>(cuda::std::distance(edge_partition_frontier_key_index_first,
                                                       edge_partition_frontier_key_index_last))) {
    auto key_idx      = *(edge_partition_frontier_key_index_first + idx);
    auto key          = *(edge_partition_frontier_key_first + key_idx);
    auto major        = thrust_tuple_get_or_identity<key_t, 0>(key);
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    auto this_key_value_first = value_first + edge_partition_frontier_local_degree_offsets[key_idx];
    if (edge_partition_e_mask) {
      auto rounded_up_local_degree =
        ((static_cast<size_t>(local_degree) + (transform_v_frontier_e_kernel_block_size - 1)) /
         transform_v_frontier_e_kernel_block_size) *
        transform_v_frontier_e_kernel_block_size;
      edge_t base_offset{0};
      for (size_t i = threadIdx.x; i < rounded_up_local_degree; i += blockDim.x) {
        auto valid = (i < local_degree) && (*edge_partition_e_mask).get(edge_offset + i);
        edge_t intra_block_offset{};
        BlockScan(temp_storage).ExclusiveSum(valid ? edge_t{1} : edge_t{0}, intra_block_offset);
        if (valid) {
          transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
            edge_partition,
            key,
            major_offset,
            indices[i],
            edge_offset + i,
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            e_op,
            this_key_value_first + base_offset + intra_block_offset);
        }
        if (threadIdx.x == transform_v_frontier_e_kernel_block_size - 1) {
          increment = intra_block_offset + (valid ? edge_t{1} : edge_t{0});
        }
        __syncthreads();
        base_offset += increment;
      }
    } else {
      for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
        transform_v_frontier_e_update_buffer_element<key_t, GraphViewType>(
          edge_partition,
          key,
          major_offset,
          indices[i],
          edge_offset + i,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          this_key_value_first + i);
      }
    }

    idx += gridDim.x;
  }
}

// return std::tuple of e_op results and offsets
template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp>
auto transform_v_frontier_e(raft::handle_t const& handle,
                            GraphViewType const& graph_view,
                            KeyIterator aggregate_local_frontier_key_first,
                            EdgeSrcValueInputWrapper edge_src_value_input,
                            EdgeDstValueInputWrapper edge_dst_value_input,
                            EdgeValueInputWrapper edge_value_input,
                            EdgeOp e_op,
                            raft::host_span<size_t const> local_frontier_offsets)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using e_op_result_t =
    typename detail::edge_op_result_type<key_t,
                                         vertex_t,
                                         typename EdgeSrcValueInputWrapper::value_type,
                                         typename EdgeDstValueInputWrapper::value_type,
                                         typename EdgeValueInputWrapper::value_type,
                                         EdgeOp>::type;
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<e_op_result_t>::value);

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
    std::is_same_v<typename EdgeValueInputWrapper::value_type, cuda::std::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. update aggregate_local_frontier_local_degree_offsets

  auto aggregate_local_frontier_local_degree_offsets =
    rmm::device_uvector<size_t>(local_frontier_offsets.back() + 1, handle.get_stream());
  aggregate_local_frontier_local_degree_offsets.set_element_to_zero_async(
    aggregate_local_frontier_local_degree_offsets.size() - 1, handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    auto edge_partition_frontier_key_first =
      aggregate_local_frontier_key_first + local_frontier_offsets[i];
    auto edge_partition_frontier_major_first =
      thrust_tuple_get_or_identity<KeyIterator, 0>(edge_partition_frontier_key_first);

    auto edge_partition_frontier_local_degrees =
      edge_partition_e_mask ? edge_partition.compute_local_degrees_with_mask(
                                (*edge_partition_e_mask).value_first(),
                                edge_partition_frontier_major_first,
                                edge_partition_frontier_major_first +
                                  (local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
                                handle.get_stream())
                            : edge_partition.compute_local_degrees(
                                edge_partition_frontier_major_first,
                                edge_partition_frontier_major_first +
                                  (local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
                                handle.get_stream());

    // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
    // to the output array
    thrust::copy(handle.get_thrust_policy(),
                 edge_partition_frontier_local_degrees.begin(),
                 edge_partition_frontier_local_degrees.end(),
                 aggregate_local_frontier_local_degree_offsets.begin() + local_frontier_offsets[i]);
  }
  thrust::exclusive_scan(handle.get_thrust_policy(),
                         aggregate_local_frontier_local_degree_offsets.begin(),
                         aggregate_local_frontier_local_degree_offsets.end(),
                         aggregate_local_frontier_local_degree_offsets.begin());

  // 2. update aggregate_value_buffer

  auto aggregate_value_buffer = allocate_dataframe_buffer<e_op_result_t>(
    aggregate_local_frontier_local_degree_offsets.back_element(handle.get_stream()),
    handle.get_stream());

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    auto edge_partition_frontier_key_first =
      aggregate_local_frontier_key_first + local_frontier_offsets[i];
    auto edge_partition_frontier_major_first =
      thrust_tuple_get_or_identity<KeyIterator, 0>(edge_partition_frontier_key_first);

    rmm::device_uvector<size_t> edge_partition_key_indices(
      local_frontier_offsets[i + 1] - local_frontier_offsets[i], handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     edge_partition_key_indices.begin(),
                     edge_partition_key_indices.end(),
                     size_t{0});

    auto edge_partition_frontier_local_degree_offsets = raft::device_span<size_t const>(
      aggregate_local_frontier_local_degree_offsets.data() + local_frontier_offsets[i],
      (local_frontier_offsets[i + 1] - local_frontier_offsets[i]) + 1);

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

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    if (segment_offsets) {
      auto [edge_partition_key_indices, edge_partition_v_frontier_partition_offsets] =
        partition_v_frontier(
          handle,
          edge_partition_frontier_major_first,
          edge_partition_frontier_major_first +
            (local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
          std::vector<vertex_t>{edge_partition.major_range_first() + (*segment_offsets)[1],
                                edge_partition.major_range_first() + (*segment_offsets)[2],
                                edge_partition.major_range_first() + (*segment_offsets)[3]});

      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one
      // more segment for very high degree vertices and running segmented reduction
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      auto high_size = edge_partition_v_frontier_partition_offsets[1];
      if (high_size > 0) {
        raft::grid_1d_block_t update_grid(high_size,
                                          detail::transform_v_frontier_e_kernel_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        detail::transform_v_frontier_e_high_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[0],
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[1],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            edge_partition_frontier_local_degree_offsets,
            e_op,
            get_dataframe_buffer_begin(aggregate_value_buffer));
      }
      auto mid_size = edge_partition_v_frontier_partition_offsets[2] -
                      edge_partition_v_frontier_partition_offsets[1];
      if (mid_size > 0) {
        raft::grid_1d_warp_t update_grid(mid_size,
                                         detail::transform_v_frontier_e_kernel_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        detail::transform_v_frontier_e_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[1],
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[2],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            edge_partition_frontier_local_degree_offsets,
            e_op,
            get_dataframe_buffer_begin(aggregate_value_buffer));
      }
      auto low_size = edge_partition_v_frontier_partition_offsets[3] -
                      edge_partition_v_frontier_partition_offsets[2];
      if (low_size > 0) {
        raft::grid_1d_thread_t update_grid(low_size,
                                           detail::transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::transform_v_frontier_e_hypersparse_or_low_degree<false, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[2],
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[3],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            edge_partition_frontier_local_degree_offsets,
            e_op,
            get_dataframe_buffer_begin(aggregate_value_buffer));
      }
      auto hypersparse_size = edge_partition_v_frontier_partition_offsets[4] -
                              edge_partition_v_frontier_partition_offsets[3];
      if (hypersparse_size > 0) {
        raft::grid_1d_thread_t update_grid(hypersparse_size,
                                           detail::transform_v_frontier_e_kernel_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::transform_v_frontier_e_hypersparse_or_low_degree<true, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition_frontier_key_first,
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[3],
            edge_partition_key_indices.begin() + edge_partition_v_frontier_partition_offsets[4],
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_e_mask,
            edge_partition_frontier_local_degree_offsets,
            e_op,
            get_dataframe_buffer_begin(aggregate_value_buffer));
      }
    } else {
      raft::grid_1d_thread_t update_grid(
        (local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
        detail::transform_v_frontier_e_kernel_block_size,
        handle.get_device_properties().maxGridSize[0]);

      detail::transform_v_frontier_e_hypersparse_or_low_degree<false, GraphViewType>
        <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
          edge_partition,
          edge_partition_frontier_key_first,
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          edge_partition_e_mask,
          edge_partition_frontier_local_degree_offsets,
          e_op,
          get_dataframe_buffer_begin(aggregate_value_buffer));
    }
  }

  return std::make_tuple(std::move(aggregate_value_buffer),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

}  // namespace detail

}  // namespace cugraph
