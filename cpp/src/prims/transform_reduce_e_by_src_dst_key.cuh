/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <prims/property_op_utils.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_by_src_dst_key_kernel_block_size = 128;

template <bool edge_partition_src_key,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__device__ void update_buffer_element(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu>& edge_partition,
  typename GraphViewType::vertex_type major,
  typename GraphViewType::vertex_type minor,
  typename GraphViewType::edge_type edge_offset,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionSrcDstKeyInputWrapper edge_partition_src_dst_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* key,
  ValueIterator value)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
  auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
  auto src          = GraphViewType::is_storage_transposed ? minor : major;
  auto dst          = GraphViewType::is_storage_transposed ? major : minor;
  auto src_offset   = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
  auto dst_offset   = GraphViewType::is_storage_transposed ? major_offset : minor_offset;

  *key = edge_partition_src_dst_key_input.get(
    ((GraphViewType::is_storage_transposed != edge_partition_src_key) ? major_offset
                                                                      : minor_offset));
  *value = e_op(src,
                dst,
                edge_partition_src_value_input.get(src_offset),
                edge_partition_dst_value_input.get(dst_offset),
                edge_partition_e_value_input.get(edge_offset));
}

template <bool edge_partition_src_key,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ void transform_reduce_by_src_dst_key_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionSrcDstKeyInputWrapper edge_partition_src_dst_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  ValueIterator value_iter)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(*(edge_partition.major_hypersparse_first()) -
                                                edge_partition.major_range_first());
  auto idx                = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(edge_partition.dcs_nzd_vertex_count());

  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(edge_partition.major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_idx));
    auto local_offset = edge_partition.local_offset(major_idx);
    for (edge_t i = 0; i < local_degree; ++i) {
      update_buffer_element<edge_partition_src_key, GraphViewType>(edge_partition,
                                                                   major,
                                                                   indices[i],
                                                                   edge_offset + i,
                                                                   edge_partition_src_value_input,
                                                                   edge_partition_dst_value_input,
                                                                   edge_partition_e_value_input,
                                                                   edge_partition_src_dst_key_input,
                                                                   e_op,
                                                                   keys + local_offset + i,
                                                                   value_iter + local_offset + i);
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <bool edge_partition_src_key,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ void transform_reduce_by_src_dst_key_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionSrcDstKeyInputWrapper edge_partition_src_dst_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  ValueIterator value_iter)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      edge_partition.major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = edge_partition.local_offset(major_offset);
    for (edge_t i = 0; i < local_degree; ++i) {
      update_buffer_element<edge_partition_src_key, GraphViewType>(edge_partition,
                                                                   major,
                                                                   indices[i],
                                                                   edge_offset + i,
                                                                   edge_partition_src_value_input,
                                                                   edge_partition_dst_value_input,
                                                                   edge_partition_e_value_input,
                                                                   edge_partition_src_dst_key_input,
                                                                   e_op,
                                                                   keys + local_offset + i,
                                                                   value_iter + local_offset + i);
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <bool edge_partition_src_key,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ void transform_reduce_by_src_dst_key_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionSrcDstKeyInputWrapper edge_partition_src_dst_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  ValueIterator value_iter)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_reduce_e_by_src_dst_key_kernel_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      edge_partition.major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = edge_partition.local_offset(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      update_buffer_element<edge_partition_src_key, GraphViewType>(edge_partition,
                                                                   major,
                                                                   indices[i],
                                                                   edge_offset + i,
                                                                   edge_partition_src_value_input,
                                                                   edge_partition_dst_value_input,
                                                                   edge_partition_e_value_input,
                                                                   edge_partition_src_dst_key_input,
                                                                   e_op,
                                                                   keys + local_offset + i,
                                                                   value_iter + local_offset + i);
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool edge_partition_src_key,
          typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgePartitionSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ValueIterator>
__global__ void transform_reduce_by_src_dst_key_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input,
  EdgePartitionSrcDstKeyInputWrapper edge_partition_src_dst_key_input,
  EdgeOp e_op,
  typename GraphViewType::vertex_type* keys,
  ValueIterator value_iter)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  auto idx = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      edge_partition.major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = edge_partition.local_offset(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      update_buffer_element<edge_partition_src_key, GraphViewType>(edge_partition,
                                                                   major,
                                                                   indices[i],
                                                                   edge_offset + i,
                                                                   edge_partition_src_value_input,
                                                                   edge_partition_dst_value_input,
                                                                   edge_partition_e_value_input,
                                                                   edge_partition_src_dst_key_input,
                                                                   e_op,
                                                                   keys + local_offset + i,
                                                                   value_iter + local_offset + i);
    }

    idx += gridDim.x;
  }
}

// FIXME: better derive value_t from BufferType
template <typename vertex_t, typename value_t, typename BufferType, typename ReduceOp>
std::tuple<rmm::device_uvector<vertex_t>, BufferType> reduce_to_unique_kv_pairs(
  rmm::device_uvector<vertex_t>&& keys,
  BufferType&& value_buffer,
  ReduceOp reduce_op,
  cudaStream_t stream)
{
  thrust::sort_by_key(
    rmm::exec_policy(stream), keys.begin(), keys.end(), get_dataframe_buffer_begin(value_buffer));
  auto num_uniques =
    thrust::count_if(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(keys.size()),
                     [keys = keys.data()] __device__(auto i) {
                       return ((i == 0) || (keys[i] != keys[i - 1])) ? true : false;
                     });

  rmm::device_uvector<vertex_t> unique_keys(num_uniques, stream);
  auto value_for_unique_key_buffer = allocate_dataframe_buffer<value_t>(unique_keys.size(), stream);
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys.begin(),
                        keys.end(),
                        get_dataframe_buffer_begin(value_buffer),
                        unique_keys.begin(),
                        get_dataframe_buffer_begin(value_for_unique_key_buffer),
                        thrust::equal_to<vertex_t>{},
                        reduce_op);

  return std::make_tuple(std::move(unique_keys), std::move(value_for_unique_key_buffer));
}

template <bool edge_src_key,
          typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeSrcDstKeyInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           decltype(allocate_dataframe_buffer<T>(0, cudaStream_t{nullptr}))>
transform_reduce_e_by_src_dst_key(raft::handle_t const& handle,
                                  GraphViewType const& graph_view,
                                  EdgeSrcValueInputWrapper edge_src_value_input,
                                  EdgeDstValueInputWrapper edge_dst_value_input,
                                  EdgeValueInputWrapper edge_value_input,
                                  EdgeSrcDstKeyInputWrapper edge_src_dst_key_input,
                                  EdgeOp e_op,
                                  T init,
                                  ReduceOp reduce_op)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename EdgeSrcDstKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcValueInputWrapper::value_iterator>>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    std::conditional_t<GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeDstValueInputWrapper::value_iterator>>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator>>;
  using edge_partition_src_dst_key_device_view_t =
    std::conditional_t<edge_src_key != GraphViewType::is_storage_transposed,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcDstKeyInputWrapper::value_iterator>,
                       detail::edge_partition_endpoint_property_device_view_t<
                         vertex_t,
                         typename EdgeSrcDstKeyInputWrapper::value_iterator>>;

  rmm::device_uvector<vertex_t> keys(0, handle.get_stream());
  auto value_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    int comm_root_rank = 0;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      comm_root_rank           = i * row_comm_size + row_comm_rank;
    }

    auto num_edges = edge_partition.number_of_edges();

    rmm::device_uvector<vertex_t> tmp_keys(num_edges, handle.get_stream());
    auto tmp_value_buffer = allocate_dataframe_buffer<T>(tmp_keys.size(), handle.get_stream());

    if (graph_view.vertex_partition_range_size(comm_root_rank) > 0) {
      edge_partition_src_input_device_view_t edge_partition_src_value_input{};
      edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
      if constexpr (GraphViewType::is_storage_transposed) {
        edge_partition_src_value_input =
          edge_partition_src_input_device_view_t(edge_src_value_input);
        edge_partition_dst_value_input =
          edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
      } else {
        edge_partition_src_value_input =
          edge_partition_src_input_device_view_t(edge_src_value_input, i);
        edge_partition_dst_value_input =
          edge_partition_dst_input_device_view_t(edge_dst_value_input);
      }
      auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);

      edge_partition_src_dst_key_device_view_t edge_partition_src_dst_key_input{};
      if constexpr (edge_src_key != GraphViewType::is_storage_transposed) {
        edge_partition_src_dst_key_input =
          edge_partition_src_dst_key_device_view_t(edge_src_dst_key_input, i);
      } else {
        edge_partition_src_dst_key_input =
          edge_partition_src_dst_key_device_view_t(edge_src_dst_key_input);
      }

      auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
      if (segment_offsets) {
        // FIXME: we may further improve performance by 1) concurrently running kernels on different
        // segments; 2) individually tuning block sizes for different segments; and 3) adding one
        // more segment for very high degree vertices and running segmented reduction
        static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
        if ((*segment_offsets)[1] > 0) {
          raft::grid_1d_block_t update_grid(
            (*segment_offsets)[1],
            detail::transform_reduce_e_by_src_dst_key_kernel_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::transform_reduce_by_src_dst_key_high_degree<edge_src_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition.major_range_first(),
              edge_partition.major_range_first() + (*segment_offsets)[1],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_src_dst_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
          raft::grid_1d_warp_t update_grid(
            (*segment_offsets)[2] - (*segment_offsets)[1],
            detail::transform_reduce_e_by_src_dst_key_kernel_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::transform_reduce_by_src_dst_key_mid_degree<edge_src_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition.major_range_first() + (*segment_offsets)[1],
              edge_partition.major_range_first() + (*segment_offsets)[2],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_src_dst_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
          raft::grid_1d_thread_t update_grid(
            (*segment_offsets)[3] - (*segment_offsets)[2],
            detail::transform_reduce_e_by_src_dst_key_kernel_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::transform_reduce_by_src_dst_key_low_degree<edge_src_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition.major_range_first() + (*segment_offsets)[2],
              edge_partition.major_range_first() + (*segment_offsets)[3],
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_src_dst_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
        if (edge_partition.dcs_nzd_vertex_count() &&
            (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
          raft::grid_1d_thread_t update_grid(
            *(edge_partition.dcs_nzd_vertex_count()),
            detail::transform_reduce_e_by_src_dst_key_kernel_block_size,
            handle.get_device_properties().maxGridSize[0]);
          detail::transform_reduce_by_src_dst_key_hypersparse<edge_src_key, GraphViewType>
            <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
              edge_partition,
              edge_partition_src_value_input,
              edge_partition_dst_value_input,
              edge_partition_e_value_input,
              edge_partition_src_dst_key_input,
              e_op,
              tmp_keys.data(),
              get_dataframe_buffer_begin(tmp_value_buffer));
        }
      } else {
        raft::grid_1d_thread_t update_grid(
          edge_partition.major_range_size(),
          detail::transform_reduce_e_by_src_dst_key_kernel_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::transform_reduce_by_src_dst_key_low_degree<edge_src_key, GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_last(),
            edge_partition_src_value_input,
            edge_partition_dst_value_input,
            edge_partition_e_value_input,
            edge_partition_src_dst_key_input,
            e_op,
            tmp_keys.data(),
            get_dataframe_buffer_begin(tmp_value_buffer));
      }
    }
    std::tie(tmp_keys, tmp_value_buffer) = reduce_to_unique_kv_pairs<vertex_t, T>(
      std::move(tmp_keys), std::move(tmp_value_buffer), reduce_op, handle.get_stream());

    if (GraphViewType::is_multi_gpu) {
      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      rmm::device_uvector<vertex_t> rx_unique_keys(0, handle.get_stream());
      auto rx_value_for_unique_key_buffer = allocate_dataframe_buffer<T>(0, handle.get_stream());
      std::tie(rx_unique_keys, rx_value_for_unique_key_buffer, std::ignore) =
        groupby_gpu_id_and_shuffle_kv_pairs(
          comm,
          tmp_keys.begin(),
          tmp_keys.end(),
          get_dataframe_buffer_begin(tmp_value_buffer),
          [key_func = detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{comm_size}] __device__(
            auto val) { return key_func(val); },
          handle.get_stream());

      std::tie(tmp_keys, tmp_value_buffer) =
        reduce_to_unique_kv_pairs<vertex_t, T>(std::move(rx_unique_keys),
                                               std::move(rx_value_for_unique_key_buffer),
                                               reduce_op,
                                               handle.get_stream());
    }

    auto cur_size = keys.size();
    if (cur_size == 0) {
      keys         = std::move(tmp_keys);
      value_buffer = std::move(tmp_value_buffer);
    } else {
      // FIXME: this can lead to frequent costly reallocation; we may be able to avoid this if we
      // can reserve address space to avoid expensive reallocation.
      // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management
      keys.resize(cur_size + tmp_keys.size(), handle.get_stream());
      resize_dataframe_buffer(value_buffer, keys.size(), handle.get_stream());

      auto execution_policy = handle.get_thrust_policy();
      thrust::copy(execution_policy, tmp_keys.begin(), tmp_keys.end(), keys.begin() + cur_size);
      thrust::copy(execution_policy,
                   get_dataframe_buffer_begin(tmp_value_buffer),
                   get_dataframe_buffer_begin(tmp_value_buffer) + tmp_keys.size(),
                   get_dataframe_buffer_begin(value_buffer) + cur_size);
    }
  }

  if (GraphViewType::is_multi_gpu) {
    std::tie(keys, value_buffer) = reduce_to_unique_kv_pairs<vertex_t, T>(
      std::move(keys), std::move(value_buffer), reduce_op, handle.get_stream());
  }

  // FIXME: add init

  return std::make_tuple(std::move(keys), std::move(value_buffer));
}

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs to (key, value) pairs.
 *
 * This function is inspired by thrust::transform_reduce() and thrust::reduce_by_key(). Keys for
 * edges are determined by the edge sources.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeSrcKeyInputWrapper Type of the wrapper for edge source key values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
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
 * @param edge_src_key_input Wrapper used to access source input ke values (for the edge sources
 * assigned to this process in multi-GPU). Use  cugraph::edge_src_property_t::view(). Use
 * update_edge_src_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced to (source key, value) pairs.
 * @param init Initial value to be added to the value in each transform-reduced (source key, value)
 * pair.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is recommended to use
 * the pre-defined reduction operators whenever possible as the current (and future) implementations
 * of graph primitives may check whether @p ReduceOp is a known type (or has known member variables)
 * to take a more optimized code path. See the documentation in the reduce_op.cuh file for
 * instructions on writing custom reduction operators.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeSrcKeyInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
auto transform_reduce_e_by_src_key(raft::handle_t const& handle,
                                   GraphViewType const& graph_view,
                                   EdgeSrcValueInputWrapper edge_src_value_input,
                                   EdgeDstValueInputWrapper edge_dst_value_input,
                                   EdgeValueInputWrapper edge_value_input,
                                   EdgeSrcKeyInputWrapper edge_src_key_input,
                                   EdgeOp e_op,
                                   T init,
                                   ReduceOp reduce_op,
                                   bool do_expensive_check = false)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename EdgeSrcKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);
  static_assert(ReduceOp::pure_function, "ReduceOp should be a pure function.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return detail::transform_reduce_e_by_src_dst_key<true>(handle,
                                                         graph_view,
                                                         edge_src_value_input,
                                                         edge_dst_value_input,
                                                         edge_value_input,
                                                         edge_src_key_input,
                                                         e_op,
                                                         init,
                                                         reduce_op);
}

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs to (key, value) pairs.
 *
 * This function is inspired by thrust::transform_reduce() and thrust::reduce_by_key(). Keys for
 * edges are determined by the edge destinations.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeDstKeyInputWrapper Type of the wrapper for edge destination key values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the values in (key, value) pairs.
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
 * @param edge_partition_dst_key_input Wrapper used to access destination input key values (for the
 * edge destinations assigned to this process in multi-GPU). Use
 * cugraph::edge_dst_property_t::view(). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be reduced to (destination key, value) pairs.
 * @param init Initial value to be added to the value in each transform-reduced (destination key,
 * value) pair.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in src/prims/reduce_op.cuh. It is recommended to use
 * the pre-defined reduction operators whenever possible as the current (and future) implementations
 * of graph primitives may check whether @p ReduceOp is a known type (or has known member variables)
 * to take a more optimized code path. See the documentation in the reduce_op.cuh file for
 * instructions on writing custom reduction operators.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of rmm::device_uvector<typename GraphView::vertex_type> and
 * rmm::device_uvector<T> (if T is arithmetic scalar) or a tuple of rmm::device_uvector objects (if
 * T is a thrust::tuple type of arithmetic scalar types, one rmm::device_uvector object per scalar
 * type).
 */
template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeDstKeyInputWrapper,
          typename EdgeOp,
          typename ReduceOp,
          typename T>
auto transform_reduce_e_by_dst_key(raft::handle_t const& handle,
                                   GraphViewType const& graph_view,
                                   EdgeSrcValueInputWrapper edge_src_value_input,
                                   EdgeDstValueInputWrapper edge_dst_value_input,
                                   EdgeValueInputWrapper edge_value_input,
                                   EdgeDstKeyInputWrapper edge_dst_key_input,
                                   EdgeOp e_op,
                                   T init,
                                   ReduceOp reduce_op,
                                   bool do_expensive_check = false)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(std::is_same<typename EdgeDstKeyInputWrapper::value_type,
                             typename GraphViewType::vertex_type>::value);
  static_assert(ReduceOp::pure_function, "ReduceOp should be a pure function.");

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return detail::transform_reduce_e_by_src_dst_key<false>(handle,
                                                          graph_view,
                                                          edge_src_value_input,
                                                          edge_dst_value_input,
                                                          edge_value_input,
                                                          edge_dst_key_input,
                                                          e_op,
                                                          init,
                                                          reduce_op);
}

}  // namespace cugraph
