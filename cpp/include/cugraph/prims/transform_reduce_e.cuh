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
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/tuple.h>

#include <cstdint>
#include <type_traits>

namespace cugraph {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_for_all_block_size = 128;

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_hypersparse(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_hypersparse_first,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_hypersparse_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(edge_partition.dcs_nzd_vertex_count());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(edge_partition.major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_idx);
    auto sum                                    = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&edge_partition,
       &edge_partition_src_value_input,
       &edge_partition_dst_value_input,
       &e_op,
       major,
       indices,
       weights] __device__(auto i) {
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        auto minor        = indices[i];
        auto weight       = weights ? (*weights)[i] : weight_t{1.0};
        auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
        auto src          = GraphViewType::is_storage_transposed ? minor : major;
        auto dst          = GraphViewType::is_storage_transposed ? major : minor;
        auto src_offset =
          GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
        auto dst_offset =
          GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
        return evaluate_edge_op<GraphViewType,
                                vertex_t,
                                EdgePartitionSrcValueInputWrapper,
                                EdgePartitionDstValueInputWrapper,
                                EdgeOp>()
          .compute(src,
                   dst,
                   weight,
                   edge_partition_src_value_input.get(src_offset),
                   edge_partition_dst_value_input.get(dst_offset),
                   e_op);
      },
      e_op_result_t{},
      edge_property_add);

    e_op_result_sum = edge_property_add(e_op_result_sum, sum);
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
    auto sum                                    = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&edge_partition,
       &edge_partition_src_value_input,
       &edge_partition_dst_value_input,
       &e_op,
       major_offset,
       indices,
       weights] __device__(auto i) {
        auto minor        = indices[i];
        auto weight       = weights ? (*weights)[i] : weight_t{1.0};
        auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
        auto src          = GraphViewType::is_storage_transposed
                                                                 ? minor
                                                                 : edge_partition.major_from_major_offset_nocheck(major_offset);
        auto dst          = GraphViewType::is_storage_transposed
                                                                 ? edge_partition.major_from_major_offset_nocheck(major_offset)
                                                                 : minor;
        auto src_offset =
          GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
        auto dst_offset =
          GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
        return evaluate_edge_op<GraphViewType,
                                vertex_t,
                                EdgePartitionSrcValueInputWrapper,
                                EdgePartitionDstValueInputWrapper,
                                EdgeOp>()
          .compute(src,
                   dst,
                   weight,
                   edge_partition_src_value_input.get(src_offset),
                   edge_partition_dst_value_input.get(dst_offset),
                   e_op);
      },
      e_op_result_t{},
      edge_property_add);

    e_op_result_sum = edge_property_add(e_op_result_sum, sum);
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_mid_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_reduce_e_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid / raft::warp_size());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed
                            ? minor
                            : edge_partition.major_from_major_offset_nocheck(major_offset);
      auto dst          = GraphViewType::is_storage_transposed
                            ? edge_partition.major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto src_offset =
        GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
      auto dst_offset =
        GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
      auto e_op_result = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          EdgePartitionSrcValueInputWrapper,
                                          EdgePartitionDstValueInputWrapper,
                                          EdgeOp>()
                           .compute(src,
                                    dst,
                                    weight,
                                    edge_partition_src_value_input.get(src_offset),
                                    edge_partition_dst_value_input.get(dst_offset),
                                    e_op);
      e_op_result_sum = edge_property_add(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_high_degree(
  edge_partition_device_view_t<typename GraphViewType::vertex_type,
                               typename GraphViewType::edge_type,
                               typename GraphViewType::weight_type,
                               GraphViewType::is_multi_gpu> edge_partition,
  typename GraphViewType::vertex_type major_range_first,
  typename GraphViewType::vertex_type major_range_last,
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);
      auto src          = GraphViewType::is_storage_transposed
                            ? minor
                            : edge_partition.major_from_major_offset_nocheck(major_offset);
      auto dst          = GraphViewType::is_storage_transposed
                            ? edge_partition.major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto src_offset =
        GraphViewType::is_storage_transposed ? minor_offset : static_cast<vertex_t>(major_offset);
      auto dst_offset =
        GraphViewType::is_storage_transposed ? static_cast<vertex_t>(major_offset) : minor_offset;
      auto e_op_result = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          EdgePartitionSrcValueInputWrapper,
                                          EdgePartitionDstValueInputWrapper,
                                          EdgeOp>()
                           .compute(src,
                                    dst,
                                    weight,
                                    edge_partition_src_value_input.get(src_offset),
                                    edge_partition_dst_value_input.get(dst_offset),
                                    e_op);
      e_op_result_sum = edge_property_add(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x;
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam EdgePartitionSrcValueInputWrapper Type of the wrapper for edge partition source property
 * values.
 * @tparam EdgePartitionDstValueInputWrapper Type of the wrapper for edge partition destination
 * property values.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
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
 * value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p edge_op outputs.
 */
template <typename GraphViewType,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp,
          typename T>
T transform_reduce_e(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     EdgePartitionSrcValueInputWrapper edge_partition_src_value_input,
                     EdgePartitionDstValueInputWrapper edge_partition_dst_value_input,
                     EdgeOp e_op,
                     T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  property_op<T, thrust::plus> edge_property_add{};

  auto result_buffer = allocate_dataframe_buffer<T>(1, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               get_dataframe_buffer_begin(result_buffer),
               get_dataframe_buffer_begin(result_buffer) + 1,
               T{});

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_src_value_input_copy = edge_partition_src_value_input;
    auto edge_partition_dst_value_input_copy = edge_partition_dst_value_input;
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_dst_value_input_copy.set_local_edge_partition_idx(i);
    } else {
      edge_partition_src_value_input_copy.set_local_edge_partition_idx(i);
    }

    auto segment_offsets = graph_view.local_edge_partition_segment_offsets(i);
    if (segment_offsets) {
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
      if ((*segment_offsets)[1] > 0) {
        raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                          detail::transform_reduce_e_for_all_block_size,
                                          handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_high_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
        raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                         detail::transform_reduce_e_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[1],
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
        raft::grid_1d_thread_t update_grid((*segment_offsets)[3] - (*segment_offsets)[2],
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[2],
            edge_partition.major_range_first() + (*segment_offsets)[3],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
        raft::grid_1d_thread_t update_grid(*(edge_partition.dcs_nzd_vertex_count()),
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first() + (*segment_offsets)[3],
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
    } else {
      if (edge_partition.major_range_size() > 0) {
        raft::grid_1d_thread_t update_grid(edge_partition.major_range_size(),
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            edge_partition,
            edge_partition.major_range_first(),
            edge_partition.major_range_last(),
            edge_partition_src_value_input_copy,
            edge_partition_dst_value_input_copy,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
    }
  }

  auto result = thrust::reduce(
    handle.get_thrust_policy(),
    get_dataframe_buffer_begin(result_buffer),
    get_dataframe_buffer_begin(result_buffer) + 1,
    ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() != 0)) ? T{} : init,
    edge_property_add);

  if constexpr (GraphViewType::is_multi_gpu) {
    result = host_scalar_allreduce(
      handle.get_comms(), result, raft::comms::op_t::SUM, handle.get_stream());
  }

  return result;
}

}  // namespace cugraph
