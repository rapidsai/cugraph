/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cugraph/graph_view.hpp>
#include <cugraph/matrix_partition_view.hpp>
#include <cugraph/prims/property_op_utils.cuh>
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
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_hypersparse(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_hypersparse_first,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset =
    static_cast<size_t>(major_hypersparse_first - matrix_partition.get_major_first());
  size_t idx = static_cast<size_t>(tid);

  auto dcs_nzd_vertex_count = *(matrix_partition.get_dcs_nzd_vertex_count());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(dcs_nzd_vertex_count)) {
    auto major =
      *(matrix_partition.get_major_from_major_hypersparse_idx_nocheck(static_cast<vertex_t>(idx)));
    auto major_idx =
      major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_idx);
    auto sum                                    = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&matrix_partition,
       &adj_matrix_row_value_input,
       &adj_matrix_col_value_input,
       &e_op,
       major,
       indices,
       weights] __device__(auto i) {
        auto major_offset = matrix_partition.get_major_offset_from_major_nocheck(major);
        auto minor        = indices[i];
        auto weight       = weights ? (*weights)[i] : weight_t{1.0};
        auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
        auto row          = GraphViewType::is_adj_matrix_transposed ? minor : major;
        auto col          = GraphViewType::is_adj_matrix_transposed ? major : minor;
        auto row_offset   = GraphViewType::is_adj_matrix_transposed
                                                                 ? minor_offset
                                                                 : static_cast<vertex_t>(major_offset);
        auto col_offset   = GraphViewType::is_adj_matrix_transposed
                                                                 ? static_cast<vertex_t>(major_offset)
                                                                 : minor_offset;
        return evaluate_edge_op<GraphViewType,
                                vertex_t,
                                AdjMatrixRowValueInputWrapper,
                                AdjMatrixColValueInputWrapper,
                                EdgeOp>()
          .compute(row,
                   col,
                   weight,
                   adj_matrix_row_value_input.get(row_offset),
                   adj_matrix_col_value_input.get(col_offset),
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
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  size_t idx              = static_cast<size_t>(tid);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto sum                                    = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&matrix_partition,
       &adj_matrix_row_value_input,
       &adj_matrix_col_value_input,
       &e_op,
       major_offset,
       indices,
       weights] __device__(auto i) {
        auto minor        = indices[i];
        auto weight       = weights ? (*weights)[i] : weight_t{1.0};
        auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
        auto row          = GraphViewType::is_adj_matrix_transposed
                                                                 ? minor
                                                                 : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
        auto col          = GraphViewType::is_adj_matrix_transposed
                                                                 ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                                                                 : minor;
        auto row_offset   = GraphViewType::is_adj_matrix_transposed
                                                                 ? minor_offset
                                                                 : static_cast<vertex_t>(major_offset);
        auto col_offset   = GraphViewType::is_adj_matrix_transposed
                                                                 ? static_cast<vertex_t>(major_offset)
                                                                 : minor_offset;
        return evaluate_edge_op<GraphViewType,
                                vertex_t,
                                AdjMatrixRowValueInputWrapper,
                                AdjMatrixColValueInputWrapper,
                                EdgeOp>()
          .compute(row,
                   col,
                   weight,
                   adj_matrix_row_value_input.get(row_offset),
                   adj_matrix_col_value_input.get(col_offset),
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
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_mid_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(transform_reduce_e_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id      = tid % raft::warp_size();
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  size_t idx              = static_cast<size_t>(tid / raft::warp_size());

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                            ? minor
                            : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col          = GraphViewType::is_adj_matrix_transposed
                            ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      auto e_op_result  = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          AdjMatrixRowValueInputWrapper,
                                          AdjMatrixColValueInputWrapper,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    adj_matrix_row_value_input.get(row_offset),
                                    adj_matrix_col_value_input.get(col_offset),
                                    e_op);
      e_op_result_sum = edge_property_add(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }

  e_op_result_sum = BlockReduce(temp_storage).Reduce(e_op_result_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_high_degree(
  matrix_partition_device_view_t<typename GraphViewType::vertex_type,
                                 typename GraphViewType::edge_type,
                                 typename GraphViewType::weight_type,
                                 GraphViewType::is_multi_gpu> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
  AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  size_t idx              = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<e_op_result_t, transform_reduce_e_for_all_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<e_op_result_t, thrust::plus> edge_property_add{};
  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto weight       = weights ? (*weights)[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                            ? minor
                            : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col          = GraphViewType::is_adj_matrix_transposed
                            ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                            : minor;
      auto row_offset   = GraphViewType::is_adj_matrix_transposed
                            ? minor_offset
                            : static_cast<vertex_t>(major_offset);
      auto col_offset   = GraphViewType::is_adj_matrix_transposed
                            ? static_cast<vertex_t>(major_offset)
                            : minor_offset;
      auto e_op_result  = evaluate_edge_op<GraphViewType,
                                          vertex_t,
                                          AdjMatrixRowValueInputWrapper,
                                          AdjMatrixColValueInputWrapper,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    adj_matrix_row_value_input.get(row_offset),
                                    adj_matrix_col_value_input.get(col_offset),
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
 * @tparam AdjMatrixRowValueInputWrapper Type of the wrapper for graph adjacency matrix row input
 * properties.
 * @tparam AdjMatrixColValueInputWrapper Type of the wrapper for graph adjacency matrix column input
 * properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input Device-copyable wrapper used to access row input properties
 * (for the rows assigned to this process in multi-GPU). Use either
 * cugraph::row_properties_t::device_view() (if @p e_op needs to access row properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access row properties). Use
 * copy_to_adj_matrix_row to fill the wrapper.
 * @param adj_matrix_col_value_input Device-copyable wrapper used to access column input properties
 * (for the columns assigned to this process in multi-GPU). Use either
 * cugraph::col_properties_t::device_view() (if @p e_op needs to access column properties) or
 * cugraph::dummy_properties_t::device_view() (if @p e_op does not access column properties). Use
 * copy_to_adj_matrix_col to fill the wrapper.
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), properties for the row (i.e. source), and properties for the column  (i.e. destination)
 * and returns a value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p edge_op outputs.
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputWrapper,
          typename AdjMatrixColValueInputWrapper,
          typename EdgeOp,
          typename T>
T transform_reduce_e(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     AdjMatrixRowValueInputWrapper adj_matrix_row_value_input,
                     AdjMatrixColValueInputWrapper adj_matrix_col_value_input,
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

  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    auto matrix_partition =
      matrix_partition_device_view_t<vertex_t, edge_t, weight_t, GraphViewType::is_multi_gpu>(
        graph_view.get_matrix_partition_view(i));

    auto matrix_partition_row_value_input = adj_matrix_row_value_input;
    auto matrix_partition_col_value_input = adj_matrix_col_value_input;
    if constexpr (GraphViewType::is_adj_matrix_transposed) {
      matrix_partition_col_value_input.set_local_adj_matrix_partition_idx(i);
    } else {
      matrix_partition_row_value_input.set_local_adj_matrix_partition_idx(i);
    }

    auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
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
            matrix_partition,
            matrix_partition.get_major_first(),
            matrix_partition.get_major_first() + (*segment_offsets)[1],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
        raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                         detail::transform_reduce_e_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_mid_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[1],
            matrix_partition.get_major_first() + (*segment_offsets)[2],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
        raft::grid_1d_thread_t update_grid((*segment_offsets)[3] - (*segment_offsets)[2],
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[2],
            matrix_partition.get_major_first() + (*segment_offsets)[3],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
      if (matrix_partition.get_dcs_nzd_vertex_count() &&
          (*(matrix_partition.get_dcs_nzd_vertex_count()) > 0)) {
        raft::grid_1d_thread_t update_grid(*(matrix_partition.get_dcs_nzd_vertex_count()),
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);
        detail::for_all_major_for_all_nbr_hypersparse<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first() + (*segment_offsets)[3],
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
            get_dataframe_buffer_begin(result_buffer),
            e_op);
      }
    } else {
      if (matrix_partition.get_major_size() > 0) {
        raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_low_degree<GraphViewType>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            matrix_partition.get_major_first(),
            matrix_partition.get_major_last(),
            matrix_partition_row_value_input,
            matrix_partition_col_value_input,
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
