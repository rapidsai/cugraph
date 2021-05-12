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

#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/matrix_partition_device.cuh>
#include <cugraph/patterns/edge_op_utils.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/tuple.h>

#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_for_all_block_size = 128;

template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
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

  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto sum                                    = thrust::transform_reduce(
      thrust::seq,
      thrust::make_counting_iterator(edge_t{0}),
      thrust::make_counting_iterator(local_degree),
      [&matrix_partition,
       &adj_matrix_row_value_input_first,
       &adj_matrix_col_value_input_first,
       &e_op,
       idx,
       indices,
       weights] __device__(auto i) {
        auto minor        = indices[i];
        auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
        auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
        auto row          = GraphViewType::is_adj_matrix_transposed
                     ? minor
                     : matrix_partition.get_major_from_major_offset_nocheck(idx);
        auto col = GraphViewType::is_adj_matrix_transposed
                     ? matrix_partition.get_major_from_major_offset_nocheck(idx)
                     : minor;
        auto row_offset =
          GraphViewType::is_adj_matrix_transposed ? minor_offset : static_cast<vertex_t>(idx);
        auto col_offset =
          GraphViewType::is_adj_matrix_transposed ? static_cast<vertex_t>(idx) : minor_offset;
        return evaluate_edge_op<GraphViewType,
                                AdjMatrixRowValueInputIterator,
                                AdjMatrixColValueInputIterator,
                                EdgeOp>()
          .compute(row,
                   col,
                   weight,
                   *(adj_matrix_row_value_input_first + row_offset),
                   *(adj_matrix_col_value_input_first + col_offset),
                   e_op);
      },
      e_op_result_t{},
      [] __device__(auto lhs, auto rhs) { return plus_edge_op_result(lhs, rhs); });

    e_op_result_sum = plus_edge_op_result(e_op_result_sum, sum);
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum =
    block_reduce_edge_op_result<e_op_result_t, transform_reduce_e_for_all_block_size>().compute(
      e_op_result_sum);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_mid_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
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

  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      auto minor        = indices[i];
      auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                   ? minor
                   : matrix_partition.get_major_from_major_offset_nocheck(idx);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(idx)
                   : minor;
      auto row_offset =
        GraphViewType::is_adj_matrix_transposed ? minor_offset : static_cast<vertex_t>(idx);
      auto col_offset =
        GraphViewType::is_adj_matrix_transposed ? static_cast<vertex_t>(idx) : minor_offset;
      auto e_op_result = evaluate_edge_op<GraphViewType,
                                          AdjMatrixRowValueInputIterator,
                                          AdjMatrixColValueInputIterator,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    *(adj_matrix_row_value_input_first + row_offset),
                                    *(adj_matrix_col_value_input_first + col_offset),
                                    e_op);
      e_op_result_sum = plus_edge_op_result<e_op_result_t>(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }

  e_op_result_sum =
    block_reduce_edge_op_result<e_op_result_t, transform_reduce_e_for_all_block_size>().compute(
      e_op_result_sum);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_high_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultIterator result_iter /* size 1 */,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultIterator>::value_type;

  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  size_t idx              = static_cast<size_t>(blockIdx.x);

  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(major_last - major_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                   ? minor
                   : matrix_partition.get_major_from_major_offset_nocheck(idx);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(idx)
                   : minor;
      auto row_offset =
        GraphViewType::is_adj_matrix_transposed ? minor_offset : static_cast<vertex_t>(idx);
      auto col_offset =
        GraphViewType::is_adj_matrix_transposed ? static_cast<vertex_t>(idx) : minor_offset;
      auto e_op_result = evaluate_edge_op<GraphViewType,
                                          AdjMatrixRowValueInputIterator,
                                          AdjMatrixColValueInputIterator,
                                          EdgeOp>()
                           .compute(row,
                                    col,
                                    weight,
                                    *(adj_matrix_row_value_input_first + row_offset),
                                    *(adj_matrix_col_value_input_first + col_offset),
                                    e_op);
      e_op_result_sum = plus_edge_op_result(e_op_result_sum, e_op_result);
    }
    idx += gridDim.x;
  }

  e_op_result_sum =
    block_reduce_edge_op_result<e_op_result_t, transform_reduce_e_for_all_block_size>().compute(
      e_op_result_sum);
  if (threadIdx.x == 0) { atomic_accumulate_edge_op_result(result_iter, e_op_result_sum); }
}

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), *(@p adj_matrix_row_value_input_first + i), and *(@p adj_matrix_col_value_input_first +
 * j) (where i is in [0, graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p edge_op outputs.
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T>
T transform_reduce_e(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                     AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                     EdgeOp e_op,
                     T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t = typename GraphViewType::vertex_type;

  auto result_buffer = allocate_dataframe_buffer<T>(1, handle.get_stream());
  thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               get_dataframe_buffer_begin<T>(result_buffer),
               get_dataframe_buffer_begin<T>(result_buffer) + 1,
               T{});

  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                    ? vertex_t{0}
                                    : matrix_partition.get_major_value_start_offset();
    auto col_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                    ? matrix_partition.get_major_value_start_offset()
                                    : vertex_t{0};
    auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
    if (segment_offsets.size() > 0) {
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      static_assert(detail::num_segments_per_vertex_partition == 3);
      if (segment_offsets[1] > 0) {
        raft::grid_1d_block_t update_grid(segment_offsets[1],
                                          detail::transform_reduce_e_for_all_block_size,
                                          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_high_degree<<<update_grid.num_blocks,
                                                        update_grid.block_size,
                                                        0,
                                                        handle.get_stream()>>>(
          matrix_partition,
          matrix_partition.get_major_first(),
          matrix_partition.get_major_first() + segment_offsets[1],
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first + col_value_input_offset,
          get_dataframe_buffer_begin<T>(result_buffer),
          e_op);
      }
      if (segment_offsets[2] - segment_offsets[1] > 0) {
        raft::grid_1d_warp_t update_grid(segment_offsets[2] - segment_offsets[1],
                                         detail::transform_reduce_e_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_mid_degree<<<update_grid.num_blocks,
                                                       update_grid.block_size,
                                                       0,
                                                       handle.get_stream()>>>(
          matrix_partition,
          matrix_partition.get_major_first() + segment_offsets[1],
          matrix_partition.get_major_first() + segment_offsets[2],
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first + col_value_input_offset,
          get_dataframe_buffer_begin<T>(result_buffer),
          e_op);
      }
      if (segment_offsets[3] - segment_offsets[2] > 0) {
        raft::grid_1d_thread_t update_grid(segment_offsets[3] - segment_offsets[2],
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_low_degree<<<update_grid.num_blocks,
                                                       update_grid.block_size,
                                                       0,
                                                       handle.get_stream()>>>(
          matrix_partition,
          matrix_partition.get_major_first() + segment_offsets[2],
          matrix_partition.get_major_last(),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first + col_value_input_offset,
          get_dataframe_buffer_begin<T>(result_buffer),
          e_op);
      }
    } else {
      if (matrix_partition.get_major_size() > 0) {
        raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                           detail::transform_reduce_e_for_all_block_size,
                                           handle.get_device_properties().maxGridSize[0]);

        detail::for_all_major_for_all_nbr_low_degree<<<update_grid.num_blocks,
                                                       update_grid.block_size,
                                                       0,
                                                       handle.get_stream()>>>(
          matrix_partition,
          matrix_partition.get_major_first(),
          matrix_partition.get_major_last(),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first + col_value_input_offset,
          get_dataframe_buffer_begin<T>(result_buffer),
          e_op);
      }
    }
  }

  auto result =
    thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   get_dataframe_buffer_begin<T>(result_buffer),
                   get_dataframe_buffer_begin<T>(result_buffer) + 1,
                   T{},
                   [] __device__(T lhs, T rhs) { return plus_edge_op_result(lhs, rhs); });

  if (GraphViewType::is_multi_gpu) {
    result = host_scalar_allreduce(handle.get_comms(), result, handle.get_stream());
  }

  return plus_edge_op_result(init, result);
}

}  // namespace experimental
}  // namespace cugraph
