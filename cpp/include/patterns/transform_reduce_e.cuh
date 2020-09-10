/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <graph_device_view.cuh>
#include <patterns/edge_op_utils.cuh>
#include <utilities/cuda.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_for_all_low_out_degree_block_size = 128;

template <typename EdgeOpResultType, size_t BlockSize>
struct block_reduce_edge_op_result {
  template <typename T = EdgeOpResultType>
  __device__ std::enable_if_t<std::is_arithmetic<T>::value, T> compute(T const& edge_op_result)
  {
    using BlockReduce = cub::BlockReduce<T, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    return BlockReduce(temp_storage).Sum(edge_op_result);
  }

  template <typename T = EdgeOpResultType>
  __device__ std::enable_if_t<is_thrust_tuple<T>::value, T> compute(T const& edge_op_result)
  {
    return block_reduce_thrust_tuple<T, BlockSize>()(edge_op_result);
  }
};

template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BlockResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_out_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  BlockResultIterator block_result_first,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = typename std::iterator_traits<BlockResultIterator>::value_type;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx     = static_cast<size_t>(tid);

  e_op_result_t e_op_result_sum{};
  while (idx < static_cast<size_t>(matrix_partition.get_major_size())) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(idx);
    for (vertex_t i = 0; i < local_degree; ++i) {
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
    idx += gridDim.x * blockDim.x;
  }

  e_op_result_sum =
    block_reduce_edge_op_result<e_op_result_t,
                                transform_reduce_e_for_all_low_out_degree_block_size>()
      .compute(e_op_result_sum);
  if (threadIdx.x == 0) { *(block_result_first + blockIdx.x) = e_op_result_sum; }
}

}  // namespace detail

/**
 * @brief Iterate over the entire set of edges and reduce @p edge_op outputs.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam HandleType HandleType Type of the RAFT handle (e.g. for single-GPU or multi-GPU).
 * @tparam GraphViewType Type of the passed graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the binary (or ternary) edge operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_device_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_device_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Binary (or ternary) operator takes *(@p adj_matrix_row_value_input_first + i), *(@p
 * adj_matrix_col_value_input_first + j), (and optionally edge weight) (where i and j are row and
 * column indices, respectively) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p edge_op outputs.
 */
template <typename HandleType,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T>
T transform_reduce_e(HandleType& handle,
                     GraphViewType const& graph_view,
                     AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                     AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                     EdgeOp e_op,
                     T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  using vertex_t = typename GraphViewType::vertex_type;

  T result{};
  vertex_t row_value_input_offset{0};
  vertex_t col_value_input_offset{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    grid_1d_thread_t update_grid(
      matrix_partition.get_major_last() - matrix_partition.get_major_first(),
      detail::transform_reduce_e_for_all_low_out_degree_block_size,
      get_max_num_blocks_1D());

    rmm::device_vector<T> block_results(update_grid.num_blocks);

    detail::for_all_major_for_all_nbr_low_out_degree<<<update_grid.num_blocks,
                                                       update_grid.block_size,
                                                       0,
                                                       handle.get_stream()>>>(
      matrix_partition,
      adj_matrix_row_value_input_first + row_value_input_offset,
      adj_matrix_col_value_input_first + col_value_input_offset,
      block_results.data(),
      e_op);

    // FIXME: we have several options to implement this. With cooperative group support
    // (https://devblogs.nvidia.com/cooperative-groups/), we can run this synchronization within the
    // previous kernel. Using atomics at the end of the previous kernel is another option
    // (sequentialization due to atomics may not be bad as different blocks may reach the
    // synchronization point in varying timings and the number of SMs is not very big)
    auto partial_result =
      thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     block_results.begin(),
                     block_results.end(),
                     T(),
                     plus_thrust_tuple<T>());

    plus_thrust_tuple<T>()(result, partial_result);

    if (GraphViewType::is_adj_matrix_transposed) {
      col_value_input_offset += matrix_partition.get_major_size();
    } else {
      row_value_input_offset += matrix_partition.get_major_size();
    }
  }

  if (GraphViewType::is_multi_gpu) {
    // need reduction
    CUGRAPH_FAIL("unimplemented.");
  }

  return plus_thrust_tuple<T>()(init, result);
}

}  // namespace experimental
}  // namespace cugraph
