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

#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr count_if_e_for_all_low_out_degree_block_size = 128;

// FIXME: function names conflict if included with transform_reduce_e.cuh
template <typename GraphType,
          typename MajorIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_out_degree(
  GraphType graph_device_view,
  MajorIterator major_first,
  MajorIterator major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  typename GraphType::edge_type* block_counts,
  EdgeOp e_op)
{
  using vertex_t = typename GraphType::vertex_type;
  using edge_t   = typename GraphType::edge_type;
  using weight_t = typename GraphType::weight_type;

  auto num_majors = static_cast<size_t>(thrust::distance(major_first, major_last));
  auto const tid  = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx      = tid;

  edge_t count{0};
  while (idx < num_majors) {
    auto major = *(major_first + idx);
    auto major_offset =
      GraphType::is_adj_matrix_transposed
        ? graph_device_view.get_adj_matrix_local_col_offset_from_col_nocheck(major)
        : graph_device_view.get_adj_matrix_local_row_offset_from_row_nocheck(major);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    vertex_t local_degree{};
    thrust::tie(indices, weights, local_degree) = graph_device_view.get_local_edges(major_offset);
    for (vertex_t i = 0; i < local_degree; ++i) {
      auto minor_vid = indices[i];
      auto weight    = weights != nullptr ? weights[i] : 1.0;
      auto minor_offset =
        GraphType::is_adj_matrix_transposed
          ? graph_device_view.get_adj_matrix_local_row_offset_from_row_nocheck(minor_vid)
          : graph_device_view.get_adj_matrix_local_col_offset_from_col_nocheck(minor_vid);
      auto row_offset = GraphType::is_adj_matrix_transposed ? minor_offset : major_offset;
      auto col_offset = GraphType::is_adj_matrix_transposed ? major_offset : minor_offset;
      if (e_op(*(adj_matrix_row_value_input_first + row_offset),
               *(adj_matrix_col_value_input_first + col_offset),
               weight)) {
        count++;
      }
    }
    idx += gridDim.x * blockDim.x;
  }

  using BlockReduce = cub::BlockReduce<edge_t, count_if_e_for_all_low_out_degree_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  count = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) { *(block_counts + blockIdx.x) = count; }
}

}  // namespace detail

/**
 * @brief Count the number of edges that satisfies the given predicate.
 *
 * This function is inspired by thrust::count_if().
 *
 * @tparam HandleType Type of the RAFT handle (e.g. for single-GPU or multi-GPU).
 * @tparam GraphType Type of the passed graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the binary (or ternary) edge operator.
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
 * column indices, respectively) and returns true if this edge should be included in the returned
 * count.
 * @return GraphType::edge_type Number of times @p e_op returned true.
 */
template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp>
typename GraphType::edge_type count_if_e(
  HandleType& handle,
  GraphType const& graph_device_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  EdgeOp e_op)
{
  using edge_t = typename GraphType::edge_type;

  grid_1d_thread_t update_grid(GraphType::is_adj_matrix_transposed
                                 ? graph_device_view.get_number_of_adj_matrix_local_cols()
                                 : graph_device_view.get_number_of_adj_matrix_local_rows(),
                               detail::count_if_e_for_all_low_out_degree_block_size,
                               get_max_num_blocks_1D());

  rmm::device_vector<edge_t> block_counts(update_grid.num_blocks);

  detail::for_all_major_for_all_nbr_low_out_degree<<<update_grid.num_blocks,
                                                     update_grid.block_size,
                                                     0,
                                                     handle.get_stream()>>>(
    graph_device_view,
    GraphType::is_adj_matrix_transposed ? graph_device_view.adj_matrix_local_col_begin()
                                        : graph_device_view.adj_matrix_local_row_begin(),
    GraphType::is_adj_matrix_transposed ? graph_device_view.adj_matrix_local_col_end()
                                        : graph_device_view.adj_matrix_local_row_end(),
    adj_matrix_row_value_input_first,
    adj_matrix_col_value_input_first,
    block_counts.data().get(),
    e_op);

  // FIXME: we have several options to implement this. With cooperative group support
  // (https://devblogs.nvidia.com/cooperative-groups/), we can run this synchronization within the
  // previous kernel. Using atomics at the end of the previous kernel is another option
  // (sequentialization due to atomics may not be bad as different blocks may reach the
  // synchronization point in varying timings and the number of SMs is not very big)
  auto count = thrust::reduce(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                              block_counts.begin(),
                              block_counts.end(),
                              edge_t{0},
                              thrust::plus<edge_t>());

  if (GraphType::is_multi_gpu) {
    // need to reduce count
    CUGRAPH_FAIL("unimplemented.");
  }

  return count;
}

}  // namespace experimental
}  // namespace cugraph
