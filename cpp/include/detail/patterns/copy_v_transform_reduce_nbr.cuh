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

#include <detail/patterns/edge_op_utils.cuh>
#include <detail/patterns/reduce_op.cuh>
#include <detail/utilities/cuda.cuh>
#include <graph.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <type_traits>
#include <utility>

namespace {

// FIXME: block size requires tuning
int32_t constexpr copy_v_transform_reduce_nbr_for_all_low_out_degree_block_size = 128;

template <bool update_major, typename T>
__device__ std::enable_if_t<update_major, void> accumulate_edge_op_result(T& lhs, T const& rhs)
{
  lhs = cugraph::experimental::detail::plus_edge_op_result(lhs, rhs);
}

template <bool update_major, typename T>
__device__ std::enable_if_t<!update_major, void> accumulate_edge_op_result(T& lhs, T const& rhs)
{
  cugraph::experimental::detail::atomic_add(&lhs, rhs);
}

template <bool update_major,
          typename GraphType,
          typename MajorIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultValueOutputIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_out_degree(
  GraphType graph_device_view,
  MajorIterator major_first,
  MajorIterator major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  typename std::iterator_traits<ResultValueOutputIterator>::value_type
    init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphType::vertex_type;
  using weight_t      = typename GraphType::weight_type;
  using e_op_result_t = typename std::iterator_traits<ResultValueOutputIterator>::value_type;

  auto num_majors = static_cast<size_t>(thrust::distance(major_first, major_last));
  auto const tid  = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx      = tid;

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
    e_op_result_t e_op_result_sum{init};  // relevent only if update_major == true
    for (vertex_t i = 0; i < local_degree; ++i) {
      auto minor_vid = indices[i];
      auto weight    = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset =
        GraphType::is_adj_matrix_transposed
          ? graph_device_view.get_adj_matrix_local_row_offset_from_row_nocheck(minor_vid)
          : graph_device_view.get_adj_matrix_local_col_offset_from_col_nocheck(minor_vid);
      auto row_offset = GraphType::is_adj_matrix_transposed ? minor_offset : major_offset;
      auto col_offset = GraphType::is_adj_matrix_transposed ? major_offset : minor_offset;
      auto e_op_result =
        cugraph::experimental::detail::evaluate_edge_op<GraphType,
                                                        EdgeOp,
                                                        AdjMatrixRowValueInputIterator,
                                                        AdjMatrixColValueInputIterator>()
          .compute(*(adj_matrix_row_value_input_first + row_offset),
                   *(adj_matrix_col_value_input_first + col_offset),
                   weight,
                   e_op);
      if (update_major) {
        accumulate_edge_op_result<update_major>(e_op_result_sum, e_op_result);
      } else {
        accumulate_edge_op_result<update_major>(*(result_value_output_first + minor_offset),
                                                e_op_result);
      }
    }
    if (update_major) { *(result_value_output_first + idx) = e_op_result_sum; }
    idx += gridDim.x * blockDim.x;
  }
}

}  // namespace

namespace cugraph {
namespace experimental {
namespace detail {

/**
 * @brief Iterate over the incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the incoming edges part)
 * and thrust::copy() (update vertex properties part, take transform_reduce output as copy input).
 *
 * @tparam HandleType Type of the RAFT handle (e.g. for single-GPU or OPG).
 * @tparam GraphType Type of the passed graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the binary (or ternary) edge operator.
 * @tparam T Type of the initial value for reduction over the incoming edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in OPG).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_device_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in OPG).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_device_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Binary (or ternary) operator takes *(@p adj_matrix_row_value_input_first + i), *(@p
 * adj_matrix_col_value_input_first + j), (and optionally edge weight) (where i and j are row and
 * column indices, respectively) and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in OPG). `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p graph_device_view.get_number_of_local_vertices().
 */
template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_in_nbr(HandleType& handle,
                                    GraphType const& graph_device_view,
                                    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                                    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                                    EdgeOp e_op,
                                    T init,
                                    VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(GraphType::is_adj_matrix_transposed ||
                cugraph::experimental::detail::is_atomically_addable<T>::value);

  grid_1d_thread_t update_grid(graph_device_view.get_number_of_adj_matrix_local_cols(),
                               copy_v_transform_reduce_nbr_for_all_low_out_degree_block_size,
                               get_max_num_blocks_1D());

  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    if (!GraphType::is_adj_matrix_transposed) {
      thrust::fill(thrust::cuda::par.on(handle.get_stream()),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_device_view.get_number_of_local_vertices(),
                   init);
    }

    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_cols());
    for_all_major_for_all_nbr_low_out_degree<GraphType::is_adj_matrix_transposed>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
        graph_device_view,
        graph_device_view.adj_matrix_local_col_begin(),
        graph_device_view.adj_matrix_local_col_end(),
        adj_matrix_row_value_input_first,
        adj_matrix_col_value_input_first,
        vertex_value_output_first,
        e_op,
        init);
  }
}

/**
 * @brief Iterate over the outgoing edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges part)
 * and thrust::copy() (update vertex properties part, take transform_reduce output as copy input).
 *
 * @tparam HandleType Type of the RAFT handle (e.g. for single-GPU or OPG).
 * @tparam GraphType Type of the passed graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the binary (or ternary) edge operator.
 * @tparam T Type of the initial value for reduction over the outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in OPG).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_device_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in OPG).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_device_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Binary (or ternary) operator takes *(@p adj_matrix_row_value_input_first + i), *(@p
 * adj_matrix_col_value_input_first + j), (and optionally edge weight) (where i and j are row and
 * column indices, respectively) and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in OPG). `vertex_value_output_last` (exclusive) is
 * deduced as @p vertex_value_output_first + @p graph_device_view.get_number_of_local_vertices().
 */
template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_out_nbr(
  HandleType& handle,
  GraphType const& graph_device_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  EdgeOp e_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);
  static_assert(!GraphType::is_adj_matrix_transposed ||
                cugraph::experimental::detail::is_atomically_addable<T>::value);

  grid_1d_thread_t update_grid(graph_device_view.get_number_of_adj_matrix_local_rows(),
                               copy_v_transform_reduce_nbr_for_all_low_out_degree_block_size,
                               get_max_num_blocks_1D());

  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    if (GraphType::is_adj_matrix_transposed) {
      thrust::fill(thrust::cuda::par.on(handle.get_stream()),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_device_view.get_number_of_local_vertices(),
                   init);
    }

    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_cols());
    for_all_major_for_all_nbr_low_out_degree<!GraphType::is_adj_matrix_transposed>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
        graph_device_view,
        graph_device_view.adj_matrix_local_row_begin(),
        graph_device_view.adj_matrix_local_row_end(),
        adj_matrix_row_value_input_first,
        adj_matrix_col_value_input_first,
        vertex_value_output_first,
        e_op,
        init);
  }
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
