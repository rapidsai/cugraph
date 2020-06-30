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
int32_t constexpr copy_v_transform_reduce_e_for_all_low_out_degree_block_size = 128;

template <typename GraphType,
          typename MajorIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultValueOutputIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_out_degree(
  GraphType const& graph_device_view,
  MajorIterator major_first,
  MajorIterator major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op)
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
    e_op_result_t e_op_result_sum{};
    for (vertex_t i = 0; i < local_degree; ++i) {
      auto minor_vid = indices[i];
      auto weight    = weights != nullptr ? weights[i] : 1.0;
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
      e_op_result_sum = cugraph::experimental::detail::plus_edge_op_result(e_op_result_sum, e_op_result);
    }
    *(result_value_output_first + idx) = e_op_result_sum;
    idx += gridDim.x * blockDim.x;
  }
}

}  // namespace

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexValueOutputIterator,
          typename EdgeOp,
          typename T>
void copy_v_transform_reduce_e(HandleType& handle,
                             GraphType const& graph_device_view,
                             AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                             AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                             VertexValueOutputIterator vertex_value_output_first,
                             EdgeOp e_op,
                             T init)
{
  grid_1d_thread_t update_grid(GraphType::is_adj_matrix_transposed
                                 ? graph_device_view.get_number_of_adj_matrix_local_cols()
                                 : graph_device_view.get_number_of_adj_matrix_local_rows(),
                               copy_v_transform_reduce_e_for_all_low_out_degree_block_size,
                               get_max_num_blocks_1D());

  if (GraphType::is_opg) {
    CUGRAPH_FAIL("unimplemented.");
  } else {
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_rows());
    assert(graph_device_view.get_number_of_local_vertices() ==
           graph_device_view.get_number_of_adj_matrix_local_cols());
    for_all_major_for_all_nbr_low_out_degree<<<update_grid.num_blocks,
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
      vertex_value_output_first,
      e_op);
  }
  // FIXME: init
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph