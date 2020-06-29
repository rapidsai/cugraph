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

#include <detail/graph_device_view.cuh>
#include <detail/patterns/edge_op_utils.cuh>
#include <detail/utilities/cuda.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/tuple.h>
#include <cub/cub.cuh>

#include <cstdint>
#include <type_traits>

namespace {

// FIXME: block size requires tuning
int32_t constexpr transform_reduce_e_for_all_low_out_degree_block_size = 128;

template <typename T>
__device__ std::enable_if_t<std::is_arithmetic<T>::value, T> plus_edge_op_result(T const& lhs,
                                                                                 T const& rhs)
{
  return lhs + rhs;
}

template <typename T>
__device__ std::enable_if_t<cugraph::experimental::detail::is_thrust_tuple<T>::value, T>
plus_edge_op_result(T const& lhs, T const& rhs)
{
  return cugraph::experimental::detail::plus_thrust_tuple<T>()(lhs, rhs);
}

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
  __device__ std::enable_if_t<cugraph::experimental::detail::is_thrust_tuple<T>::value, T> compute(
    T const& edge_op_result)
  {
    return cugraph::experimental::detail::block_reduce_thrust_tuple<T, BlockSize>()(edge_op_result);
  }
};

template <typename GraphType,
          typename MajorIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BlockResultIterator,
          typename EdgeOp>
__global__ void for_all_major_for_all_nbr_low_out_degree(
  GraphType const& graph_device_view,
  MajorIterator major_first,
  MajorIterator major_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  BlockResultIterator block_result_first,
  EdgeOp e_op)
{
  using vertex_t      = typename GraphType::vertex_type;
  using weight_t      = typename GraphType::weight_type;
  using e_op_result_t = typename std::iterator_traits<BlockResultIterator>::value_type;

  auto num_majors = static_cast<size_t>(thrust::distance(major_first, major_last));
  auto const tid  = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx      = tid;

  e_op_result_t e_op_result_sum{};
  while (idx < num_majors) {
    auto major_offset =
      GraphType::is_adj_matrix_transposed
        ? graph_device_view.get_adj_matrix_local_col_offset_from_col_nocheck(*(major_first + idx))
        : graph_device_view.get_adj_matrix_local_row_offset_from_row_nocheck(*(major_first + idx));
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
      auto e_op_result =
        cugraph::experimental::detail::evaluate_edge_op<GraphType,
                                                        EdgeOp,
                                                        AdjMatrixRowValueInputIterator,
                                                        AdjMatrixColValueInputIterator>()
          .compute(*(adj_matrix_row_value_input_first + row_offset),
                   *(adj_matrix_col_value_input_first + col_offset),
                   weight,
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

}  // namespace

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType,
          typename GraphType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T>
T transform_reduce_e(HandleType& handle,
                     GraphType const& graph_device_view,
                     AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                     AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                     EdgeOp e_op,
                     T init)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  grid_1d_thread_t update_grid(GraphType::is_adj_matrix_transposed
                                 ? graph_device_view.get_number_of_adj_matrix_local_cols()
                                 : graph_device_view.get_number_of_adj_matrix_local_rows(),
                               transform_reduce_e_for_all_low_out_degree_block_size,
                               get_max_num_blocks_1D());

  rmm::device_vector<T> block_results(update_grid.num_blocks);

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
    block_results.data(),
    e_op);

  // FIXME: we have several options to implement this. With cooperative group support
  // (https://devblogs.nvidia.com/cooperative-groups/), we can run this synchronization within the
  // previous kernel. Using atomics at the end of the previous kernel is another option
  // (sequentialization due to atomics may not be bad as different blocks may reach the
  // synchronization point in varying timings and the number of SMs is not very big)
  auto result = thrust::reduce(thrust::cuda::par.on(handle.get_stream()),
                               block_results.begin(),
                               block_results.end(),
                               T(),
                               plus_thrust_tuple<T>());

  if (GraphType::is_opg) {
    // need reduction
    CUGRAPH_FAIL("unimplemented.");
  }

  return plus_thrust_tuple<T>()(init, result);
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
