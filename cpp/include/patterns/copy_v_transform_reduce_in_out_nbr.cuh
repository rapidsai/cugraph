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

#include <experimental/graph_view.hpp>
#include <matrix_partition_device.cuh>
#include <patterns/edge_op_utils.cuh>
#include <patterns/reduce_op.cuh>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <type_traits>
#include <utility>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr copy_v_transform_reduce_nbr_for_all_block_size = 128;

#if 0
// FIXME: delete this once we verify that the thrust replace in for_all_major_for_all_nbr_low_degree is no slower than the original for loop based imoplementation
template <bool update_major, typename T>
__device__ std::enable_if_t<update_major, void> accumulate_edge_op_result(T& lhs, T const& rhs)
{
  lhs = plus_edge_op_result(lhs, rhs);
}

template <bool update_major, typename T>
__device__ std::enable_if_t<!update_major, void> accumulate_edge_op_result(T& lhs, T const& rhs)
{
  atomic_add(&lhs, rhs);
}
#endif

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_low_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = T;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(matrix_partition.get_major_size())) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(idx);
#if 1
    auto transform_op = [&matrix_partition,
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
    };

    if (update_major) {
      *(result_value_output_first + idx) = thrust::transform_reduce(
        thrust::seq,
        thrust::make_counting_iterator(edge_t{0}),
        thrust::make_counting_iterator(local_degree),
        transform_op,
        init,
        [] __device__(auto lhs, auto rhs) { return plus_edge_op_result(lhs, rhs); });
    } else {
      thrust::for_each(
        thrust::seq,
        thrust::make_counting_iterator(edge_t{0}),
        thrust::make_counting_iterator(local_degree),
        [&matrix_partition, indices, &result_value_output_first, &transform_op] __device__(auto i) {
          auto e_op_result  = transform_op(i);
          auto minor        = indices[i];
          auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
          atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
        });
    }
#else
    // FIXME: delete this once we verify that the code above is not slower than this.
    e_op_result_t e_op_result_sum{init};  // relevent only if update_major == true
    for (edge_t i = 0; i < local_degree; ++i) {
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
      if (update_major) {
        accumulate_edge_op_result<update_major>(e_op_result_sum, e_op_result);
      } else {
        accumulate_edge_op_result<update_major>(*(result_value_output_first + minor_offset),
                                                e_op_result);
      }
    }
    if (update_major) { *(result_value_output_first + idx) = e_op_result_sum; }
#endif
    idx += gridDim.x * blockDim.x;
  }
}

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_mid_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = T;

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(copy_v_transform_reduce_nbr_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(matrix_partition.get_major_size())) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(idx);
    auto e_op_result_sum =
      lane_id == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size) {
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
      if (update_major) {
        e_op_result_sum = plus_edge_op_result(e_op_result_sum, e_op_result);
      } else {
        atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
      }
    }
    if (update_major) {
      e_op_result_sum = warp_reduce_edge_op_result<e_op_result_t>().compute(e_op_result_sum);
      if (lane_id == 0) { *(result_value_output_first + idx) = e_op_result_sum; }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <bool update_major,
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename ResultValueOutputIterator,
          typename EdgeOp,
          typename T>
__global__ void for_all_major_for_all_nbr_high_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  ResultValueOutputIterator result_value_output_first,
  EdgeOp e_op,
  T init /* relevent only if update_major == true */)
{
  using vertex_t      = typename GraphViewType::vertex_type;
  using edge_t        = typename GraphViewType::edge_type;
  using weight_t      = typename GraphViewType::weight_type;
  using e_op_result_t = T;

  auto idx = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(matrix_partition.get_major_size())) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(idx);
    auto e_op_result_sum =
      threadIdx.x == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
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
      if (update_major) {
        e_op_result_sum = plus_edge_op_result(e_op_result_sum, e_op_result);
      } else {
        atomic_accumulate_edge_op_result(result_value_output_first + minor_offset, e_op_result);
      }
    }
    if (update_major) {
      e_op_result_sum =
        block_reduce_edge_op_result<e_op_result_t, copy_v_transform_reduce_nbr_for_all_block_size>()
          .compute(e_op_result_sum);
      if (threadIdx.x == 0) { *(result_value_output_first + idx) = e_op_result_sum; }
    }

    idx += gridDim.x;
  }
}

}  // namespace detail

/**
 * @brief Iterate over the incoming edges to update vertex properties.
 *
 * This function is inspired by thrust::transfrom_reduce() (iteration over the incoming edges part)
 * and thrust::copy() (update vertex properties part, take transform_reduce output as copy input).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value for reduction over the incoming edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first +
 * @p graph_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional edge
 * weight), *(@p adj_matrix_row_value_input_first + i), and *(@p adj_matrix_col_value_input_first +
 * j) (where i is in [0, graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_in_nbr(raft::handle_t const& handle,
                                    GraphViewType const& graph_view,
                                    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                                    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                                    EdgeOp e_op,
                                    T init,
                                    VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  if (GraphViewType::is_multi_gpu) {
    for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

      raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                         detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);

      auto tmp_buffer   = allocate_comm_buffer<T>(GraphViewType::is_adj_matrix_transposed
                                                  ? matrix_partition.get_major_size()
                                                  : matrix_partition.get_minor_size(),
                                                handle.get_stream());
      auto buffer_first = get_comm_buffer_begin<T>(tmp_buffer);

      if (!GraphViewType::is_adj_matrix_transposed) {
        thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     buffer_first,
                     buffer_first + (GraphViewType::is_adj_matrix_transposed
                                       ? matrix_partition.get_major_size()
                                       : matrix_partition.get_minor_size()),
                     T{});
      }

      detail::for_all_major_for_all_nbr_low_degree<GraphViewType::is_adj_matrix_transposed>
        <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
          matrix_partition,
          adj_matrix_row_value_input_first,
          adj_matrix_col_value_input_first,
          buffer_first,
          e_op,
          T{});

      if (GraphViewType::is_adj_matrix_transposed) {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto const row_comm_rank = row_comm.get_rank();
        auto const row_comm_size = row_comm.get_size();
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();
        auto const col_comm_size = col_comm.get_size();

        if (graph_view.is_hypergraph_partitioned()) {
          device_reduce(col_comm,
                        buffer_first,
                        vertex_value_output_first,
                        graph_view.get_vertex_partition_last(i * row_comm_size + i) -
                          graph_view.get_vertex_partition_first(i * row_comm_size + i),
                        raft::comms::op_t::SUM,
                        i,
                        handle.get_stream());
        } else {
          for (int j = 0; j < row_comm_size; ++j) {
            auto comm_root_rank = col_comm_rank * row_comm_size + j;
            device_reduce(
              row_comm,
              buffer_first + (graph_view.get_vertex_partition_first(comm_root_rank) -
                              graph_view.get_vertex_partition_first(col_comm_rank * row_comm_size)),
              vertex_value_output_first,
              static_cast<size_t>(graph_view.get_vertex_partition_last(comm_root_rank) -
                                  graph_view.get_vertex_partition_first(comm_root_rank)),
              raft::comms::op_t::SUM,
              j,
              handle.get_stream());
          }
        }
      } else {
        CUGRAPH_FAIL("unimplemented.");
      }
    }

    // FIXME: actually, we can avoid this.
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      vertex_value_output_first,
                      vertex_value_output_first +
                        (graph_view.get_local_vertex_last() - graph_view.get_local_vertex_first()),
                      vertex_value_output_first,
                      [init] __device__(auto val) { return plus_edge_op_result(val, init); });
  } else {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, 0);

    raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                       detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                       handle.get_device_properties().maxGridSize[0]);

    if (!GraphViewType::is_adj_matrix_transposed) {
      thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.get_number_of_local_vertices(),
                   init);
    }

    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_adj_matrix_local_rows());
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_adj_matrix_local_cols());
    detail::for_all_major_for_all_nbr_low_degree<GraphViewType::is_adj_matrix_transposed>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
        matrix_partition,
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
 * This function is inspired by thrust::transfrom_reduce() (iteration over the outgoing edges
 * part) and thrust::copy() (update vertex properties part, take transform_reduce output as copy
 * input).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam T Type of the initial value for reduction over the outgoing edges.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex output property variables.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param adj_matrix_row_value_input_first Iterator pointing to the adjacency matrix row input
 * properties for the first (inclusive) row (assigned to this process in multi-GPU).
 * `adj_matrix_row_value_input_last` (exclusive) is deduced as @p adj_matrix_row_value_input_first
 * +
 * @p graph_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p
 * adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Quaternary (or quinary) operator takes edge source, edge destination, (optional
 * edge weight), *(@p adj_matrix_row_value_input_first + i), and *(@p
 * adj_matrix_col_value_input_first + j) (where i is in [0,
 * graph_view.get_number_of_local_adj_matrix_partition_rows()) and j is in [0,
 * get_number_of_local_adj_matrix_partition_cols())) and returns a value to be reduced.
 * @param init Initial value to be added to the reduced @e_op return values for each vertex.
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the
 * first (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 */
template <typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  EdgeOp e_op,
  T init,
  VertexValueOutputIterator vertex_value_output_first)
{
  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  // FIXME: this code is highly repetitive of copy_v_transform_reduce_in_nbr, better factor out.
  if (GraphViewType::is_multi_gpu) {
    for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
      matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

      raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                         detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);

      auto tmp_buffer   = allocate_comm_buffer<T>(GraphViewType::is_adj_matrix_transposed
                                                  ? matrix_partition.get_minor_size()
                                                  : matrix_partition.get_major_size(),
                                                handle.get_stream());
      auto buffer_first = get_comm_buffer_begin<T>(tmp_buffer);

      if (!GraphViewType::is_adj_matrix_transposed) {
        thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     buffer_first,
                     buffer_first + (GraphViewType::is_adj_matrix_transposed
                                       ? matrix_partition.get_minor_size()
                                       : matrix_partition.get_major_size()),
                     T{});
      }

      detail::for_all_major_for_all_nbr_low_degree<GraphViewType::is_adj_matrix_transposed>
        <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
          matrix_partition,
          adj_matrix_row_value_input_first,
          adj_matrix_col_value_input_first,
          buffer_first,
          e_op,
          T{});

      if (GraphViewType::is_adj_matrix_transposed) {
        CUGRAPH_FAIL("unimplemented.");
      } else {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto const row_comm_rank = row_comm.get_rank();
        auto const row_comm_size = row_comm.get_size();
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();
        auto const col_comm_size = col_comm.get_size();

        if (graph_view.is_hypergraph_partitioned()) {
          device_reduce(col_comm,
                        buffer_first,
                        vertex_value_output_first,
                        graph_view.get_vertex_partition_last(i * row_comm_size + i) -
                          graph_view.get_vertex_partition_first(i * row_comm_size + i),
                        raft::comms::op_t::SUM,
                        i,
                        handle.get_stream());
        } else {
          for (int j = 0; j < row_comm_size; ++j) {
            auto comm_root_rank = col_comm_rank * row_comm_size + j;
            device_reduce(
              row_comm,
              buffer_first + (graph_view.get_vertex_partition_first(comm_root_rank) -
                              graph_view.get_vertex_partition_first(col_comm_rank * row_comm_size)),
              vertex_value_output_first,
              static_cast<size_t>(graph_view.get_vertex_partition_last(comm_root_rank) -
                                  graph_view.get_vertex_partition_first(comm_root_rank)),
              raft::comms::op_t::SUM,
              j,
              handle.get_stream());
          }
        }
      }
    }

    // FIXME: actually, we can avoid this.
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      vertex_value_output_first,
                      vertex_value_output_first +
                        (graph_view.get_local_vertex_last() - graph_view.get_local_vertex_first()),
                      vertex_value_output_first,
                      [init] __device__(auto val) { return plus_edge_op_result(val, init); });
  } else {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, 0);

    raft::grid_1d_thread_t update_grid(matrix_partition.get_major_size(),
                                       detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                       handle.get_device_properties().maxGridSize[0]);

    if (GraphViewType::is_adj_matrix_transposed) {
      thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.get_number_of_local_vertices(),
                   init);
    }

    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_rows());
    assert(graph_view.get_number_of_local_vertices() ==
           graph_view.get_number_of_local_adj_matrix_partition_cols());
    detail::for_all_major_for_all_nbr_low_degree<!GraphViewType::is_adj_matrix_transposed>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
        matrix_partition,
        adj_matrix_row_value_input_first,
        adj_matrix_col_value_input_first,
        vertex_value_output_first,
        e_op,
        init);
  }
}

}  // namespace experimental
}  // namespace cugraph
