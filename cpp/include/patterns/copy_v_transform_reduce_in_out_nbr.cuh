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
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
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

  auto const tid          = threadIdx.x + blockIdx.x * blockDim.x;
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(major_last - major_first)) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    auto major_offset = major_start_offset + idx;
    thrust::tie(indices, weights, local_degree) =
      matrix_partition.get_local_edges(static_cast<vertex_t>(major_offset));
#if 1
    auto transform_op = [&matrix_partition,
                         &adj_matrix_row_value_input_first,
                         &adj_matrix_col_value_input_first,
                         &e_op,
                         major_offset,
                         indices,
                         weights] __device__(auto i) {
      auto minor        = indices[i];
      auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                   ? minor
                   : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                   : minor;
      auto row_offset = GraphViewType::is_adj_matrix_transposed
                          ? minor_offset
                          : static_cast<vertex_t>(major_offset);
      auto col_offset = GraphViewType::is_adj_matrix_transposed
                          ? static_cast<vertex_t>(major_offset)
                          : minor_offset;
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
                   : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                   : minor;
      auto row_offset = GraphViewType::is_adj_matrix_transposed
                          ? minor_offset
                          : static_cast<vertex_t>(major_offset);
      auto col_offset = GraphViewType::is_adj_matrix_transposed
                          ? static_cast<vertex_t>(major_offset)
                          : minor_offset;
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
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
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
  auto const lane_id      = tid % raft::warp_size();
  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(major_last - major_first)) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    auto major_offset                           = major_start_offset + idx;
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto e_op_result_sum =
      lane_id == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size) {
      auto minor        = indices[i];
      auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                   ? minor
                   : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                   : minor;
      auto row_offset = GraphViewType::is_adj_matrix_transposed
                          ? minor_offset
                          : static_cast<vertex_t>(major_offset);
      auto col_offset = GraphViewType::is_adj_matrix_transposed
                          ? static_cast<vertex_t>(major_offset)
                          : minor_offset;
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
  typename GraphViewType::vertex_type major_first,
  typename GraphViewType::vertex_type major_last,
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

  auto major_start_offset = static_cast<size_t>(major_first - matrix_partition.get_major_first());
  auto idx                = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(major_last - major_first)) {
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_degree{};
    auto major_offset                           = major_start_offset + idx;
    thrust::tie(indices, weights, local_degree) = matrix_partition.get_local_edges(major_offset);
    auto e_op_result_sum =
      threadIdx.x == 0 ? init : e_op_result_t{};  // relevent only if update_major == true
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      auto minor        = indices[i];
      auto weight       = weights != nullptr ? weights[i] : weight_t{1.0};
      auto minor_offset = matrix_partition.get_minor_offset_from_minor_nocheck(minor);
      auto row          = GraphViewType::is_adj_matrix_transposed
                   ? minor
                   : matrix_partition.get_major_from_major_offset_nocheck(major_offset);
      auto col = GraphViewType::is_adj_matrix_transposed
                   ? matrix_partition.get_major_from_major_offset_nocheck(major_offset)
                   : minor;
      auto row_offset = GraphViewType::is_adj_matrix_transposed
                          ? minor_offset
                          : static_cast<vertex_t>(major_offset);
      auto col_offset = GraphViewType::is_adj_matrix_transposed
                          ? static_cast<vertex_t>(major_offset)
                          : minor_offset;
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

template <bool in,  // iterate over incoming edges (in == true) or outgoing edges (in == false)
          typename GraphViewType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename T,
          typename VertexValueOutputIterator>
void copy_v_transform_reduce_nbr(raft::handle_t const& handle,
                                 GraphViewType const& graph_view,
                                 AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                                 AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                                 EdgeOp e_op,
                                 T init,
                                 VertexValueOutputIterator vertex_value_output_first)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(is_arithmetic_or_thrust_tuple_of_arithmetic<T>::value);

  auto loop_count = size_t{1};
  if (GraphViewType::is_multi_gpu) {
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    loop_count               = graph_view.is_hypergraph_partitioned()
                   ? graph_view.get_number_of_local_adj_matrix_partitions()
                   : static_cast<size_t>(row_comm_size);
  }
  auto comm_rank = handle.comms_initialized() ? handle.get_comms().get_rank() : int{0};

  auto minor_tmp_buffer_size =
    (GraphViewType::is_multi_gpu && (in != GraphViewType::is_adj_matrix_transposed))
      ? GraphViewType::is_adj_matrix_transposed
          ? graph_view.get_number_of_local_adj_matrix_partition_rows()
          : graph_view.get_number_of_local_adj_matrix_partition_cols()
      : vertex_t{0};
  auto minor_tmp_buffer   = allocate_comm_buffer<T>(minor_tmp_buffer_size, handle.get_stream());
  auto minor_buffer_first = get_comm_buffer_begin<T>(minor_tmp_buffer);

  if (in != GraphViewType::is_adj_matrix_transposed) {
    auto minor_init = init;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      minor_init = graph_view.is_hypergraph_partitioned() ? (row_comm_rank == 0) ? init : T{}
                                                          : (col_comm_rank == 0) ? init : T{};
    }

    if (GraphViewType::is_multi_gpu) {
      thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   minor_buffer_first,
                   minor_buffer_first + minor_tmp_buffer_size,
                   minor_init);
    } else {
      thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   vertex_value_output_first,
                   vertex_value_output_first + graph_view.get_number_of_local_vertices(),
                   minor_init);
    }
  } else {
    assert(minor_tmp_buffer_size == 0);
  }

  for (size_t i = 0; i < loop_count; ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(
      graph_view, (GraphViewType::is_multi_gpu && !graph_view.is_hypergraph_partitioned()) ? 0 : i);

    auto major_tmp_buffer_size = vertex_t{0};
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      major_tmp_buffer_size =
        (in == GraphViewType::is_adj_matrix_transposed)
          ? graph_view.is_hypergraph_partitioned()
              ? matrix_partition.get_major_size()
              : graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i)
          : vertex_t{0};
    }
    auto major_tmp_buffer   = allocate_comm_buffer<T>(major_tmp_buffer_size, handle.get_stream());
    auto major_buffer_first = get_comm_buffer_begin<T>(major_tmp_buffer);

    auto major_init = T{};
    if (in == GraphViewType::is_adj_matrix_transposed) {
      if (GraphViewType::is_multi_gpu) {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto const row_comm_rank = row_comm.get_rank();
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();
        major_init = graph_view.is_hypergraph_partitioned() ? (col_comm_rank == 0) ? init : T{}
                                                            : (row_comm_rank == 0) ? init : T{};
      } else {
        major_init = init;
      }
    }

    int comm_root_rank = 0;
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      comm_root_rank = graph_view.is_hypergraph_partitioned() ? i * row_comm_size + row_comm_rank
                                                              : col_comm_rank * row_comm_size + i;
    }

    if (graph_view.get_vertex_partition_size(comm_root_rank) > 0) {
      raft::grid_1d_thread_t update_grid(graph_view.get_vertex_partition_size(comm_root_rank),
                                         detail::copy_v_transform_reduce_nbr_for_all_block_size,
                                         handle.get_device_properties().maxGridSize[0]);

      if (GraphViewType::is_multi_gpu) {
        auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
        auto const row_comm_size = row_comm.get_size();
        auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
        auto const col_comm_rank = col_comm.get_rank();

        auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                        ? vertex_t{0}
                                        : matrix_partition.get_major_value_start_offset();
        auto col_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                        ? matrix_partition.get_major_value_start_offset()
                                        : vertex_t{0};

        detail::for_all_major_for_all_nbr_low_degree<in == GraphViewType::is_adj_matrix_transposed>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            graph_view.get_vertex_partition_first(comm_root_rank),
            graph_view.get_vertex_partition_last(comm_root_rank),
            adj_matrix_row_value_input_first + row_value_input_offset,
            adj_matrix_col_value_input_first + col_value_input_offset,
            (in == GraphViewType::is_adj_matrix_transposed) ? major_buffer_first
                                                            : minor_buffer_first,
            e_op,
            major_init);
      } else {
        detail::for_all_major_for_all_nbr_low_degree<in == GraphViewType::is_adj_matrix_transposed>
          <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
            matrix_partition,
            graph_view.get_vertex_partition_first(comm_root_rank),
            graph_view.get_vertex_partition_last(comm_root_rank),
            adj_matrix_row_value_input_first,
            adj_matrix_col_value_input_first,
            vertex_value_output_first,
            e_op,
            major_init);
      }
    }

    if (GraphViewType::is_multi_gpu && (in == GraphViewType::is_adj_matrix_transposed)) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();
      auto const col_comm_size = col_comm.get_size();

      if (graph_view.is_hypergraph_partitioned()) {
        device_reduce(
          col_comm,
          major_buffer_first,
          vertex_value_output_first,
          static_cast<size_t>(graph_view.get_vertex_partition_size(i * row_comm_size + i)),
          raft::comms::op_t::SUM,
          i,
          handle.get_stream());
      } else {
        device_reduce(row_comm,
                      major_buffer_first,
                      vertex_value_output_first,
                      static_cast<size_t>(
                        graph_view.get_vertex_partition_size(col_comm_rank * row_comm_size + i)),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
      }
    }

    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // this is as necessary major_tmp_buffer will become out-of-scope once
                              // control flow exits this block (FIXME: we can reduce stream
                              // synchronization if we compute the maximum major_tmp_buffer_size and
                              // allocate major_tmp_buffer outside the loop)
  }

  if (GraphViewType::is_multi_gpu && (in != GraphViewType::is_adj_matrix_transposed)) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    if (graph_view.is_hypergraph_partitioned()) {
      CUGRAPH_FAIL("unimplemented.");
    } else {
      for (int i = 0; i < col_comm_size; ++i) {
        auto offset = (graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + i) -
                       graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size));
        auto size   = static_cast<size_t>(
          graph_view.get_vertex_partition_size(row_comm_rank * col_comm_size + i));
        device_reduce(col_comm,
                      minor_buffer_first + offset,
                      minor_buffer_first + offset,
                      size,
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
      }

      // FIXME: this P2P is unnecessary if we apply the partitioning scheme used with hypergraph
      // partitioning
      auto comm_src_rank = (comm_rank % col_comm_size) * row_comm_size + comm_rank / col_comm_size;
      auto comm_dst_rank = row_comm_rank * col_comm_size + col_comm_rank;
      // FIXME: this branch may no longer necessary with NCCL backend
      if (comm_src_rank == comm_rank) {
        assert(comm_dst_rank == comm_rank);
        auto offset =
          graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + col_comm_rank) -
          graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size);
        auto size = static_cast<size_t>(
          graph_view.get_vertex_partition_size(row_comm_rank * col_comm_size + col_comm_rank));
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     minor_buffer_first + offset,
                     minor_buffer_first + offset + size,
                     vertex_value_output_first);
      } else {
        device_sendrecv<decltype(minor_buffer_first), VertexValueOutputIterator>(
          comm,
          minor_buffer_first +
            (graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size + col_comm_rank) -
             graph_view.get_vertex_partition_first(row_comm_rank * col_comm_size)),
          static_cast<size_t>(
            graph_view.get_vertex_partition_size(row_comm_rank * col_comm_size + col_comm_rank)),
          comm_dst_rank,
          vertex_value_output_first,
          static_cast<size_t>(graph_view.get_vertex_partition_size(comm_rank)),
          comm_src_rank,
          handle.get_stream());
      }
    }
  }

  CUDA_TRY(cudaStreamSynchronize(
    handle.get_stream()));  // this is as necessary minor_tmp_buffer will become out-of-scope once
                            // control flow exits this block
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
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
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
  detail::copy_v_transform_reduce_nbr<true>(handle,
                                            graph_view,
                                            adj_matrix_row_value_input_first,
                                            adj_matrix_col_value_input_first,
                                            e_op,
                                            init,
                                            vertex_value_output_first);
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
 * @p graph_view.get_number_of_local_adj_matrix_partition_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p
 * adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_local_adj_matrix_partition_cols().
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
  detail::copy_v_transform_reduce_nbr<false>(handle,
                                             graph_view,
                                             adj_matrix_row_value_input_first,
                                             adj_matrix_col_value_input_first,
                                             e_op,
                                             init,
                                             vertex_value_output_first);
}

}  // namespace experimental
}  // namespace cugraph
