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

#include <cstdlib>
#include <experimental/graph_view.hpp>
#include <matrix_partition_device.cuh>
#include <partition_manager.hpp>
#include <patterns/edge_op_utils.cuh>
#include <patterns/reduce_op.cuh>
#include <utilities/comm_utils.cuh>
#include <utilities/error.hpp>
#include <utilities/thrust_tuple_utils.cuh>
#include <vertex_partition_device.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {
namespace experimental {

namespace detail {

// FIXME: block size requires tuning
int32_t constexpr update_frontier_v_push_if_out_nbr_for_all_block_size = 128;
int32_t constexpr update_frontier_v_push_if_out_nbr_update_block_size  = 128;

template <typename GraphViewType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_low_degree(
  matrix_partition_device_t<GraphViewType> matrix_partition,
  RowIterator row_first,
  RowIterator row_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using weight_t = typename GraphViewType::weight_type;

  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  auto num_rows  = static_cast<size_t>(thrust::distance(row_first, row_last));
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx     = tid;

  while (idx < num_rows) {
    vertex_t row    = *(row_first + idx);
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (vertex_t i = 0; i < local_out_degree; ++i) {
      auto col         = indices[i];
      auto weight      = weights != nullptr ? weights[i] : 1.0;
      auto col_offset  = matrix_partition.get_minor_offset_from_minor_nocheck(col);
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
      if (thrust::get<0>(e_op_result) == true) {
        // FIXME: This atomicAdd serializes execution. If we renumber vertices to insure that rows
        // within a partition are sorted by their out-degree in decreasing order, we can compute
        // a tight uppper bound for the maximum number of pushes per warp/block and use shared
        // memory buffer to reduce the number of atomicAdd operations.
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        auto buffer_idx = atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                    static_cast<unsigned long long int>(1));
        *(buffer_key_output_first + buffer_idx) = col;
        *(buffer_payload_output_first + buffer_idx) =
          remove_first_thrust_tuple_element<decltype(e_op_result)>()(e_op_result);
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <typename BufferKeyOutputIterator, typename BufferPayloadOutputIterator, typename ReduceOp>
size_t reduce_buffer_elements(raft::handle_t const& handle,
                              BufferKeyOutputIterator buffer_key_output_first,
                              BufferPayloadOutputIterator buffer_payload_output_first,
                              size_t num_buffer_elements,
                              ReduceOp reduce_op)
{
  thrust::sort_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      buffer_key_output_first,
                      buffer_key_output_first + num_buffer_elements,
                      buffer_payload_output_first);

  if (std::is_same<ReduceOp, reduce_op::any<typename ReduceOp::type>>::value) {
    // FIXME: if ReducOp is any, we may have a cheaper alternative than sort & uique (i.e. discard
    // non-first elements)
    auto it = thrust::unique_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first);
    return static_cast<size_t>(thrust::distance(buffer_key_output_first, thrust::get<0>(it)));
  } else {
    using key_t     = typename std::iterator_traits<BufferKeyOutputIterator>::value_type;
    using payload_t = typename std::iterator_traits<BufferPayloadOutputIterator>::value_type;
    // FIXME: better avoid temporary buffer or at least limit the maximum buffer size (if we adopt
    // CUDA cooperative group https://devblogs.nvidia.com/cooperative-groups and global sync(), we
    // can use aggregate shared memory as a temporary buffer, or we can limit the buffer size, and
    // split one thrust::reduce_by_key call to multiple thrust::reduce_by_key calls if the
    // temporary buffer size exceeds the maximum buffer size (may be definied as percentage of the
    // system HBM size or a function of the maximum number of threads in the system))
    // FIXME: actually, we can find how many unique keys are here by now.
    // FIXME: if GraphViewType::is_multi_gpu is true, this should be executed on the GPU holding the
    // vertex unless reduce_op is a pure function.
    rmm::device_vector<key_t> keys(num_buffer_elements);
    rmm::device_vector<payload_t> values(num_buffer_elements);
    auto it = thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first,
                                    keys.begin(),
                                    values.begin(),
                                    thrust::equal_to<key_t>(),
                                    reduce_op);
    auto num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(keys.begin(), thrust::get<0>(it)));
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 keys.begin(),
                 keys.begin() + num_reduced_buffer_elements,
                 buffer_key_output_first);
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 values.begin(),
                 values.begin() + num_reduced_buffer_elements,
                 buffer_payload_output_first);
    CUDA_TRY(cudaStreamSynchronize(
      handle.get_stream()));  // this is necessary as kyes & values will become out-of-scope once
                              // this function returns
    return num_reduced_buffer_elements;
  }
}

template <size_t num_buckets,
          typename GraphViewType,
          typename BufferKeyInputIterator,
          typename BufferPayloadInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename vertex_t,
          typename VertexOp>
__global__ void update_frontier_and_vertex_output_values(
  vertex_partition_device_t<GraphViewType> vertex_partition,
  BufferKeyInputIterator buffer_key_input_first,
  BufferPayloadInputIterator buffer_payload_input_first,
  size_t num_buffer_elements,
  VertexValueInputIterator vertex_value_input_first,
  VertexValueOutputIterator vertex_value_output_first,
  vertex_t** bucket_ptrs,
  size_t* bucket_sizes_ptr,
  size_t invalid_bucket_idx,
  vertex_t invalid_vertex,
  VertexOp v_op)
{
  static_assert(std::is_same<typename std::iterator_traits<BufferKeyInputIterator>::value_type,
                             vertex_t>::value);
  auto const tid   = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx       = tid;
  size_t block_idx = blockIdx.x;
  // FIXME: it might be more performant to process more than one element per thread
  auto num_blocks = (num_buffer_elements + blockDim.x - 1) / blockDim.x;

  using BlockScan =
    cub::BlockScan<size_t, detail::update_frontier_v_push_if_out_nbr_update_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ size_t bucket_block_start_offsets[num_buckets];

  size_t bucket_block_local_offsets[num_buckets];
  size_t bucket_block_aggregate_sizes[num_buckets];

  while (block_idx < num_blocks) {
    for (size_t i = 0; i < num_buckets; ++i) { bucket_block_local_offsets[i] = 0; }

    size_t selected_bucket_idx{invalid_bucket_idx};
    vertex_t key{invalid_vertex};

    if (idx < num_buffer_elements) {
      key                 = *(buffer_key_input_first + idx);
      auto key_offset     = vertex_partition.get_local_vertex_offset_from_vertex_nocheck(key);
      auto v_val          = *(vertex_value_input_first + key_offset);
      auto payload        = *(buffer_payload_input_first + idx);
      auto v_op_result    = v_op(v_val, payload);
      selected_bucket_idx = thrust::get<0>(v_op_result);
      if (selected_bucket_idx != invalid_bucket_idx) {
        *(vertex_value_output_first + key_offset) =
          remove_first_thrust_tuple_element<decltype(v_op_result)>()(v_op_result);
        bucket_block_local_offsets[selected_bucket_idx] = 1;
      }
    }

    for (size_t i = 0; i < num_buckets; ++i) {
      BlockScan(temp_storage)
        .ExclusiveSum(bucket_block_local_offsets[i],
                      bucket_block_local_offsets[i],
                      bucket_block_aggregate_sizes[i]);
    }

    if (threadIdx.x == 0) {
      for (size_t i = 0; i < num_buckets; ++i) {
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        bucket_block_start_offsets[i] =
          atomicAdd(reinterpret_cast<unsigned long long int*>(bucket_sizes_ptr + i),
                    static_cast<unsigned long long int>(bucket_block_aggregate_sizes[i]));
      }
    }

    __syncthreads();

    // FIXME: better use shared memory buffer to aggreaget global memory writes
    if (selected_bucket_idx != invalid_bucket_idx) {
      bucket_ptrs[selected_bucket_idx][bucket_block_start_offsets[selected_bucket_idx] +
                                       bucket_block_local_offsets[selected_bucket_idx]] = key;
    }

    idx += gridDim.x * blockDim.x;
    block_idx += gridDim.x;
  }
}

}  // namespace detail

/**
 * @brief Update vertex frontier and vertex property values iterating over the outgoing edges.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexIterator Type of the iterator for vertex identifiers.
 * @tparam AdjMatrixRowValueInputIterator Type of the iterator for graph adjacency matrix row
 * input properties.
 * @tparam AdjMatrixColValueInputIterator Type of the iterator for graph adjacency matrix column
 * input properties.
 * @tparam EdgeOp Type of the quaternary (or quinary) edge operator.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam VertexValueOutputIterator Type of the iterator for vertex property variables.
 * @tparam VertexFrontierType Type of the vertex frontier class which abstracts vertex frontier
 * managements.
 * @tparam VertexOp Type of the binary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_first Iterator pointing to the first (inclusive) vertex in the current frontier. v
 * in [vertex_first, vertex_last) should be distinct (and should belong to this process in
 * multi-GPU), otherwise undefined behavior
 * @param vertex_last Iterator pointing to the last (exclusive) vertex in the current frontier.
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
 * get_number_of_local_adj_matrix_partition_cols())) and returns a value to reduced by the @p
 * reduce_op.
 * @param reduce_op Binary operator takes two input arguments and reduce the two variables to one.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param vertex_value_output_first Iterator pointing to the vertex property variables for the first
 * (inclusive) vertex (assigned to tihs process in multi-GPU). `vertex_value_output_last`
 * (exclusive) is deduced as @p vertex_value_output_first + @p
 * graph_view.get_number_of_local_vertices().
 * @param vertex_frontier vertex frontier class object for vertex frontier managements. This object
 * includes multiple bucket objects.
 * @param v_op Binary operator takes *(@p vertex_value_input_first + i) (where i is [0, @p
 * graph_view.get_number_of_local_vertices())) and reduced value of the @p e_op outputs for
 * this vertex and returns the target bucket index (for frontier update) and new verrtex property
 * values (to update *(@p vertex_value_output_first + i)).
 */
template <typename GraphViewType,
          typename VertexIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename ReduceOp,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexFrontierType,
          typename VertexOp>
void update_frontier_v_push_if_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator vertex_first,
  VertexIterator vertex_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  EdgeOp e_op,
  ReduceOp reduce_op,
  VertexValueInputIterator vertex_value_input_first,
  VertexValueOutputIterator vertex_value_output_first,
  VertexFrontierType& vertex_frontier,
  VertexOp v_op)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  // 1. fill the buffer

  vertex_frontier.set_buffer_idx_value(0);

  auto loop_count = size_t{1};
  if (GraphViewType::is_multi_gpu) {
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    loop_count               = graph_view.is_hypergraph_partitioned()
                   ? graph_view.get_number_of_local_adj_matrix_partitions()
                   : static_cast<size_t>(row_comm_size);
  }

  for (size_t i = 0; i < loop_count; ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(
      graph_view, (GraphViewType::is_multi_gpu && !graph_view.is_hypergraph_partitioned()) ? 0 : i);

    rmm::device_uvector<vertex_t> frontier_rows(
      0, handle.get_stream());  // relevant only if GraphViewType::is_multi_gpu is true

    size_t frontier_size{};
    if (GraphViewType::is_multi_gpu) {
      auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
      auto const row_comm_rank = row_comm.get_rank();
      auto const row_comm_size = row_comm.get_size();
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      auto sub_comm_rank = graph_view.is_hypergraph_partitioned() ? col_comm_rank : row_comm_rank;
      frontier_size      = host_scalar_bcast(
        graph_view.is_hypergraph_partitioned() ? col_comm : row_comm,
        (static_cast<size_t>(sub_comm_rank) == i) ? thrust::distance(vertex_first, vertex_last)
                                                  : size_t{0},
        i,
        handle.get_stream());
      if (static_cast<size_t>(sub_comm_rank) != i) {
        frontier_rows.resize(frontier_size, handle.get_stream());
      }
      device_bcast(graph_view.is_hypergraph_partitioned() ? col_comm : row_comm,
                   vertex_first,
                   frontier_rows.begin(),
                   frontier_size,
                   i,
                   handle.get_stream());
    } else {
      frontier_size = thrust::distance(vertex_first, vertex_last);
    }

    edge_t max_pushes =
      frontier_size > 0
        ? frontier_rows.size() > 0
            ? thrust::transform_reduce(
                rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                frontier_rows.begin(),
                frontier_rows.end(),
                [matrix_partition] __device__(auto row) {
                  auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
                  return matrix_partition.get_local_degree(row_offset);
                },
                edge_t{0},
                thrust::plus<edge_t>())
            : thrust::transform_reduce(
                rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                vertex_first,
                vertex_last,
                [matrix_partition] __device__(auto row) {
                  auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
                  return matrix_partition.get_local_degree(row_offset);
                },
                edge_t{0},
                thrust::plus<edge_t>())
        : edge_t{0};

    // FIXME: This is highly pessimistic for single GPU (and multi-GPU as well if we maintain
    // additional per column data for filtering in e_op). If we can pause & resume execution if
    // buffer needs to be increased (and if we reserve address space to avoid expensive
    // reallocation;
    // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/), we can
    // start with a smaller buffer size (especially when the frontier size is large).
    // for special cases when we can assure that there is no more than one push per destination
    // (e.g. if cugraph::experimental::reduce_op::any is used), we can limit the buffer size to
    // std::min(max_pushes, matrix_partition.get_minor_size()).
    // For Volta+, we can limit the buffer size to std::min(max_pushes,
    // matrix_partition.get_minor_size()) if the reduction operation is a pure function if we use
    // locking.
    // FIXME: if i != 0, this will require costly reallocation if we don't use the new CUDA feature
    // to reserve address space.
    vertex_frontier.resize_buffer(vertex_frontier.get_buffer_idx_value() + max_pushes);
    auto buffer_first         = vertex_frontier.buffer_begin();
    auto buffer_key_first     = std::get<0>(buffer_first);
    auto buffer_payload_first = std::get<1>(buffer_first);

    auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                    ? vertex_t{0}
                                    : matrix_partition.get_major_value_start_offset();

    // FIXME: This is highly inefficeint for graphs with high-degree vertices. If we renumber
    // vertices to insure that rows within a partition are sorted by their out-degree in decreasing
    // order, we will apply this kernel only to low out-degree vertices.
    if (frontier_size > 0) {
      raft::grid_1d_thread_t for_all_low_degree_grid(
        frontier_size,
        detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
        handle.get_device_properties().maxGridSize[0]);

      if (frontier_rows.size() > 0) {
        detail::for_all_frontier_row_for_all_nbr_low_degree<<<for_all_low_degree_grid.num_blocks,
                                                              for_all_low_degree_grid.block_size,
                                                              0,
                                                              handle.get_stream()>>>(
          matrix_partition,
          frontier_rows.begin(),
          frontier_rows.end(),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          buffer_key_first,
          buffer_payload_first,
          vertex_frontier.get_buffer_idx_ptr(),
          e_op);
      } else {
        detail::for_all_frontier_row_for_all_nbr_low_degree<<<for_all_low_degree_grid.num_blocks,
                                                              for_all_low_degree_grid.block_size,
                                                              0,
                                                              handle.get_stream()>>>(
          matrix_partition,
          vertex_first,
          vertex_last,
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          buffer_key_first,
          buffer_payload_first,
          vertex_frontier.get_buffer_idx_ptr(),
          e_op);
      }
    }
  }

  // 2. reduce the buffer

  auto num_buffer_offset = edge_t{0};

  auto buffer_first         = vertex_frontier.buffer_begin();
  auto buffer_key_first     = std::get<0>(buffer_first) + num_buffer_offset;
  auto buffer_payload_first = std::get<1>(buffer_first) + num_buffer_offset;

  auto num_buffer_elements = detail::reduce_buffer_elements(handle,
                                                            buffer_key_first,
                                                            buffer_payload_first,
                                                            vertex_frontier.get_buffer_idx_value(),
                                                            reduce_op);

  if (GraphViewType::is_multi_gpu) {
    auto& comm               = handle.get_comms();
    auto const comm_rank     = comm.get_rank();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();

    std::vector<vertex_t> h_vertex_lasts(graph_view.is_hypergraph_partitioned() ? row_comm_size
                                                                                : col_comm_size);
    for (size_t i = 0; i < h_vertex_lasts.size(); ++i) {
      h_vertex_lasts[i] = graph_view.get_vertex_partition_last(
        graph_view.is_hypergraph_partitioned() ? col_comm_rank * row_comm_size + i
                                               : row_comm_rank * col_comm_size + i);
    }

    rmm::device_uvector<vertex_t> d_vertex_lasts(h_vertex_lasts.size(), handle.get_stream());
    raft::update_device(
      d_vertex_lasts.data(), h_vertex_lasts.data(), h_vertex_lasts.size(), handle.get_stream());
    rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(d_vertex_lasts.size(),
                                                            handle.get_stream());
    thrust::lower_bound(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        buffer_key_first,
                        buffer_key_first + num_buffer_elements,
                        d_vertex_lasts.begin(),
                        d_vertex_lasts.end(),
                        d_tx_buffer_last_boundaries.begin());
    std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
    raft::update_host(h_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.size(),
                      handle.get_stream());
    CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
    std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
    std::adjacent_difference(
      h_tx_buffer_last_boundaries.begin(), h_tx_buffer_last_boundaries.end(), tx_counts.begin());

    std::vector<size_t> rx_counts(graph_view.is_hypergraph_partitioned() ? row_comm_size
                                                                         : col_comm_size);
    std::vector<raft::comms::request_t> count_requests(tx_counts.size() + rx_counts.size());
    size_t tx_self_i = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < tx_counts.size(); ++i) {
      auto comm_dst_rank = graph_view.is_hypergraph_partitioned()
                             ? col_comm_rank * row_comm_size + static_cast<int>(i)
                             : row_comm_rank * col_comm_size + static_cast<int>(i);
      if (comm_dst_rank == comm_rank) {
        tx_self_i = i;
        // FIXME: better define request_null (similar to MPI_REQUEST_NULL) under raft::comms
        count_requests[i] = std::numeric_limits<raft::comms::request_t>::max();
      } else {
        comm.isend(&tx_counts[i], 1, comm_dst_rank, 0 /* tag */, count_requests.data() + i);
      }
    }
    for (size_t i = 0; i < rx_counts.size(); ++i) {
      auto comm_src_rank = graph_view.is_hypergraph_partitioned()
                             ? col_comm_rank * row_comm_size + static_cast<int>(i)
                             : static_cast<int>(i) * row_comm_size + comm_rank / col_comm_size;
      if (comm_src_rank == comm_rank) {
        assert(tx_self_i != std::numeric_limits<size_t>::max());
        rx_counts[i] = tx_counts[tx_self_i];
        // FIXME: better define request_null (similar to MPI_REQUEST_NULL) under raft::comms
        count_requests[tx_counts.size() + i] = std::numeric_limits<raft::comms::request_t>::max();
      } else {
        comm.irecv(&rx_counts[i],
                   1,
                   comm_src_rank,
                   0 /* tag */,
                   count_requests.data() + tx_counts.size() + i);
      }
    }
    // FIXME: better define request_null (similar to MPI_REQUEST_NULL) under raft::comms, if
    // raft::comms::wait immediately returns on seeing request_null, this remove is unnecessary
    count_requests.erase(std::remove(count_requests.begin(),
                                     count_requests.end(),
                                     std::numeric_limits<raft::comms::request_t>::max()),
                         count_requests.end());
    comm.waitall(count_requests.size(), count_requests.data());

    std::vector<size_t> tx_offsets(tx_counts.size() + 1, edge_t{0});
    std::partial_sum(tx_counts.begin(), tx_counts.end(), tx_offsets.begin() + 1);
    std::vector<size_t> rx_offsets(rx_counts.size() + 1, edge_t{0});
    std::partial_sum(rx_counts.begin(), rx_counts.end(), rx_offsets.begin() + 1);

    // FIXME: this will require costly reallocation if we don't use the new CUDA feature to reserve
    // address space.
    // FIXME: std::max(actual size, 1) as ncclRecv currently hangs if recvuff is nullptr even if
    // count is 0
    vertex_frontier.resize_buffer(std::max(num_buffer_elements + rx_offsets.back(), size_t(1)));

    auto buffer_first         = vertex_frontier.buffer_begin();
    auto buffer_key_first     = std::get<0>(buffer_first) + num_buffer_offset;
    auto buffer_payload_first = std::get<1>(buffer_first) + num_buffer_offset;

    std::vector<int> tx_dst_ranks(tx_counts.size());
    std::vector<int> rx_src_ranks(rx_counts.size());
    for (size_t i = 0; i < tx_dst_ranks.size(); ++i) {
      tx_dst_ranks[i] = graph_view.is_hypergraph_partitioned()
                          ? col_comm_rank * row_comm_size + static_cast<int>(i)
                          : row_comm_rank * col_comm_size + static_cast<int>(i);
    }
    for (size_t i = 0; i < rx_src_ranks.size(); ++i) {
      rx_src_ranks[i] = graph_view.is_hypergraph_partitioned()
                          ? col_comm_rank * row_comm_size + static_cast<int>(i)
                          : static_cast<int>(i) * row_comm_size + comm_rank / col_comm_size;
    }

    device_multicast_sendrecv<decltype(buffer_key_first), decltype(buffer_key_first)>(
      comm,
      buffer_key_first,
      tx_counts,
      tx_offsets,
      tx_dst_ranks,
      buffer_key_first + num_buffer_elements,
      rx_counts,
      rx_offsets,
      rx_src_ranks,
      handle.get_stream());
    device_multicast_sendrecv<decltype(buffer_payload_first), decltype(buffer_payload_first)>(
      comm,
      buffer_payload_first,
      tx_counts,
      tx_offsets,
      tx_dst_ranks,
      buffer_payload_first + num_buffer_elements,
      rx_counts,
      rx_offsets,
      rx_src_ranks,
      handle.get_stream());

    // FIXME: this does not exploit the fact that each segment is sorted. Lost performance
    // optimization opportunities.
    // FIXME: we can use [vertex_frontier.buffer_begin(), vertex_frontier.buffer_begin() +
    // num_buffer_elements) as temporary buffer inside reduce_buffer_elements().
    num_buffer_offset   = num_buffer_elements;
    num_buffer_elements = detail::reduce_buffer_elements(handle,
                                                         buffer_key_first + num_buffer_elements,
                                                         buffer_payload_first + num_buffer_elements,
                                                         rx_offsets.back(),
                                                         reduce_op);
  }

  // 3. update vertex properties

  if (num_buffer_elements > 0) {
    auto buffer_first         = vertex_frontier.buffer_begin();
    auto buffer_key_first     = std::get<0>(buffer_first) + num_buffer_offset;
    auto buffer_payload_first = std::get<1>(buffer_first) + num_buffer_offset;

    raft::grid_1d_thread_t update_grid(num_buffer_elements,
                                       detail::update_frontier_v_push_if_out_nbr_update_block_size,
                                       handle.get_device_properties().maxGridSize[0]);

    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    vertex_partition_device_t<GraphViewType> vertex_partition(graph_view);

    auto bucket_and_bucket_size_device_ptrs =
      vertex_frontier.get_bucket_and_bucket_size_device_pointers();
    detail::update_frontier_and_vertex_output_values<VertexFrontierType::kNumBuckets>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
        vertex_partition,
        buffer_key_first,
        buffer_payload_first,
        num_buffer_elements,
        vertex_value_input_first,
        vertex_value_output_first,
        std::get<0>(bucket_and_bucket_size_device_ptrs).get(),
        std::get<1>(bucket_and_bucket_size_device_ptrs).get(),
        VertexFrontierType::kInvalidBucketIdx,
        invalid_vertex,
        v_op);

    auto bucket_sizes_device_ptr = std::get<1>(bucket_and_bucket_size_device_ptrs);
    thrust::host_vector<size_t> bucket_sizes(
      bucket_sizes_device_ptr, bucket_sizes_device_ptr + VertexFrontierType::kNumBuckets);
    for (size_t i = 0; i < VertexFrontierType::kNumBuckets; ++i) {
      vertex_frontier.get_bucket(i).set_size(bucket_sizes[i]);
    }
  }
}

/*

FIXME:

iterating over lower triangular (or upper triangular) : triangle counting
LRB might be necessary if the cost of processing an edge (i, j) is a function of degree(i) and
degree(j) : triangle counting
push-pull switching support (e.g. DOBFS), in this case, we need both
CSR & CSC (trade-off execution time vs memory requirement, unless graph is symmetric)
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree ==
out-degree) For BFS, sending a bit vector (for the entire set of dest vertices per partitoin may
work better we can use thrust::set_intersection for triangle counting think about adding thrust
wrappers for reduction functions. Can I pass nullptr for dummy
instead of thrust::make_counting_iterator(0)?
*/

}  // namespace experimental
}  // namespace cugraph
