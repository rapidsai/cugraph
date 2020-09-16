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
#include <utilities/cuda.cuh>
#include <utilities/error.hpp>
#include <utilities/thrust_tuple_utils.cuh>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
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
int32_t constexpr update_frontier_v_push_if_out_nbr_for_all_low_out_degree_block_size = 128;
int32_t constexpr update_frontier_v_push_if_out_nbr_update_block_size                 = 128;

template <typename GraphViewType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_low_out_degree(
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
        *(buffer_key_output_first + buffer_idx) = col_offset;
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
    return num_reduced_buffer_elements;
  }
}

template <size_t num_buckets,
          typename BufferKeyInputIterator,
          typename BufferPayloadInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename vertex_t,
          typename VertexOp>
__global__ void update_frontier_and_vertex_output_values(
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
      auto v_val          = *(vertex_value_input_first + key);
      auto payload        = *(buffer_payload_input_first + idx);
      auto v_op_result    = v_op(v_val, payload);
      selected_bucket_idx = thrust::get<0>(v_op_result);
      if (selected_bucket_idx != invalid_bucket_idx) {
        *(vertex_value_output_first + key) =
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
 * @tparam EdgeOp Type of the binary (or ternary) edge operator.
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
 * @p graph_view.get_number_of_adj_matrix_local_rows().
 * @param adj_matrix_col_value_input_first Iterator pointing to the adjacency matrix column input
 * properties for the first (inclusive) column (assigned to this process in multi-GPU).
 * `adj_matrix_col_value_output_last` (exclusive) is deduced as @p adj_matrix_col_value_output_first
 * + @p graph_view.get_number_of_adj_matrix_local_cols().
 * @param e_op Binary (or ternary) operator takes *(@p adj_matrix_row_value_input_first + i), *(@p
 * adj_matrix_col_value_input_first + j), (and optionally edge weight) (where i and j are row and
 * column indices, respectively) and returns a value to reduced by the @p reduce_op.
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

  using vertex_t          = typename GraphViewType::vertex_type;
  using edge_t            = typename GraphViewType::edge_type;
  using reduce_op_input_t = typename ReduceOp::type;

  // FIXME: Better make this a memeber variable of VertexFrontier to reduce # of memory allocations
  thrust::device_vector<vertex_t>
    frontier_rows{};  // relevant only if GraphViewType::is_multi_gpu is true
  std::vector<vertex_t> frontier_adj_matrix_partition_offsets(
    graph_view.get_number_of_local_adj_matrix_partitions() + 1, 0);

  if (GraphViewType::is_multi_gpu) {
    // need to merge row_frontier and update frontier_offsets;
    CUGRAPH_FAIL("unimplemented.");
  } else {
    frontier_adj_matrix_partition_offsets[1] = thrust::distance(vertex_first, vertex_last);
  }

  edge_t max_pushes{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    max_pushes += thrust::transform_reduce(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      frontier_rows.begin() + frontier_adj_matrix_partition_offsets[i],
      frontier_rows.begin() + frontier_adj_matrix_partition_offsets[i + 1],
      [matrix_partition] __device__(auto row) {
        auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
        return matrix_partition.get_local_degree(row_offset);
      },
      size_t{0},
      thrust::plus<size_t>());
  }

  // FIXME: This is highly pessimistic for single GPU (and multi-GPU as well if we maintain
  // additional per column data for filtering in e_op). If we can pause & resume execution if
  // buffer needs to be increased (and if we reserve address space to avoid expensive
  // reallocation;
  // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/), we can
  // start with a smaller buffer size (especially when the frontier size is large).
  vertex_frontier.resize_buffer(max_pushes);
  vertex_frontier.set_buffer_idx_value(0);
  auto buffer_first         = vertex_frontier.buffer_begin();
  auto buffer_key_first     = std::get<0>(buffer_first);
  auto buffer_payload_first = std::get<1>(buffer_first);

  vertex_t row_value_input_offset{0};
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    grid_1d_thread_t for_all_low_out_degree_grid(
      frontier_adj_matrix_partition_offsets[i + 1] - frontier_adj_matrix_partition_offsets[i],
      detail::update_frontier_v_push_if_out_nbr_for_all_low_out_degree_block_size,
      get_max_num_blocks_1D());

    // FIXME: This is highly inefficeint for graphs with high-degree vertices. If we renumber
    // vertices to insure that rows within a partition are sorted by their out-degree in
    // decreasing order, we will apply this kernel only to low out-degree vertices.
    detail::
      for_all_frontier_row_for_all_nbr_low_out_degree<<<for_all_low_out_degree_grid.num_blocks,
                                                        for_all_low_out_degree_grid.block_size,
                                                        0,
                                                        handle.get_stream()>>>(
        matrix_partition,
        frontier_rows.begin() + frontier_adj_matrix_partition_offsets[i],
        frontier_rows.begin() + frontier_adj_matrix_partition_offsets[i + 1],
        adj_matrix_row_value_input_first + row_value_input_offset,
        adj_matrix_col_value_input_first,
        buffer_key_first,
        buffer_payload_first,
        vertex_frontier.get_buffer_idx_ptr(),
        e_op);

    row_value_input_offset += matrix_partition.get_major_size();
  }

  auto num_buffer_elements = detail::reduce_buffer_elements(handle,
                                                            buffer_key_first,
                                                            buffer_payload_first,
                                                            vertex_frontier.get_buffer_idx_value(),
                                                            reduce_op);

  if (GraphViewType::is_multi_gpu) {
    // need to exchange buffer elements (and may reduce again)
    CUGRAPH_FAIL("unimplemented.");
  }

  if (num_buffer_elements > 0) {
    grid_1d_thread_t update_grid(num_buffer_elements,
                                 detail::update_frontier_v_push_if_out_nbr_update_block_size,
                                 get_max_num_blocks_1D());

    auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

    auto bucket_and_bucket_size_device_ptrs =
      vertex_frontier.get_bucket_and_bucket_size_device_pointers();
    detail::update_frontier_and_vertex_output_values<VertexFrontierType::kNumBuckets>
      <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
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
is_fully_functional type trait (???) for reduce_op

iterating over lower triangular (or upper triangular) : triangle counting
LRB might be necessary if the cost of processing an edge (i, j) is a function of degree(i) and
degree(j) : triangle counting
push-pull switching support (e.g. DOBFS), in this case, we need both
CSR & CSC (trade-off execution time vs memory requirement, unless graph is symmetric)
should I take multi-GPU support as a template argument?
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree ==
out-degree) For BFS, sending a bit vector (for the entire set of dest vertices per partitoin may
work better we can use thrust::set_intersection for triangle counting think about adding thrust
wrappers for reduction functions. Can I pass nullptr for dummy
instead of thrust::make_counting_iterator(0)?
*/

}  // namespace experimental
}  // namespace cugraph
