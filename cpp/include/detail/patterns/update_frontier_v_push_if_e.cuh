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
#include <detail/utilities/thrust_tuple_utils.cuh>
#include <utilities/error.hpp>

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

namespace {

// FIXME: block size requires tuning
int32_t constexpr update_frontier_v_push_if_e_for_all_low_out_degree_block_size = 128;
int32_t constexpr update_frontier_v_push_if_e_update_block_size                 = 128;

template <typename GraphType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_low_out_degree(
  GraphType graph_device_view,
  RowIterator row_first,
  RowIterator row_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  BufferKeyOutputIterator buffer_key_output_first,
  BufferPayloadOutputIterator buffer_payload_output_first,
  size_t* buffer_idx_ptr,
  EdgeOp e_op)
{
  using vertex_t = typename GraphType::vertex_type;
  using weight_t = typename GraphType::weight_type;

  static_assert(!GraphType::is_adj_matrix_transposed, "GraphType should support the push model.");

  auto num_rows  = static_cast<size_t>(thrust::distance(row_first, row_last));
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx     = tid;

  while (idx < num_rows) {
    auto row        = *(row_first + idx);
    auto row_offset = graph_device_view.get_adj_matrix_local_row_offset_from_row_nocheck(row);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    vertex_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = graph_device_view.get_local_edges(row);
    for (vertex_t i = 0; i < local_out_degree; ++i) {
      auto dst_vid    = indices[i];
      auto weight     = weights != nullptr ? weights[i] : 1.0;
      auto col_offset = graph_device_view.get_adj_matrix_local_col_offset_from_col_nocheck(dst_vid);
      auto e_op_result =
        cugraph::experimental::detail::evaluate_edge_op<GraphType,
                                                        EdgeOp,
                                                        AdjMatrixRowValueInputIterator,
                                                        AdjMatrixColValueInputIterator>()
          .compute(*(adj_matrix_row_value_input_first + row_offset),
                   *(adj_matrix_col_value_input_first + col_offset),
                   weight,
                   e_op);
      if (thrust::get<0>(e_op_result) == true) {
        // FIXME: This atomicInc serializes execution. If we renumber vertices to insure that rows
        // within a partition are sorted by their out-degree in decreasing order, we can compute
        // a tight uppper bound for the maximum number of pushes per warp/block and use shared
        // memory buffer to reduce the number of atomicAdd operations.
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        auto buffer_idx = atomicAdd(reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
                                    static_cast<unsigned long long int>(1));
        *(buffer_key_output_first + buffer_idx) = col_offset;
        *(buffer_payload_output_first + buffer_idx) =
          cugraph::experimental::detail::remove_first_thrust_tuple_element<decltype(e_op_result)>()(
            e_op_result);
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

template <typename HandleType,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename ReduceOp>
size_t reduce_buffer_elements(HandleType& handle,
                              BufferKeyOutputIterator buffer_key_output_first,
                              BufferPayloadOutputIterator buffer_payload_output_first,
                              size_t num_buffer_elements,
                              ReduceOp reduce_op)
{
  thrust::sort_by_key(thrust::cuda::par.on(handle.get_stream()),
                      buffer_key_output_first,
                      buffer_key_output_first + num_buffer_elements,
                      buffer_payload_output_first);

  if (std::is_same<ReduceOp,
                   cugraph::experimental::detail::reduce_op::any<typename ReduceOp::type>>::value) {
    // FIXME: if ReducOp is any, we may have a cheaper alternative than sort & uique (i.e. discard
    // non-first elements)
    auto it = thrust::unique_by_key(thrust::cuda::par.on(handle.get_stream()),
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
    auto it = thrust::reduce_by_key(thrust::cuda::par.on(handle.get_stream()),
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first,
                                    keys.begin(),
                                    values.begin(),
                                    thrust::equal_to<key_t>(),
                                    reduce_op);
    auto num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(keys.begin(), thrust::get<0>(it)));
    thrust::copy(thrust::cuda::par.on(handle.get_stream()),
                 keys.begin(),
                 keys.begin() + num_reduced_buffer_elements,
                 buffer_key_output_first);
    thrust::copy(thrust::cuda::par.on(handle.get_stream()),
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

  using BlockScan = cub::BlockScan<size_t, update_frontier_v_push_if_e_update_block_size>;
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
          cugraph::experimental::detail::remove_first_thrust_tuple_element<decltype(v_op_result)>()(
            v_op_result);
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

}  // namespace

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType,
          typename GraphType,
          typename RowIterator,
          typename VertexValueInputIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename VertexValueOutputIterator,
          typename AdjMatrixRowValueOutputIterator,
          typename AdjMatrixColValueOutputIterator,
          typename RowFrontierType,
          typename EdgeOp,
          typename ReduceOp,
          typename VertexOp>
void update_frontier_v_push_if_e(HandleType& handle,
                                 GraphType const& graph_device_view,
                                 RowIterator row_first,
                                 RowIterator row_last,
                                 VertexValueInputIterator vertex_value_input_first,
                                 AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
                                 AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
                                 VertexValueOutputIterator vertex_value_output_first,
                                 AdjMatrixRowValueOutputIterator adj_matrix_row_value_output_first,
                                 AdjMatrixColValueOutputIterator adj_matrix_col_value_output_first,
                                 RowFrontierType& row_frontier,
                                 EdgeOp e_op,
                                 ReduceOp reduce_op,
                                 VertexOp v_op)
{
  static_assert(!GraphType::is_adj_matrix_transposed, "GraphType should support the push model.");

  using vertex_t          = typename GraphType::vertex_type;
  using reduce_op_input_t = typename ReduceOp::type;

  auto max_pushes = thrust::transform_reduce(
    thrust::cuda::par.on(handle.get_stream()),
    row_first,
    row_last,
    [graph_device_view] __device__(auto row) {
      return graph_device_view.get_local_out_degree_nocheck(row);
    },
    static_cast<size_t>(0),
    thrust::plus<size_t>());
  // FIXME: This is highly pessimistic for single GPU (and OPG as well if we maintain additional
  // per column data for filtering in e_op). If we can pause & resume execution if buffer needs to
  // be increased (and if we reserve address space to avoid expensive reallocation;
  // https://devblogs.nvidia.com/introducing-low-level-gpu-virtual-memory-management/), we can
  // start with a smaller buffer size (especially when the frontier size is large).
  row_frontier.resize_buffer(max_pushes);
  row_frontier.set_buffer_idx_value(0);
  auto buffer_first         = row_frontier.buffer_begin();
  auto buffer_key_first     = std::get<0>(buffer_first);
  auto buffer_payload_first = std::get<1>(buffer_first);

  grid_1d_thread_t for_all_low_out_degree_grid(
    thrust::distance(row_first, row_last),
    update_frontier_v_push_if_e_for_all_low_out_degree_block_size,
    get_max_num_blocks_1D());

  // FIXME: This is highly inefficeint for graphs with high-degree vertices. If we renumber
  // vertices to insure that rows within a partition are sorted by their out-degree in decreasing
  // order, we will apply this kernel only to low out-degree vertices.
  for_all_frontier_row_for_all_nbr_low_out_degree<<<for_all_low_out_degree_grid.num_blocks,
                                                    for_all_low_out_degree_grid.block_size,
                                                    0,
                                                    handle.get_stream()>>>(
    graph_device_view,
    row_first,
    row_last,
    adj_matrix_row_value_input_first,
    adj_matrix_col_value_input_first,
    buffer_key_first,
    buffer_payload_first,
    row_frontier.get_buffer_idx_ptr(),
    e_op);

  auto num_buffer_elements = reduce_buffer_elements(
    handle, buffer_key_first, buffer_payload_first, row_frontier.get_buffer_idx_value(), reduce_op);

  if (GraphType::is_opg) {
    // need to exchange buffer elements (and may reduce again)
    CUGRAPH_FAIL("unimplemented.");
  }

  grid_1d_thread_t update_grid(thrust::distance(row_first, row_last),
                               update_frontier_v_push_if_e_update_block_size,
                               get_max_num_blocks_1D());

  auto constexpr invalid_vertex = invalid_vertex_id<vertex_t>::value;

  auto bucket_and_bucket_size_device_ptrs =
    row_frontier.get_bucket_and_bucket_size_device_pointers();
  update_frontier_and_vertex_output_values<RowFrontierType::kNumBuckets>
    <<<update_grid.num_blocks, update_grid.block_size, 0, handle.get_stream()>>>(
      buffer_key_first,
      buffer_payload_first,
      num_buffer_elements,
      vertex_value_input_first,
      vertex_value_output_first,
      std::get<0>(bucket_and_bucket_size_device_ptrs).get(),
      std::get<1>(bucket_and_bucket_size_device_ptrs).get(),
      RowFrontierType::kInvalidBucketIdx,
      invalid_vertex,
      v_op);

  auto bucket_sizes_device_ptr = std::get<1>(bucket_and_bucket_size_device_ptrs);
  thrust::host_vector<size_t> bucket_sizes(bucket_sizes_device_ptr,
                                           bucket_sizes_device_ptr + RowFrontierType::kNumBuckets);
  for (size_t i = 0; i < RowFrontierType::kNumBuckets; ++i) {
    row_frontier.get_bucket(i).set_size(bucket_sizes[i]);
  }

  if (!std::is_same<AdjMatrixRowValueOutputIterator, thrust::discard_iterator<>>::value) {
    CUGRAPH_FAIL("unimplemented.");
  }

  if (!std::is_same<AdjMatrixRowValueOutputIterator, thrust::discard_iterator<>>::value) {
    CUGRAPH_FAIL("unimplemented.");
  }

  if (GraphType::is_opg) {
    // need to merge row_frontier
    CUGRAPH_FAIL("unimplemented.");
  }
}

/*
iterating over lower triangular (or upper triangular) : triangle counting
LRB might be necessary if the cost of processing an edge (i, j) is a function of degree(i) and
degree(j) : triangle counting
push-pull switching support (e.g. DOBFS), in this case, we need both
CSR & CSC (trade-off execution time vs memory requirement, unless graph is symmetric)
should I take multi-GPU support as a template argument?
Add bool expensive_check = false ?
cugraph::count_if as a multi-GPU wrapper of thrust::count_if? (for expensive check)
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree ==
out-degree) For BFS, sending a bit vector (for the entire set of dest vertices per partitoin may
work better we can use thrust::set_intersection for triangle counting think about adding thrust
wrappers for reduction functions. thrust::(); if (opg) { allreduce }; can be cugraph::(), and be
more consistant with other APIs that hide communication inside if opg. Can I pass nullptr for dummy
instead of thrust::make_counting_iterator(0)?
*/

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
