/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <partition_manager.hpp>
#include <patterns/edge_op_utils.cuh>
#include <patterns/reduce_op.cuh>
#include <utilities/dataframe_buffer.cuh>
#include <utilities/device_comm.cuh>
#include <utilities/error.hpp>
#include <utilities/host_barrier.hpp>
#include <utilities/host_scalar_comm.cuh>
#include <utilities/shuffle_comm.cuh>
#include <utilities/thrust_tuple_utils.cuh>
#include <vertex_partition_device.cuh>

#include <raft/cudart_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/type_traits/integer_sequence.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {
namespace experimental {

namespace detail {

int32_t constexpr update_frontier_v_push_if_out_nbr_for_all_block_size = 512;

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

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  auto idx       = static_cast<size_t>(tid);

  while (idx < static_cast<size_t>(thrust::distance(row_first, row_last))) {
    vertex_t row    = *(row_first + idx);
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = 0; i < local_out_degree; ++i) {
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
        *(buffer_key_output_first + buffer_idx)     = col;
        *(buffer_payload_output_first + buffer_idx) = thrust::get<1>(e_op_result);
      }
    }
    idx += gridDim.x * blockDim.x;
  }
}

template <typename GraphViewType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_mid_degree(
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

  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(update_frontier_v_push_if_out_nbr_for_all_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto idx           = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(thrust::distance(row_first, row_last))) {
    vertex_t row    = *(row_first + idx);
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = lane_id; i < local_out_degree; i += raft::warp_size()) {
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
        *(buffer_key_output_first + buffer_idx)     = col;
        *(buffer_payload_output_first + buffer_idx) = thrust::get<1>(e_op_result);
      }
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename GraphViewType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename BufferKeyOutputIterator,
          typename BufferPayloadOutputIterator,
          typename EdgeOp>
__global__ void for_all_frontier_row_for_all_nbr_high_degree(
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

  auto idx = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(thrust::distance(row_first, row_last))) {
    vertex_t row    = *(row_first + idx);
    auto row_offset = matrix_partition.get_major_offset_from_major_nocheck(row);
    vertex_t const* indices{nullptr};
    weight_t const* weights{nullptr};
    edge_t local_out_degree{};
    thrust::tie(indices, weights, local_out_degree) = matrix_partition.get_local_edges(row_offset);
    for (edge_t i = threadIdx.x; i < local_out_degree; i += blockDim.x) {
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
        *(buffer_key_output_first + buffer_idx)     = col;
        *(buffer_payload_output_first + buffer_idx) = thrust::get<1>(e_op_result);
      }
    }

    idx += gridDim.x;
  }
}

template <typename BufferKeyOutputIterator, typename BufferPayloadOutputIterator, typename ReduceOp>
size_t sort_and_reduce_buffer_elements(raft::handle_t const& handle,
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
    // FIXME: if GraphViewType::is_multi_gpu is true, this should be executed on the GPU holding
    // the vertex unless reduce_op is a pure function.
    rmm::device_uvector<key_t> keys(num_buffer_elements, handle.get_stream());
    auto value_buffer =
      allocate_dataframe_buffer<payload_t>(num_buffer_elements, handle.get_stream());
    auto it = thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    buffer_key_output_first,
                                    buffer_key_output_first + num_buffer_elements,
                                    buffer_payload_output_first,
                                    keys.begin(),
                                    get_dataframe_buffer_begin<payload_t>(value_buffer),
                                    thrust::equal_to<key_t>(),
                                    reduce_op);
    auto num_reduced_buffer_elements =
      static_cast<size_t>(thrust::distance(keys.begin(), thrust::get<0>(it)));
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 keys.begin(),
                 keys.begin() + num_reduced_buffer_elements,
                 buffer_key_output_first);
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 get_dataframe_buffer_begin<payload_t>(value_buffer),
                 get_dataframe_buffer_begin<payload_t>(value_buffer) + num_reduced_buffer_elements,
                 buffer_payload_output_first);
    return num_reduced_buffer_elements;
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
 * @param vertex_frontier VertexFrontier class object for vertex frontier managements. This object
 * includes multiple bucket objects.
 * @param cur_fontier_bucket_idx Index of the VertexFrontier bucket holding vertices for the current
 * iteration.
 * @param next_frontier_bucket_indices Indices of the VertexFrontier buckets to store new frontier
 * vertices for the next iteration.
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
 * @param v_op Binary operator takes *(@p vertex_value_input_first + i) (where i is [0, @p
 * graph_view.get_number_of_local_vertices())) and reduced value of the @p e_op outputs for
 * this vertex and returns the target bucket index (for frontier update) and new verrtex property
 * values (to update *(@p vertex_value_output_first + i)). The target bucket index should either be
 * VertexFrontier::kInvalidBucketIdx or an index in @p next_frontier_bucket_indices.
 */
template <typename GraphViewType,
          typename VertexFrontierType,
          typename AdjMatrixRowValueInputIterator,
          typename AdjMatrixColValueInputIterator,
          typename EdgeOp,
          typename ReduceOp,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename VertexOp>
void update_frontier_v_push_if_out_nbr(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierType& vertex_frontier,
  size_t cur_frontier_bucket_idx,
  std::vector<size_t> const& next_frontier_bucket_indices,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  EdgeOp e_op,
  ReduceOp reduce_op,
  VertexValueInputIterator vertex_value_input_first,
  VertexValueOutputIterator vertex_value_output_first,
  VertexOp v_op)
{
  static_assert(!GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the push model.");

  using vertex_t  = typename GraphViewType::vertex_type;
  using edge_t    = typename GraphViewType::edge_type;
  using weight_t  = typename GraphViewType::weight_type;
  using payload_t = typename ReduceOp::type;

  auto cur_frontier_vertex_first = vertex_frontier.get_bucket(cur_frontier_bucket_idx).begin();
  auto cur_frontier_vertex_last  = vertex_frontier.get_bucket(cur_frontier_bucket_idx).end();

  // 1. fill the buffer

  if (GraphViewType::is_multi_gpu) {
    auto& comm = handle.get_comms();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of col_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }

  rmm::device_uvector<vertex_t> keys(size_t{0}, handle.get_stream());
  auto payload_buffer = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
  rmm::device_scalar<size_t> buffer_idx(size_t{0}, handle.get_stream());
  for (size_t i = 0; i < graph_view.get_number_of_local_adj_matrix_partitions(); ++i) {
    matrix_partition_device_t<GraphViewType> matrix_partition(graph_view, i);

    rmm::device_uvector<vertex_t> frontier_rows(0, handle.get_stream());
    if (GraphViewType::is_multi_gpu) {
      auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
      auto const col_comm_rank = col_comm.get_rank();

      auto frontier_size =
        host_scalar_bcast(col_comm,
                          (static_cast<size_t>(col_comm_rank) == i)
                            ? thrust::distance(cur_frontier_vertex_first, cur_frontier_vertex_last)
                            : size_t{0} /* dummy */,
                          i,
                          handle.get_stream());
      frontier_rows.resize(frontier_size, handle.get_stream());

      if (static_cast<size_t>(col_comm_rank) == i) {
        thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                     cur_frontier_vertex_first,
                     cur_frontier_vertex_last,
                     frontier_rows.begin());
      }

      device_bcast(col_comm,
                   cur_frontier_vertex_first,
                   frontier_rows.begin(),
                   frontier_size,
                   i,
                   handle.get_stream());
    } else {
      frontier_rows.resize(thrust::distance(cur_frontier_vertex_first, cur_frontier_vertex_last),
                           handle.get_stream());
      thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   cur_frontier_vertex_first,
                   cur_frontier_vertex_last,
                   frontier_rows.begin());
    }

    auto max_pushes = frontier_rows.size() > 0
                        ? thrust::transform_reduce(
                            rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                            frontier_rows.begin(),
                            frontier_rows.end(),
                            [matrix_partition] __device__(auto row) {
                              auto row_offset =
                                matrix_partition.get_major_offset_from_major_nocheck(row);
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
    keys.resize(buffer_idx.value(handle.get_stream()) + max_pushes, handle.get_stream());
    resize_dataframe_buffer<payload_t>(payload_buffer, keys.size(), handle.get_stream());

    auto row_value_input_offset = GraphViewType::is_adj_matrix_transposed
                                    ? vertex_t{0}
                                    : matrix_partition.get_major_value_start_offset();
    auto segment_offsets = graph_view.get_local_adj_matrix_partition_segment_offsets(i);
    if (segment_offsets.size() > 0) {
      static_assert(detail::num_segments_per_vertex_partition == 3);
      std::vector<vertex_t> h_thresholds(detail::num_segments_per_vertex_partition - 1);
      h_thresholds[0] = matrix_partition.get_major_first() + segment_offsets[1];
      h_thresholds[1] = matrix_partition.get_major_first() + segment_offsets[2];
      rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), handle.get_stream());
      raft::update_device(
        d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> d_offsets(d_thresholds.size(), handle.get_stream());
      thrust::lower_bound(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                          frontier_rows.begin(),
                          frontier_rows.end(),
                          d_thresholds.begin(),
                          d_thresholds.end(),
                          d_offsets.begin());
      std::vector<vertex_t> h_offsets(d_offsets.size());
      raft::update_host(h_offsets.data(), d_offsets.data(), d_offsets.size(), handle.get_stream());
      CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
      // FIXME: we may further improve performance by 1) concurrently running kernels on different
      // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
      // segment for very high degree vertices and running segmented reduction
      if (h_offsets[0] > 0) {
        raft::grid_1d_block_t update_grid(
          h_offsets[0],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_row_for_all_nbr_high_degree<<<update_grid.num_blocks,
                                                               update_grid.block_size,
                                                               0,
                                                               handle.get_stream()>>>(
          matrix_partition,
          frontier_rows.begin(),
          frontier_rows.begin() + h_offsets[0],
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          keys.begin(),
          get_dataframe_buffer_begin<payload_t>(payload_buffer),
          buffer_idx.data(),
          e_op);
      }
      if (h_offsets[1] - h_offsets[0] > 0) {
        raft::grid_1d_warp_t update_grid(
          h_offsets[1] - h_offsets[0],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_row_for_all_nbr_mid_degree<<<update_grid.num_blocks,
                                                              update_grid.block_size,
                                                              0,
                                                              handle.get_stream()>>>(
          matrix_partition,
          frontier_rows.begin() + h_offsets[0],
          frontier_rows.begin() + h_offsets[1],
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          keys.begin(),
          get_dataframe_buffer_begin<payload_t>(payload_buffer),
          buffer_idx.data(),
          e_op);
      }
      if (frontier_rows.size() - h_offsets[1] > 0) {
        raft::grid_1d_thread_t update_grid(
          frontier_rows.size() - h_offsets[1],
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_row_for_all_nbr_low_degree<<<update_grid.num_blocks,
                                                              update_grid.block_size,
                                                              0,
                                                              handle.get_stream()>>>(
          matrix_partition,
          frontier_rows.begin() + h_offsets[1],
          frontier_rows.end(),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          keys.begin(),
          get_dataframe_buffer_begin<payload_t>(payload_buffer),
          buffer_idx.data(),
          e_op);
      }
    } else {
      if (frontier_rows.size() > 0) {
        raft::grid_1d_thread_t update_grid(
          frontier_rows.size(),
          detail::update_frontier_v_push_if_out_nbr_for_all_block_size,
          handle.get_device_properties().maxGridSize[0]);

        detail::for_all_frontier_row_for_all_nbr_low_degree<<<update_grid.num_blocks,
                                                              update_grid.block_size,
                                                              0,
                                                              handle.get_stream()>>>(
          matrix_partition,
          frontier_rows.begin(),
          frontier_rows.end(),
          adj_matrix_row_value_input_first + row_value_input_offset,
          adj_matrix_col_value_input_first,
          keys.begin(),
          get_dataframe_buffer_begin<payload_t>(payload_buffer),
          buffer_idx.data(),
          e_op);
      }
    }
  }

  if (GraphViewType::is_multi_gpu) {
    auto& comm = handle.get_comms();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of col_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }

  // 2. reduce the buffer

  auto num_buffer_elements =
    detail::sort_and_reduce_buffer_elements(handle,
                                            keys.begin(),
                                            get_dataframe_buffer_begin<payload_t>(payload_buffer),
                                            buffer_idx.value(handle.get_stream()),
                                            reduce_op);
  if (GraphViewType::is_multi_gpu) {
    // FIXME: this step is unnecessary if row_comm_size== 1
    auto& comm               = handle.get_comms();
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_size = row_comm.get_size();
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (beginning of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif

    std::vector<vertex_t> h_vertex_lasts(row_comm_size);
    for (size_t i = 0; i < h_vertex_lasts.size(); ++i) {
      h_vertex_lasts[i] = graph_view.get_vertex_partition_last(col_comm_rank * row_comm_size + i);
    }

    rmm::device_uvector<vertex_t> d_vertex_lasts(h_vertex_lasts.size(), handle.get_stream());
    raft::update_device(
      d_vertex_lasts.data(), h_vertex_lasts.data(), h_vertex_lasts.size(), handle.get_stream());
    rmm::device_uvector<edge_t> d_tx_buffer_last_boundaries(d_vertex_lasts.size(),
                                                            handle.get_stream());
    thrust::lower_bound(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        keys.begin(),
                        keys.begin() + num_buffer_elements,
                        d_vertex_lasts.begin(),
                        d_vertex_lasts.end(),
                        d_tx_buffer_last_boundaries.begin());
    std::vector<edge_t> h_tx_buffer_last_boundaries(d_tx_buffer_last_boundaries.size());
    raft::update_host(h_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.data(),
                      d_tx_buffer_last_boundaries.size(),
                      handle.get_stream());
    handle.get_stream_view().synchronize();
    std::vector<size_t> tx_counts(h_tx_buffer_last_boundaries.size());
    std::adjacent_difference(
      h_tx_buffer_last_boundaries.begin(), h_tx_buffer_last_boundaries.end(), tx_counts.begin());

    rmm::device_uvector<vertex_t> rx_keys(size_t{0}, handle.get_stream());
    std::tie(rx_keys, std::ignore) =
      shuffle_values(row_comm, keys.begin(), tx_counts, handle.get_stream());
    keys = std::move(rx_keys);

    auto rx_payload_buffer = allocate_dataframe_buffer<payload_t>(size_t{0}, handle.get_stream());
    std::tie(rx_payload_buffer, std::ignore) =
      shuffle_values(row_comm,
                     get_dataframe_buffer_begin<payload_t>(payload_buffer),
                     tx_counts,
                     handle.get_stream());
    payload_buffer = std::move(rx_payload_buffer);

    num_buffer_elements =
      detail::sort_and_reduce_buffer_elements(handle,
                                              keys.begin(),
                                              get_dataframe_buffer_begin<payload_t>(payload_buffer),
                                              keys.size(),
                                              reduce_op);

    // barrier is necessary here to avoid potential overlap (which can leads to deadlock) between
    // two different communicators (end of row_comm)
#if 1
    // FIXME: temporary hack till UCC is integrated into RAFT (so we can use UCC barrier with DASK
    // and MPI barrier with MPI)
    host_barrier(comm, handle.get_stream_view());
#else
    handle.get_stream_view().synchronize();
    comm.barrier();  // currently, this is ncclAllReduce
#endif
  }

  // 3. update vertex properties

  if (num_buffer_elements > 0) {
    static_assert(VertexFrontierType::kNumBuckets <= std::numeric_limits<uint8_t>::max());
    rmm::device_uvector<uint8_t> bucket_indices(num_buffer_elements, handle.get_stream());

    vertex_partition_device_t<GraphViewType> vertex_partition(graph_view);

    auto key_payload_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(keys.begin(), get_dataframe_buffer_begin<payload_t>(payload_buffer)));
    thrust::transform(
      rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
      key_payload_pair_first,
      key_payload_pair_first + num_buffer_elements,
      bucket_indices.begin(),
      [vertex_value_input_first,
       vertex_value_output_first,
       v_op,
       vertex_partition,
       invalid_bucket_idx = VertexFrontierType::kInvalidBucketIdx] __device__(auto pair) {
        auto key         = thrust::get<0>(pair);
        auto payload     = thrust::get<1>(pair);
        auto key_offset  = vertex_partition.get_local_vertex_offset_from_vertex_nocheck(key);
        auto v_val       = *(vertex_value_input_first + key_offset);
        auto v_op_result = v_op(v_val, payload);
        auto bucket_idx  = thrust::get<0>(v_op_result);
        if (bucket_idx != invalid_bucket_idx) {
          *(vertex_value_output_first + key_offset) = thrust::get<1>(v_op_result);
          return static_cast<uint8_t>(bucket_idx);
        } else {
          return std::numeric_limits<uint8_t>::max();
        }
      });

    resize_dataframe_buffer<payload_t>(payload_buffer, size_t{0}, handle.get_stream());
    shrink_to_fit_dataframe_buffer<payload_t>(payload_buffer, handle.get_stream());

    auto bucket_key_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), keys.begin()));
    keys.resize(thrust::distance(
                  bucket_key_pair_first,
                  thrust::remove_if(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                                    bucket_key_pair_first,
                                    bucket_key_pair_first + num_buffer_elements,
                                    [] __device__(auto pair) {
                                      return thrust::get<0>(pair) ==
                                             std::numeric_limits<uint8_t>::max();
                                    })),
                handle.get_stream());
    bucket_indices.resize(keys.size(), handle.get_stream());
    keys.shrink_to_fit(handle.get_stream());
    bucket_indices.shrink_to_fit(handle.get_stream());

    bucket_key_pair_first =
      thrust::make_zip_iterator(thrust::make_tuple(bucket_indices.begin(), keys.begin()));
    if (next_frontier_bucket_indices.size() == 1) {
      vertex_frontier.get_bucket(next_frontier_bucket_indices[0]).insert(keys.begin(), keys.size());
    } else if (next_frontier_bucket_indices.size() == 2) {
      auto first_bucket_size = thrust::distance(
        bucket_key_pair_first,
        thrust::stable_partition(  // stalbe_partition to maintain sorted order within each bucket
          rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
          bucket_key_pair_first,
          bucket_key_pair_first + bucket_indices.size(),
          [first_bucket_idx = static_cast<uint8_t>(next_frontier_bucket_indices[0])] __device__(
            auto pair) { return thrust::get<0>(pair) == first_bucket_idx; }));
      vertex_frontier.get_bucket(next_frontier_bucket_indices[0])
        .insert(keys.begin(), first_bucket_size);
      vertex_frontier.get_bucket(next_frontier_bucket_indices[1])
        .insert(keys.begin() + first_bucket_size,
                thrust::distance(keys.begin() + first_bucket_size, keys.end()));
    } else {
      thrust::sort(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                   bucket_key_pair_first,
                   bucket_key_pair_first + bucket_indices.size());
      rmm::device_uvector<uint8_t> d_indices(next_frontier_bucket_indices.size(),
                                             handle.get_stream());
      rmm::device_uvector<size_t> d_counts(d_indices.size(), handle.get_stream());
      auto it =
        thrust::reduce_by_key(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                              bucket_indices.begin(),
                              bucket_indices.end(),
                              thrust::make_constant_iterator(size_t{1}),
                              d_indices.begin(),
                              d_counts.begin());
      d_indices.resize(thrust::distance(d_indices.begin(), thrust::get<0>(it)),
                       handle.get_stream());
      d_counts.resize(d_indices.size(), handle.get_stream());
      std::vector<uint8_t> h_indices(d_indices.size());
      std::vector<size_t> h_counts(h_indices.size());
      raft::update_host(h_indices.data(), d_indices.data(), d_indices.size(), handle.get_stream());
      raft::update_host(h_counts.data(), d_counts.data(), d_counts.size(), handle.get_stream());
      handle.get_stream_view().synchronize();
      std::vector<size_t> h_offsets(h_indices.size(), 0);
      std::partial_sum(h_counts.begin(), h_counts.end() - 1, h_offsets.begin() + 1);
      for (size_t i = 0; i < h_indices.size(); ++i) {
        if (h_counts[i] > 0) {
          vertex_frontier.get_bucket(h_indices[i]).insert(keys.begin() + h_offsets[i], h_counts[i]);
        }
      }
    }
  }
}  // namespace experimental

}  // namespace experimental
}  // namespace cugraph
