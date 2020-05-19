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

#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/type_traits/integer_sequence.h>
#include <thrust/distance.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <type_traits>


namespace {

template <typename TupleType, typename vertex_t, size_t tuple_size>
struct make_buffer_item {
  __device__ auto make(TupleType e_op_result, vertex_t col_offset) {
    assert(false);  // unimplemented.
    return thrust::make_tuple(col_offset);
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_item<TupleType, vertex_t, 1> {
  __device__ auto make(TupleType e_op_result, vertex_t col_offset) {
    // skip thrust::get<0>(e_op_result) as it is a push indicator
    return thrust::make_tuple(col_offset);
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_item<TupleType, vertex_t, 2> {
  __device__ auto make(TupleType e_op_result, vertex_t col_offset) {
    // skip thrust::get<0>(e_op_result) as it is a push indicator
    return thrust::make_tuple(col_offset, thrust::get<1>(e_op_result));
  }
};

template <typename TupleType, typename vertex_t>
struct make_buffer_item<TupleType, vertex_t, 3> {
  __device__ auto make(TupleType e_op_result, vertex_t col_offset) {
    // skip thrust::get<0>(e_op_result) as it is a push indicator
    return thrust::make_tuple(
      col_offset, thrust::get<1>(e_op_result), thrust::get<2>(e_op_result));
  }
};

template <typename GraphType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename BufferOutputIterator,
          typename EdgeOp>
__global__
void for_all_v_in_frontier_for_all_nbr_of_v_low_out_degree(
    GraphType graph_device_view,
    RowIterator row_first, RowIterator row_last,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    BufferOutputIterator buffer_output_first, size_t* buffer_idx_ptr,
    EdgeOp e_op) {
  auto num_rows = static_cast<size_t>(thrust::distance(row_first, row_last));
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t idx = tid;

  auto graph_offset_first = graph_device_view.offset_data();
  auto graph_index_first = graph_device_view.index_data();

  while (idx < num_rows) {
    auto row_offset = graph_device_view.get_this_partition_row_offset_from_row_nocheck(*(row_first + idx));
    auto nbr_offset_first = *(graph_offset_first + row_offset);
    auto nbr_offset_last = *(graph_offset_first + (row_offset + 1));
    for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
      auto nbr_vid = *(graph_index_first + nbr_offset);
      auto col_offset = graph_device_view.get_this_partition_col_offset_from_col_nocheck(nbr_vid);
      auto e_op_result =
        e_op(
          *(adj_matrix_row_value_input_first + row_offset),
          *(adj_matrix_col_value_input_first + col_offset));
      if (thrust::get<0>(e_op_result) == true) {
        // FIXME: This atomicInc serializes execution. If we renumber vertices to insure that rows
        // within a partition are sorted by their out-degree in decreasing order, we can compute
        // a tight uppper bound for the maximum number of pushes per warp/block and use shared
        // memory buffer to reduce the number of atomicAdd operations.
        static_assert(sizeof(unsigned long long int) == sizeof(size_t));
        auto buffer_idx =
          atomicAdd(
            reinterpret_cast<unsigned long long int*>(buffer_idx_ptr),
            static_cast<unsigned long long int>(1));
        *(buffer_output_first + buffer_idx) =
          make_buffer_item<
            decltype(e_op_result), decltype(col_offset), thrust::tuple_size<decltype(e_op_result)>::value
          >{}.make(e_op_result, col_offset);
      }
    }

    idx += gridDim.x * blockDim.x;
  }
}

}

namespace cugraph {
namespace experimental {
namespace detail {

template <typename HandleType, typename GraphType,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename EdgeOp, typename T>
void transform_v_transform_reduce_e(
    HandleType handle, GraphType graph,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueOutputIterator vertex_value_output_first,
    EdgeOp e_op, T init);

template <typename HandleType, typename GraphType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename RowFrontierType,
          typename EdgeOp, typename ReduceOp, typename VertexOp>
void expand_and_transform_if_v_push_if_e(
    HandleType handle, GraphType graph,
    RowIterator row_first, RowIterator row_last,
    AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
    AdjMatrixColValueInputIterator adj_matrix_row_value_input_last,
    VertexValueInputIterator vertex_value_input_first,
    VertexValueOutputIterator vertex_value_output_first,
    RowFrontierType row_frontier,
    EdgeOp e_op, ReduceOp reduce_op, VertexOp v_op);

template <typename HandleType, typename GraphType,
          typename RowIterator,
          typename AdjMatrixRowValueInputIterator, typename AdjMatrixColValueInputIterator,
          typename VertexValueInputIterator,
          typename VertexValueOutputIterator,
          typename RowFrontierType,
          typename EdgeOp, typename ReduceOp, typename VertexOp>
void expand_and_transform_if_v_push_if_e(
  HandleType handle, GraphType graph_device_view,
  RowIterator row_first, RowIterator row_last,
  AdjMatrixRowValueInputIterator adj_matrix_row_value_input_first,
  AdjMatrixColValueInputIterator adj_matrix_col_value_input_first,
  VertexValueInputIterator vertex_value_input_first,
  VertexValueOutputIterator vertex_value_output_first,
  RowFrontierType row_frontier,
  EdgeOp e_op, ReduceOp reduce_op, VertexOp v_op) {
  using reduce_op_input_t = typename ReduceOp::type;

  auto max_pushes =
    thrust::transform_reduce(
      thrust::cuda::par.on(handle.get_default_stream()),
      row_first, row_last,
      [graph_device_view] __device__ (auto row) {
        auto graph_offset_first = graph_device_view.offset_data();
        auto row_offset = graph_device_view.get_this_partition_row_offset_from_row_nocheck(row);
        return static_cast<size_t>(
          *(graph_offset_first + row_offset + 1) - *(graph_offset_first + row_offset));
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

  // FIXME: block size requires later tuning
  auto constexpr block_size{256};
  // FIXME: better add a utility function to compute this similar to cuDF
  auto num_blocks =
    static_cast<int>(
      std::min<size_t>(
        (thrust::distance(row_first, row_last) + block_size - 1) / block_size,
        handle.get_max_num_blocks_1D()));

  // FIXME: This is highly inefficeint for graphs with high-degree vertices. If we renumber
  // vertices to insure that rows within a partition are sorted by their out-degree in decreasing
  // order, we will apply this kernel only to low out-degree vertices.
  for_all_v_in_frontier_for_all_nbr_of_v_low_out_degree<<<
    num_blocks, block_size, 0, handle.get_default_stream()
  >>>(
    graph_device_view,
    row_first, row_last,
    adj_matrix_row_value_input_first,
    adj_matrix_col_value_input_first,
    row_frontier.buffer_begin(), row_frontier.get_buffer_idx_ptr(), e_op);

#if 0
  if (std::is_same<reduce_op, reduce_op::any<reduce_op_input_t>>::value) {
    thrust::sort();
    thrust::unique();
  }
  else if () {
    thrust::sort();
    thrust::unique();
  }

  for () {
    *(vertex_value_output_first + ) = v_op;
    row_froniter.insert();
  }


  //rmm::device_buffer buffer{};
  //rmm::device_vector<void*> buffer_ptrs{};

  get_zip_iterator<reduce_op_input_t>();
  
  // FIXME: this memory alloation is highly pessimistic (if we have a mechanism to suspend and
  // resume if a number of pushes exceeds the buffer size, we can start with a smaller buffer size)
  // FIXME: better not avoid reallocation in every iteration (in particular with the new CUDA
  // feature to reserve virtual address space without actually allocating physical memory)
  //rmm::device_vector<reduce_op_input_t> buffer_elements(max_pushes);

  
  


  //e_op_return_type test_type{true, static_cast<uint32_t>(1)};
  //static_assert(std::is_same<e_op_return_type, thrust::tuple<__nv_bool, unsigned int, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type, thrust::null_type>>::value);
  //e_op_return_type test{false, 3};
    //size_t constexpr i = thrust::tuple_size<e_op_return_type>::value;
  //using test_type = typename std::result_of<thrust::make_tuple(bool, int)>::type;
  //size_t constexpr i = thrust::tuple_size<test_type>::value - 1;
  //std::cout << i << "\n";
  //e_op_return_type ret{3, 5, 7};
  //static_assert(std::is_same<e_op_return_type, thrust::tuple<bool, unsigned int>>::value);
  // static_assert(thrust::tuple_size<typename e_op_return_type::tail_type>::value == 1);

auto constexpr num_tuple_elements = static_cast<size_t>(thrust::tuple_size<e_op_return_type>::value - 1)/* exclude bool push */;
  std::array<size_t, num_tuple_elements> sizes{}; 
  sizes[0] = compute_tuple_element_size<e_op_return_type, 1>().compute();
  //thrust::transform_reduce(row_first, row_last, [] __device__ {})

  rmm::device_vector<thrust::tuple<vertex_t*, size_t*>> frontier_buckets;
  for (size_t i = 0; i < row_frontier.get_num_buckets(); ++i) {
    frontier_buckets.emplace_back(
      thrust::raw_pointer_cast<row_frontier.get_bucket(i).end(), row_frontier.get_bucket(i).);
  }
#endif
  return;
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
if graph is symmetric, there will be additional optimization opportunities (e.g. in-degree == out-degree)
For BFS, sending a bit vector (for the entire set of dest vertices per partitoin may work better
we can use thrust::set_intersection for triangle counting
think about adding thrust wrappers for reduction functions.
thrust::(); if (opg) { allreduce }; can be cugraph::(), and be more consistant with other APIs that
hide communication inside if opg.
Can I pass nullptr for dummy instead of thrust::make_counting_iterator(0)?
*/

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph