/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <prims/property_op_utils.cuh>
#include <prims/reduce_v.cuh>

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename VertexValueInputIterator, typename VertexOp, typename T>
struct transform_reduce_call_v_op_t {
  vertex_t local_vertex_partition_range_first{};
  VertexValueInputIterator vertex_value_input_first{};
  VertexOp v_op{};

  __device__ T operator()(vertex_t i)
  {
    return v_op(local_vertex_partition_range_first + i, *(vertex_value_input_first + i));
  }
};

}  // namespace detail

/**
 * @brief Reduce the transformed input vertex property values.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam VertexOp Type of the unary vertex operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator takes vertex ID and *(@p vertex_value_input_first + i) (where i is
 * [0, @p graph_view.local_vertex_partition_range_size())) and returns a transformed value to be
 * reduced.
 * @param init Initial value to be reduced with the transform-reduced input vertex property values.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in include/cugraph/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return T Transformed and reduced input vertex property values.
 */
template <typename GraphViewType,
          typename ReduceOp,
          typename VertexValueInputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     VertexValueInputIterator vertex_value_input_first,
                     VertexOp v_op,
                     T init,
                     ReduceOp reduce_op,
                     bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return reduce_v(
    handle,
    graph_view,
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      detail::transform_reduce_call_v_op_t<vertex_t, VertexValueInputIterator, VertexOp, T>{
        graph_view.local_vertex_partition_range_first(), vertex_value_input_first, v_op}),
    init,
    reduce_op);
}

/**
 * @brief Reduce the transformed input vertex property values.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam VertexOp Type of the unary vertex operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator takes vertex ID and *(@p vertex_value_input_first + i) (where i is
 * [0, @p graph_view.local_vertex_partition_range_size())) and returns a transformed value to be
 * reduced.
 * @param init Initial value to be added to the transform-reduced input vertex property values.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return T Transformed and reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename VertexOp, typename T>
T transform_reduce_v(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     VertexValueInputIterator vertex_value_input_first,
                     VertexOp v_op,
                     T init,
                     bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

  return transform_reduce_v(
    handle, graph_view, vertex_value_input_first, v_op, init, reduce_op::plus<T>{});
}

/**
 * @brief Reduce the transformed input vertex property values.
 *
 * This function is inspired by thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam VertexOp Type of the unary vertex operator.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator takes vertex ID and *(@p vertex_value_input_first + i) (where i is
 * [0, @p graph_view.local_vertex_partition_range_size())) and returns a transformed value to be
 * reduced.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Transformed and reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename VertexOp>
auto transform_reduce_v(raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        VertexValueInputIterator vertex_value_input_first,
                        VertexOp v_op,
                        bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using vertex_value_input_t =
    typename thrust::iterator_traits<VertexValueInputIterator>::value_type;
  using T = std::invoke_result_t<VertexOp, vertex_t, vertex_value_input_t>;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return transform_reduce_v(
    handle, graph_view, vertex_value_input_first, v_op, T{}, reduce_op::plus<T>{});
}

}  // namespace cugraph
