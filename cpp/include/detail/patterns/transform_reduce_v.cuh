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
#include <utilities/error.hpp>

#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

namespace cugraph {
namespace experimental {
namespace detail {

/**
 * @brief Apply an operator to the vertex properties and reduce.
 *
 * This version iterates over the entire set of graph vertices. This function is inspired by
 * thrust::transform_reduce().
 *
 * @tparam HandleType Type of the RAFT handle (e.g. for single-GPU or OPG).
 * @tparam GraphType Type of the passed graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam VertexOp Type of the unary vertex operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in OPG). `vertex_value_input_last` (exclusive) is
 * deduced as @p vertex_value_input_first + @p graph_device_view.get_number_of_local_vertices().
 * @param v_op Unary operator takes *(@p vertex_value_input_first + i) (where i is [0, @p
 * graph_device_view.get_number_of_local_vertices())) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p v_op outputs.
 */
template <typename HandleType,
          typename GraphType,
          typename VertexValueInputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v(HandleType& handle,
                     GraphType const& graph_device_view,
                     VertexValueInputIterator vertex_value_input_first,
                     VertexOp v_op,
                     T init)
{
  auto ret = thrust::transform_reduce(
    thrust::cuda::par.on(handle.get_stream()),
    vertex_value_input_first,
    vertex_value_input_first + graph_device_view.get_number_of_local_vertices(),
    v_op,
    init,
    thrust::plus<T>());
  if (GraphType::is_opg) {
    // need to reduce ret
    CUGRAPH_FAIL("unimplemented.");
  }
  return ret;
}

/**
 * @brief Apply an operator to the vertex properties and reduce.
 *
 * This version (conceptually) iterates over only a subst of the graph vertices. This function
 * actually works as thrust::transform_reduce() on [@p input_first, @p input_last) (followed by inter-process reduction in OPG).
 *
 * @tparam HandleType HandleType Type of the RAFT handle (e.g. for single-GPU or OPG).
 * @tparam GraphType Type of the passed graph object.
 * @tparam InputIterator Type of the iterator for input values.
 * @tparam VertexOp
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_device_view Graph object. This graph object should support pass-by-value to device
 * kernels.
 * @param input_first Iterator pointing to the beginning (inclusive) of the values to be passed to @p v_op.
 * @param input_last Iterator pointing to the end (exclusive) of the values to be passed to @p v_op.
 * @param v_op Unary operator takes *(@p input_first + i) (where i is [0, @p input_last - @p
 * input_first)) and returns a transformed value to be reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p v_op outputs.
 */
template <typename HandleType,
          typename GraphType,
          typename InputIterator,
          typename VertexOp,
          typename T>
T transform_reduce_v(HandleType& handle,
                     GraphType const& graph_device_view,
                     InputIterator input_first,
                     InputIterator input_last,
                     VertexOp v_op,
                     T init)
{
  auto ret = thrust::transform_reduce(thrust::cuda::par.on(handle.get_stream()),
                                      input_first,
                                      input_last,
                                      v_op,
                                      init,
                                      thrust::plus<T>());
  if (GraphType::is_opg) {
    // need to reduce ret
    CUGRAPH_FAIL("unimplemented.");
  }
  return ret;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
