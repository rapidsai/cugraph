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

#include <cugraph/graph_view.hpp>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
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
 * @brief Apply an operator to the vertex properties and reduce.
 *
 * This version iterates over the entire set of graph vertices. This function is inspired by
 * thrust::transform_reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam VertexOp Type of the unary vertex operator.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param v_op Binary operator takes vertex ID and *(@p vertex_value_input_first + i) (where i is
 * [0, @p graph_view.local_vertex_partition_range_size())) and returns a transformed value to be
 * reduced.
 * @param init Initial value to be added to the transform-reduced input vertex properties.
 * @return T Reduction of the @p v_op outputs.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename VertexOp, typename T>
T transform_reduce_v(raft::handle_t const& handle,
                     GraphViewType const& graph_view,
                     VertexValueInputIterator vertex_value_input_first,
                     VertexOp v_op,
                     T init,
                     raft::comms::op_t op = raft::comms::op_t::SUM)
{
  using vertex_t = typename GraphViewType::vertex_type;

  auto id = identity_element<T>(op);
  auto it = thrust::make_transform_iterator(
    thrust::make_counting_iterator(vertex_t{0}),
    detail::transform_reduce_call_v_op_t<vertex_t, VertexValueInputIterator, VertexOp, T>{
      graph_view.local_vertex_partition_range_first(), vertex_value_input_first, v_op});
  auto ret = op_dispatch<T>(op, [&handle, &graph_view, it, id, init](auto op) {
    return thrust::reduce(
      handle.get_thrust_policy(),
      it,
      it + graph_view.local_vertex_partition_range_size(),
      ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() != 0)) ? id : init,
      op);
  });
  if (GraphViewType::is_multi_gpu) {
    ret = host_scalar_allreduce(handle.get_comms(), ret, op, handle.get_stream());
  }
  return ret;
}

}  // namespace cugraph
