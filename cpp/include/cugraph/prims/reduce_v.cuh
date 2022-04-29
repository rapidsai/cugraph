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
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace cugraph {

/**
 * @brief Reduce the input vertex property values.
 *
 * This function is inspired by thrust::reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam T Type of the initial value. T should be an arithmetic type or thrust::tuple of
 * arithmetic types.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param init Initial value to be reduced with the reduced input vertex property values.
 * @param reduce_op Binary operator that takes two input arguments and reduce the two values to one.
 * There are pre-defined reduction operators in include/cugraph/prims/reduce_op.cuh. Recommended to
 * use the pre-defined reduction operators whenever possible as the current (and future)
 * implementations of graph primitives may check whether @p ReduceOp is known type (or has known
 * member variables) to take a more optimized code path. For example, some primitive implementations
 * check whether @p ReduceOp has the compatible_raft_comms_op member variable and use
 * raft::comms::reduce() (which calls NCCL reduce()) for reduction. Otherwise, reduction may be
 * performed using a less efficient gather based approach (we may implement tree-based reduction in
 * the future but this may be still less efficient than NCCL reduce()).
 * @return T Reduced input vertex property values.
 */
template <typename GraphViewType, typename ReduceOp, typename VertexValueInputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexValueInputIterator vertex_value_input_first,
           T init,
           ReduceOp reduce_op)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(
    std::is_same_v<
      std::remove_cv_t<typename thrust::iterator_traits<VertexValueInputIterator>::value_type>,
      std::remove_cv_t<T>>);

  if (graph_view.number_of_vertices() == vertex_t{0}) { return init; }

  T ret{};
  if constexpr (std::is_same_v<ReduceOp, reduce_op::any<T>>) {  // return the first element
    if (graph_view.local_vertex_partition_range_size() > vertex_t{0}) {
      rmm::device_scalar<T> tmp(handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   vertex_value_input_first,
                   vertex_value_input_first + size_t{1},
                   tmp.data());
      ret = tmp.value(handle.get_stream());
    }
    if constexpr (GraphViewType::is_multi_gpu) {
      auto root = host_scalar_allreduce(handle.get_comms(),
                                        graph_view.local_vertex_partition_range_size() > vertex_t{0}
                                          ? handle.get_comms().get_rank()
                                          : std::numeric_limits<int>::max(),
                                        raft::comms::op_t::MIN,
                                        handle.get_stream());
      ret       = host_scalar_bcast(handle.get_comms(), ret, root, handle.get_stream());
    }
  } else {
    if constexpr (reduce_op::has_compatible_raft_comms_op_v<ReduceOp>) {
      auto raft_comms_op = ReduceOp::compatible_raft_comms_op;
      auto id            = identity_element<T>(raft_comms_op);
      ret                = thrust::reduce(
        handle.get_thrust_policy(),
        vertex_value_input_first,
        vertex_value_input_first + graph_view.local_vertex_partition_range_size(),
        ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() != 0)) ? id : init,
        reduce_op);
      if constexpr (GraphViewType::is_multi_gpu) {
        ret = host_scalar_allreduce(handle.get_comms(), ret, raft_comms_op, handle.get_stream());
      }
    } else {
      ret =
        thrust::reduce(handle.get_thrust_policy(),
                       vertex_value_input_first,
                       vertex_value_input_first + graph_view.local_vertex_partition_range_size());
      if constexpr (GraphViewType::is_multi_gpu) {
        auto rets = host_scalar_gather(handle.get_comms(), ret, int{0}, handle.get_stream());
        if (handle.get_comms().get_rank() == int{0}) {
          auto vertex_partition_range_offsets = graph_view.vertex_partition_range_offsets();
          std::vector<T> valid_rets{};
          valid_rets.reserve(rets.size());
          for (size_t i = 0; i < rets.size(); ++i) {
            if (vertex_partition_range_offsets[i + 1] - vertex_partition_range_offsets[i] > 0) {
              valid_rets.push_back(rets[i]);
            }
          }
          ret = std::reduce(valid_rets.begin(), valid_rets.end(), init, reduce_op);
          ret = host_scalar_bcast(handle.get_comms(), ret, int{0}, handle.get_stream());
        }
      }
    }
  }

  return ret;
}

/**
 * @brief Reduce the input vertex property values.
 *
 * This function is inspired by thrust::reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam ReduceOp Type of the binary reduction operator.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam T Type of the initial value. T should be an arithmetic type or thrust::tuple of
 * arithmetic types.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @param init Initial value to be added to the reduced input vertex property values.
 * @return T Reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexValueInputIterator vertex_value_input_first,
           T init)
{
  return reduce_v(handle, graph_view, vertex_value_input_first, init, reduce_op::plus<T>{});
}

/**
 * @brief Reduce the input vertex property values.
 *
 * This function is inspired by thrust::reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex property values.
 * @tparam T Type of the initial value. T should be an arithmetic type or thrust::tuple of
 * arithmetic types.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex property values for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.local_vertex_partition_range_size().
 * @return Reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator>
auto reduce_v(raft::handle_t const& handle,
              GraphViewType const& graph_view,
              VertexValueInputIterator vertex_value_input_first)
{
  using T =
    std::remove_cv_t<typename thrust::iterator_traits<VertexValueInputIterator>::value_type>;

  return reduce_v(handle, graph_view, vertex_value_input_first, T{}, reduce_op::plus<T>{});
}

}  // namespace cugraph
