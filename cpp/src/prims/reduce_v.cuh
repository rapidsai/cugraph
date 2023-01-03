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
#include <prims/reduce_op.cuh>

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
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
 * There are pre-defined reduction operators in include/cugraph/prims/reduce_op.cuh. It is
 * recommended to use the pre-defined reduction operators whenever possible as the current (and
 * future) implementations of graph primitives may check whether @p ReduceOp is a known type (or has
 * known member variables) to take a more optimized code path. See the documentation in the
 * reduce_op.cuh file for instructions on writing custom reduction operators.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return T Reduced input vertex property values.
 */
template <typename GraphViewType, typename ReduceOp, typename VertexValueInputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexValueInputIterator vertex_value_input_first,
           T init,
           ReduceOp reduce_op,
           bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  static_assert(
    std::is_same_v<
      std::remove_cv_t<typename thrust::iterator_traits<VertexValueInputIterator>::value_type>,
      std::remove_cv_t<T>>);

  if (do_expensive_check) {
    // currently, nothing to do
  }

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
    std::optional<T> local_init{std::nullopt};
    std::optional<T> local_result{std::nullopt};

    auto local_reduction_size = graph_view.local_vertex_partition_range_size();
    if constexpr (GraphViewType::is_multi_gpu) {
      if (handle.get_comms().get_rank() == int{0}) {
        local_init = init;
      } else {
        if constexpr (reduce_op::has_identity_element_v<ReduceOp>) {
          local_init = ReduceOp::identity_element;
        } else if (local_reduction_size > vertex_t{0}) {  // use the last element as local_init
          rmm::device_scalar<T> tmp(handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       vertex_value_input_first +
                         (graph_view.local_vertex_partition_range_size() - vertex_t{1}),
                       vertex_value_input_first + graph_view.local_vertex_partition_range_size(),
                       tmp.data());
          local_init = tmp.value(handle.get_stream());
          --local_reduction_size;
        } else {
          // this GPU has no value for global reduction
        }
      }
    } else {
      local_init = init;
    }

    if (local_init) {
      local_result = thrust::reduce(handle.get_thrust_policy(),
                                    vertex_value_input_first,
                                    vertex_value_input_first + local_reduction_size,
                                    *local_init,
                                    reduce_op);
    }

    if constexpr (GraphViewType::is_multi_gpu) {
      if constexpr (reduce_op::has_identity_element_v<ReduceOp>) {
        if constexpr (reduce_op::has_compatible_raft_comms_op_v<ReduceOp>) {
          ret = host_scalar_allreduce(handle.get_comms(),
                                      *local_result,
                                      ReduceOp::compatible_raft_comms_op,
                                      handle.get_stream());
        } else {
          auto rets =
            host_scalar_gather(handle.get_comms(), *local_result, int{0}, handle.get_stream());
          if (handle.get_comms().get_rank() == int{0}) {
            ret = std::reduce(rets.begin(), rets.end(), ReduceOp::identity_element, reduce_op);
          }
          ret = host_scalar_bcast(handle.get_comms(), ret, int{0}, handle.get_stream());
        }
      } else {  // no guarantee that every GPU has valid local_result
        auto rets = host_scalar_gather(
          handle.get_comms(), local_result ? *local_result : T{}, int{0}, handle.get_stream());
        if (handle.get_comms().get_rank() == int{0}) {
          std::vector<T> valid_rets{};
          valid_rets.reserve(rets.size());
          auto vertex_partition_range_offsets = graph_view.vertex_partition_range_offsets();
          for (size_t i = 0; i < rets.size(); ++i) {
            if (vertex_partition_range_offsets[i + size_t{1}] - vertex_partition_range_offsets[i] >
                vertex_t{0}) {
              valid_rets.push_back(rets[i]);
            }
          }
          ret = std::reduce(
            valid_rets.begin(), valid_rets.end() - size_t{1}, valid_rets.back(), reduce_op);
        }
        ret = host_scalar_bcast(handle.get_comms(), ret, int{0}, handle.get_stream());
      }
    } else {
      ret = *local_result;
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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return T Reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexValueInputIterator vertex_value_input_first,
           T init,
           bool do_expensive_check = false)
{
  if (do_expensive_check) {
    // currently, nothing to do
  }

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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Reduced input vertex property values.
 */
template <typename GraphViewType, typename VertexValueInputIterator>
auto reduce_v(raft::handle_t const& handle,
              GraphViewType const& graph_view,
              VertexValueInputIterator vertex_value_input_first,
              bool do_expensive_check = false)
{
  using T =
    std::remove_cv_t<typename thrust::iterator_traits<VertexValueInputIterator>::value_type>;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  return reduce_v(handle, graph_view, vertex_value_input_first, T{}, reduce_op::plus<T>{});
}

}  // namespace cugraph
