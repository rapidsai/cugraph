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

#include <cugraph/experimental/graph_view.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.cuh>

#include <raft/handle.hpp>

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

namespace cugraph {
namespace experimental {

template <typename T>
struct ValueAdd : public thrust::plus<T> {
};

template <typename... Args>
struct ValueAdd<thrust::tuple<Args...>> : public thrust::binary_function<thrust::tuple<Args...>,
                                                                         thrust::tuple<Args...>,
                                                                         thrust::tuple<Args...>> {
  using Type = thrust::tuple<Args...>;

 private:
  template <typename T, std::size_t... I>
  __device__ constexpr auto sum_impl(T& t1, T& t2, std::index_sequence<I...>)
  {
    return thrust::make_tuple((thrust::get<I>(t1) + thrust::get<I>(t2))...);
  }

 public:
  __device__ constexpr auto operator()(const Type& t1, const Type& t2)
  {
    return sum_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<Type>::value>());
  }
};

template <typename... Args>
struct ValueAdd<std::tuple<Args...>> : public ValueAdd<thrust::tuple<Args...>> {
};
/**
 * @brief Reduce the vertex properties.
 *
 * This version iterates over the entire set of graph vertices. This function is inspired by
 * thrust::reduce().
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexValueInputIterator Type of the iterator for vertex properties.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param vertex_value_input_first Iterator pointing to the vertex properties for the first
 * (inclusive) vertex (assigned to this process in multi-GPU). `vertex_value_input_last` (exclusive)
 * is deduced as @p vertex_value_input_first + @p graph_view.get_number_of_local_vertices().
 * @param init Initial value to be added to the reduced input vertex properties.
 * @return T Reduction of the input vertex properties.
 */
template <typename GraphViewType, typename VertexValueInputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           VertexValueInputIterator vertex_value_input_first,
           T init)
{
  auto ret = thrust::reduce(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    vertex_value_input_first,
    vertex_value_input_first + graph_view.get_number_of_local_vertices(),
    ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() == 0)) ? init : T{},
    ValueAdd<T>());
  if (GraphViewType::is_multi_gpu) {
    ret = host_scalar_allreduce(handle.get_comms(), ret, handle.get_stream());
  }
  return ret;
}

/**
 * @brief Reduce the vertex properties.
 *
 * This version (conceptually) iterates over only a subset of the graph vertices. This function
 * actually works as thrust::reduce() on [@p input_first, @p input_last) (followed by
 * inter-process reduction in multi-GPU).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam InputIterator Type of the iterator for input values.
 * @tparam T Type of the initial value.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input_first Iterator pointing to the beginning (inclusive) of the values to be reduced.
 * @param input_last Iterator pointing to the end (exclusive) of the values to be reduced.
 * @param init Initial value to be added to the reduced input vertex properties.
 * @return T Reduction of the input vertex properties.
 */
template <typename GraphViewType, typename InputIterator, typename T>
T reduce_v(raft::handle_t const& handle,
           GraphViewType const& graph_view,
           InputIterator input_first,
           InputIterator input_last,
           T init)
{
  auto ret = thrust::reduce(
    rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
    input_first,
    input_last,
    ((GraphViewType::is_multi_gpu) && (handle.get_comms().get_rank() == 0)) ? init : T{},
    ValueAdd<T>());
  if (GraphViewType::is_multi_gpu) {
    ret = host_scalar_allreduce(handle.get_comms(), ret, handle.get_stream());
  }
  return ret;
}

}  // namespace experimental
}  // namespace cugraph
