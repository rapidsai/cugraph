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

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/comms/comms.hpp>
#include <raft/core/device_span.hpp>

#include <cub/cub.cuh>
#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

#include <array>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename key_t,
          typename vertex_t,
          typename weight_t,
          typename src_value_t,
          typename dst_value_t,
          typename EdgeOp,
          typename Enable = void>
struct edge_op_result_type;

template <typename key_t,
          typename vertex_t,
          typename weight_t,
          typename src_value_t,
          typename dst_value_t,
          typename EdgeOp>
struct edge_op_result_type<
  key_t,
  vertex_t,
  weight_t,
  src_value_t,
  dst_value_t,
  EdgeOp,
  std::enable_if_t<
    std::is_invocable_v<EdgeOp, key_t, vertex_t, weight_t, src_value_t, dst_value_t>>> {
  using type =
    typename std::invoke_result<EdgeOp, key_t, vertex_t, weight_t, src_value_t, dst_value_t>::type;
};

template <typename key_t,
          typename vertex_t,
          typename weight_t,
          typename src_value_t,
          typename dst_value_t,
          typename EdgeOp>
struct edge_op_result_type<
  key_t,
  vertex_t,
  weight_t,
  src_value_t,
  dst_value_t,
  EdgeOp,
  std::enable_if_t<std::is_invocable_v<EdgeOp, key_t, vertex_t, src_value_t, dst_value_t>>> {
  using type = typename std::invoke_result<EdgeOp, key_t, vertex_t, src_value_t, dst_value_t>::type;
};

template <typename vertex_t,
          typename src_value_t,
          typename dst_value_t,
          typename IntersectionOp,
          typename Enable = void>
struct intersection_op_result_type;

template <typename vertex_t, typename src_value_t, typename dst_value_t, typename IntersectionOp>
struct intersection_op_result_type<
  vertex_t,
  src_value_t,
  dst_value_t,
  IntersectionOp,
  std::enable_if_t<std::is_invocable_v<IntersectionOp,
                                       vertex_t,
                                       vertex_t,
                                       src_value_t,
                                       dst_value_t,
                                       raft::device_span<vertex_t const>>>> {
  using type = typename std::invoke_result<IntersectionOp,
                                           vertex_t,
                                           vertex_t,
                                           src_value_t,
                                           dst_value_t,
                                           raft::device_span<vertex_t const>>::type;
};

}  // namespace detail

template <typename GraphViewType,
          typename key_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp>
struct evaluate_edge_op {
  using vertex_type    = typename GraphViewType::vertex_type;
  using weight_type    = typename GraphViewType::weight_type;
  using src_value_type = typename EdgePartitionSrcValueInputWrapper::value_type;
  using dst_value_type = typename EdgePartitionDstValueInputWrapper::value_type;
  using result_type    = typename detail::
    edge_op_result_type<key_t, vertex_type, weight_type, src_value_type, dst_value_type, EdgeOp>::
      type;

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename W  = weight_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, W, SV, DV>,
                              typename std::invoke_result<E, K, V, W, SV, DV>::type>
  compute(K s, V d, W w, SV sv, DV dv, E e) const
  {
    return e(s, d, w, sv, dv);
  }

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename W  = weight_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, SV, DV>,
                              typename std::invoke_result<E, K, V, SV, DV>::type>
  compute(K s, V d, W w, SV sv, DV dv, E e) const
  {
    return e(s, d, sv, dv);
  }
};

template <typename GraphViewType,
          typename src_value_t,
          typename dst_value_t,
          typename IntersectionOp>
struct evaluate_intersection_op {
  using vertex_type = typename GraphViewType::vertex_type;
  using weight_type = typename GraphViewType::weight_type;
  using result_type = typename detail::
    intersection_op_result_type<vertex_type, src_value_t, dst_value_t, IntersectionOp>::type;

  template <typename V  = vertex_type,
            typename SV = src_value_t,
            typename DV = dst_value_t,
            typename I  = IntersectionOp>
  __device__
    std::enable_if_t<std::is_invocable_v<I, V, V, SV, DV, raft::device_span<V const>>,
                     typename std::invoke_result<I, V, V, SV, DV, raft::device_span<V const>>::type>
    compute(V s, V d, SV sv, DV dv, raft::device_span<V const> intersection, I i)
  {
    return i(s, d, sv, dv, intersection);
  }
};

template <typename GraphViewType,
          typename key_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgeOp,
          typename T>
struct cast_edge_op_bool_to_integer {
  static_assert(std::is_integral<T>::value);
  using vertex_type    = typename GraphViewType::vertex_type;
  using weight_type    = typename GraphViewType::weight_type;
  using src_value_type = typename EdgePartitionSrcValueInputWrapper::value_type;
  using dst_value_type = typename EdgePartitionDstValueInputWrapper::value_type;

  EdgeOp e_op{};

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename W  = weight_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, W, SV, DV>, T> operator()(
    K s, V d, W w, SV sv, DV dv) const
  {
    return e_op(s, d, w, sv, dv) ? T{1} : T{0};
  }

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, SV, DV>, T> operator()(K s,
                                                                                  V d,
                                                                                  SV sv,
                                                                                  DV dv) const
  {
    return e_op(s, d, sv, dv) ? T{1} : T{0};
  }
};

template <typename T, template <typename> typename Op>
struct property_op : public Op<T> {
};

template <typename... Args, template <typename> typename Op>
struct property_op<thrust::tuple<Args...>, Op>
  : public thrust::
      binary_function<thrust::tuple<Args...>, thrust::tuple<Args...>, thrust::tuple<Args...>> {
  using Type = thrust::tuple<Args...>;

 private:
  template <typename T, std::size_t... Is>
  __host__ __device__ constexpr auto binary_op_impl(T& t1, T& t2, std::index_sequence<Is...>) const
  {
    return thrust::make_tuple((Op<typename thrust::tuple_element<Is, Type>::type>()(
      thrust::get<Is>(t1), thrust::get<Is>(t2)))...);
  }

 public:
  __host__ __device__ constexpr auto operator()(const Type& t1, const Type& t2) const
  {
    return binary_op_impl(t1, t2, std::make_index_sequence<thrust::tuple_size<Type>::value>());
  }
};

template <typename T>
constexpr std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value, T> min_identity_element()
{
  return thrust_tuple_of_arithmetic_numeric_limits_lowest<T>();
}

template <typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value, T> min_identity_element()
{
  return std::numeric_limits<T>::lowest();
}

template <typename T>
constexpr std::enable_if_t<is_thrust_tuple_of_arithmetic<T>::value, T> max_identity_element()
{
  return thrust_tuple_of_arithmetic_numeric_limits_max<T>();
}

template <typename T>
constexpr std::enable_if_t<std::is_arithmetic<T>::value, T> max_identity_element()
{
  return std::numeric_limits<T>::max();
}

}  // namespace cugraph
