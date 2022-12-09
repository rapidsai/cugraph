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
#include <raft/util/device_atomics.cuh>

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
          typename src_value_t,
          typename dst_value_t,
          typename e_value_t,
          typename EdgeOp,
          typename Enable = void>
struct edge_op_result_type;

template <typename key_t,
          typename vertex_t,
          typename src_value_t,
          typename dst_value_t,
          typename e_value_t,
          typename EdgeOp>
struct edge_op_result_type<
  key_t,
  vertex_t,
  src_value_t,
  dst_value_t,
  e_value_t,
  EdgeOp,
  std::enable_if_t<
    std::is_invocable_v<EdgeOp, key_t, vertex_t, src_value_t, dst_value_t, e_value_t>>> {
  using type =
    typename std::invoke_result<EdgeOp, key_t, vertex_t, src_value_t, dst_value_t, e_value_t>::type;
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

template <typename T>
__device__ std::enable_if_t<std::is_arithmetic<T>::value, void> atomic_accumulate_impl(
  thrust::detail::any_assign& /* dereferencing thrust::discard_iterator results in this type */ lhs,
  T const& rhs)
{
  // no-op
}

template <typename T>
__device__ std::enable_if_t<std::is_arithmetic<T>::value, void> atomic_accumulate_impl(T& lhs,
                                                                                       T const& rhs)
{
  atomicAdd(&lhs, rhs);
}

template <typename Iterator, typename TupleType, size_t I, size_t N>
struct atomic_accumulate_thrust_tuple_impl {
  __device__ constexpr void compute(Iterator iter, TupleType const& value) const
  {
    atomic_accumulate_impl(thrust::raw_reference_cast(thrust::get<I>(*iter)),
                           thrust::get<I>(value));
    atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, I + 1, N>().compute(iter, value);
  }
};

template <typename Iterator, typename TupleType, size_t I>
struct atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, I, I> {
  __device__ constexpr void compute(Iterator iter, TupleType const& value) const {}
};

}  // namespace detail

template <typename GraphViewType,
          typename key_t,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp>
struct evaluate_edge_op {
  using vertex_type    = typename GraphViewType::vertex_type;
  using src_value_type = typename EdgePartitionSrcValueInputWrapper::value_type;
  using dst_value_type = typename EdgePartitionDstValueInputWrapper::value_type;
  using e_value_type   = typename EdgePartitionEdgeValueInputWrapper::value_type;
  using result_type    = typename detail::
    edge_op_result_type<key_t, vertex_type, src_value_type, dst_value_type, e_value_type, EdgeOp>::
      type;

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename EV = e_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, SV, DV, EV>,
                              typename std::invoke_result<E, K, V, SV, DV, EV>::type>
  compute(K s, V d, SV sv, DV dv, EV ev, E e) const
  {
    return e(s, d, sv, dv, ev);
  }
};

template <typename GraphViewType,
          typename src_value_t,
          typename dst_value_t,
          typename IntersectionOp>
struct evaluate_intersection_op {
  using vertex_type = typename GraphViewType::vertex_type;
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
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
struct cast_edge_op_bool_to_integer {
  static_assert(std::is_integral<T>::value);
  using vertex_type    = typename GraphViewType::vertex_type;
  using src_value_type = typename EdgePartitionSrcValueInputWrapper::value_type;
  using dst_value_type = typename EdgePartitionDstValueInputWrapper::value_type;
  using e_value_type   = typename EdgePartitionEdgeValueInputWrapper::value_type;

  EdgeOp e_op{};

  template <typename K  = key_t,
            typename V  = vertex_type,
            typename SV = src_value_type,
            typename DV = dst_value_type,
            typename EV = e_value_type,
            typename E  = EdgeOp>
  __device__ std::enable_if_t<std::is_invocable_v<E, K, V, SV, DV, EV>, T> operator()(
    K s, V d, SV sv, DV dv, EV ev) const
  {
    return e_op(s, d, sv, dv, ev) ? T{1} : T{0};
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

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void>
atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   void>
  atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  atomicAdd(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   void>
  atomic_accumulate_edge_op_result(Iterator iter, T const& value)
{
  static_assert(thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
                thrust::tuple_size<T>::value);
  size_t constexpr tuple_size = thrust::tuple_size<T>::value;
  detail::atomic_accumulate_thrust_tuple_impl<Iterator, T, size_t{0}, tuple_size>().compute(iter,
                                                                                            value);
}

}  // namespace cugraph
