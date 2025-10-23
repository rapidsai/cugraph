/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/device_span.hpp>

#include <cub/cub.cuh>
#include <cuda/std/tuple>
#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/functional.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>

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

}  // namespace detail

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
struct property_op : public Op<T> {};

template <typename... Args, template <typename> typename Op>
struct property_op<cuda::std::tuple<Args...>, Op> {
  using Type = cuda::std::tuple<Args...>;

 private:
  template <typename T, std::size_t... Is>
  __host__ __device__ constexpr auto binary_op_impl(T& t1, T& t2, std::index_sequence<Is...>) const
  {
    return cuda::std::make_tuple((Op<typename cuda::std::tuple_element<Is, Type>::type>()(
      cuda::std::get<Is>(t1), cuda::std::get<Is>(t2)))...);
  }

 public:
  __host__ __device__ constexpr auto operator()(const Type& t1, const Type& t2) const
  {
    return binary_op_impl(t1, t2, std::make_index_sequence<cuda::std::tuple_size<Type>::value>());
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
