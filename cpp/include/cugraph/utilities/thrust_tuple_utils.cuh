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

#include <raft/cudart_utils.h>
#include <raft/device_atomics.cuh>
#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/tuple.h>

#include <array>
#include <type_traits>

namespace cugraph {

namespace detail {

template <typename TupleType, size_t I, size_t N>
struct is_thrust_tuple_of_arithemetic_impl {
  constexpr bool evaluate() const
  {
    if (!std::is_arithmetic<typename thrust::tuple_element<I, TupleType>::type>::value) {
      return false;
    } else {
      return is_thrust_tuple_of_arithemetic_impl<TupleType, I + 1, N>().evaluate();
    }
  }
};

template <typename TupleType, size_t I>
struct is_thrust_tuple_of_arithemetic_impl<TupleType, I, I> {
  constexpr bool evaluate() const { return true; }
};

template <typename TupleType, size_t I, size_t N>
struct compute_thrust_tuple_element_sizes_impl {
  void compute(std::array<size_t, thrust::tuple_size<TupleType>::value>& arr) const
  {
    arr[I] = sizeof(typename thrust::tuple_element<I, TupleType>::type);
    compute_thrust_tuple_element_sizes_impl<TupleType, I + 1, N>().compute(arr);
  }
};

template <typename TupleType, size_t I>
struct compute_thrust_tuple_element_sizes_impl<TupleType, I, I> {
  void compute(std::array<size_t, thrust::tuple_size<TupleType>::value>& arr) const {}
};

template <typename TupleType, std::size_t... Is>
auto thrust_tuple_to_std_tuple(TupleType tup, std::index_sequence<Is...>)
{
  return std::make_tuple(thrust::get<Is>(tup)...);
}

template <typename TupleType, std::size_t... Is>
auto std_tuple_to_thrust_tuple(TupleType tup, std::index_sequence<Is...>)
{
  constexpr size_t maximum_thrust_tuple_size = 10;
  static_assert(std::tuple_size_v<TupleType> <= maximum_thrust_tuple_size);
  return thrust::make_tuple(std::get<Is>(tup)...);
}

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

template <typename T>
struct is_thrust_tuple : std::false_type {
};

template <typename... Ts>
struct is_thrust_tuple<thrust::tuple<Ts...>> : std::true_type {
};

template <typename TupleType>
struct is_thrust_tuple_of_arithmetic : std::false_type {
};

template <typename... Args>
struct is_thrust_tuple_of_arithmetic<thrust::tuple<Args...>> {
 private:
  template <typename T>
  static constexpr bool is_valid = std::is_arithmetic_v<T> || std::is_same_v<T, thrust::null_type>;

 public:
  static constexpr bool value = (... && is_valid<Args>);
};

template <typename T>
struct is_std_tuple : std::false_type {
};

template <typename... Ts>
struct is_std_tuple<std::tuple<Ts...>> : std::true_type {
};

template <typename T, template <typename> typename Vector>
struct is_arithmetic_vector : std::false_type {
};

template <template <typename> typename Vector, typename T>
struct is_arithmetic_vector<Vector<T>, Vector>
  : std::integral_constant<bool, std::is_arithmetic<T>::value> {
};

template <typename T>
struct is_std_tuple_of_arithmetic_vectors : std::false_type {
};

template <typename... Args>
struct is_std_tuple_of_arithmetic_vectors<std::tuple<rmm::device_uvector<Args>...>> {
  static constexpr bool value = (... && std::is_arithmetic<Args>::value);
};

template <typename T>
struct is_arithmetic_or_thrust_tuple_of_arithmetic
  : std::integral_constant<bool, std::is_arithmetic<T>::value> {
};

template <typename... Ts>
struct is_arithmetic_or_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>
  : std::integral_constant<bool, is_thrust_tuple_of_arithmetic<thrust::tuple<Ts...>>::value> {
};

template <typename T>
struct thrust_tuple_size_or_one : std::integral_constant<size_t, 1> {
};

template <typename... Ts>
struct thrust_tuple_size_or_one<thrust::tuple<Ts...>>
  : std::integral_constant<size_t, thrust::tuple_size<thrust::tuple<Ts...>>::value> {
};

template <typename TupleType>
struct compute_thrust_tuple_element_sizes {
  auto operator()() const
  {
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    std::array<size_t, tuple_size> ret;
    detail::compute_thrust_tuple_element_sizes_impl<TupleType, size_t{0}, tuple_size>().compute(
      ret);
    return ret;
  }
};

template <typename TupleType>
auto thrust_tuple_to_std_tuple(TupleType tup)
{
  return detail::thrust_tuple_to_std_tuple(
    tup, std::make_index_sequence<thrust::tuple_size<TupleType>::value>{});
}

template <typename TupleType>
auto std_tuple_to_thrust_tuple(TupleType tup)
{
  constexpr size_t maximum_thrust_tuple_size = 10;
  static_assert(std::tuple_size_v<TupleType> <= maximum_thrust_tuple_size);
  return detail::std_tuple_to_thrust_tuple(
    tup, std::make_index_sequence<std::tuple_size_v<TupleType>>{});
}

// a temporary function to emulate thrust::tuple_cat (not supported) using std::tuple_cat (should
// retire once thrust::tuple is replaced with cuda::std::tuple)
template <typename... TupleTypes>
auto thrust_tuple_cat(TupleTypes... tups)
{
  return std_tuple_to_thrust_tuple(std::tuple_cat(thrust_tuple_to_std_tuple(tups)...));
}

template <typename Iterator, typename TupleType>
struct atomic_accumulate_thrust_tuple {
  __device__ constexpr void operator()(Iterator iter, TupleType const& value) const
  {
    static_assert(
      thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
      thrust::tuple_size<TupleType>::value);
    size_t constexpr tuple_size = thrust::tuple_size<TupleType>::value;
    detail::atomic_accumulate_thrust_tuple_impl<Iterator, TupleType, size_t{0}, tuple_size>()
      .compute(iter, value);
  }
};

}  // namespace cugraph
