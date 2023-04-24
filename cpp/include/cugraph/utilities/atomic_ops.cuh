/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/util/device_atomics.cuh>

#include <thrust/detail/type_traits/iterator/is_discard_iterator.h>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/tuple.h>

namespace cugraph {

namespace detail {

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_and(Iterator iter,
                                                       TupleType tup,
                                                       std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    atomicAnd(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))), thrust::get<Is>(tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_or(Iterator iter,
                                                      TupleType tup,
                                                      std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    atomicOr(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))), thrust::get<Is>(tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_add(Iterator iter,
                                                       TupleType tup,
                                                       std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    atomicAdd(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))), thrust::get<Is>(tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_cas(Iterator iter,
                                                                   TupleType comp_tup,
                                                                   TupleType val_tup,
                                                                   std::index_sequence<Is...>)
{
  return thrust::make_tuple(atomicCAS(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))),
                                      thrust::get<Is>(comp_tup),
                                      thrust::get<Is>(val_tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_min(Iterator iter,
                                                                   TupleType tup,
                                                                   std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    atomicMin(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))), thrust::get<Is>(tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_max(Iterator iter,
                                                                   TupleType tup,
                                                                   std::index_sequence<Is...>)
{
  return thrust::make_tuple(
    atomicMax(&(thrust::raw_reference_cast(thrust::get<Is>(*iter))), thrust::get<Is>(tup))...);
}

}  // namespace detail

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void> atomic_and(
  Iterator iter, T value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_and(Iterator iter, T value)
{
  return atomicAnd(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_and(Iterator iter, T value)
{
  detail::thrust_tuple_atomic_and(
    iter, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void> atomic_or(
  Iterator iter, T value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_or(Iterator iter, T value)
{
  return atomicOr(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_or(Iterator iter, T value)
{
  detail::thrust_tuple_atomic_or(
    iter, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void> atomic_add(
  Iterator iter, T value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   void>
  atomic_add(Iterator iter, T value)
{
  atomicAdd(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   void>
  atomic_add(Iterator iter, T value)
{
  static_assert(thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
                thrust::tuple_size<T>::value);
  detail::thrust_tuple_atomic_add(
    iter, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  elementwise_atomic_cas(Iterator iter, T compare, T value)
{
  return atomicCAS(&(thrust::raw_reference_cast(*iter)), compare, value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  elementwise_atomic_cas(Iterator iter, T compare, T value)
{
  detail::thrust_tuple_elementwise_atomic_cas(
    iter, compare, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void>
elementwise_atomic_min(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   void>
  elementwise_atomic_min(Iterator iter, T const& value)
{
  atomicMin(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   void>
  elementwise_atomic_min(Iterator iter, T const& value)
{
  static_assert(thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
                thrust::tuple_size<T>::value);
  detail::thrust_tuple_elementwise_atomic_min(
    iter, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<thrust::detail::is_discard_iterator<Iterator>::value, void>
elementwise_atomic_max(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   void>
  elementwise_atomic_max(Iterator iter, T const& value)
{
  atomicMax(&(thrust::raw_reference_cast(*iter)), value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   void>
  elementwise_atomic_max(Iterator iter, T const& value)
{
  static_assert(thrust::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
                thrust::tuple_size<T>::value);
  detail::thrust_tuple_elementwise_atomic_max(
    iter, value, std::make_index_sequence<thrust::tuple_size<T>::value>{});
}

}  // namespace cugraph
