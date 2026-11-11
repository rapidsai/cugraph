/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/util/device_atomics.cuh>

#include <cuda/atomic>
#include <cuda/std/tuple>
#include <thrust/iterator/detail/any_assign.h>
#include <thrust/memory.h>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

template <typename T>
__device__ auto device_atomic_ref(T& x)
{
  return cuda::atomic_ref<T, cuda::thread_scope_device>(x);
}

template <typename T>
__device__ T device_atomic_cas(T& x, T compare, T val)
{
  device_atomic_ref(x).compare_exchange_strong(compare, val, cuda::std::memory_order_relaxed);
  return compare;
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_and(Iterator iter,
                                                       TupleType tup,
                                                       std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_ref(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)))
      .fetch_and(cuda::std::get<Is>(tup), cuda::std::memory_order_relaxed)...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_or(Iterator iter,
                                                      TupleType tup,
                                                      std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_ref(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)))
      .fetch_or(cuda::std::get<Is>(tup), cuda::std::memory_order_relaxed)...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_atomic_add(Iterator iter,
                                                       TupleType tup,
                                                       std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_ref(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)))
      .fetch_add(cuda::std::get<Is>(tup), cuda::std::memory_order_relaxed)...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_cas(Iterator iter,
                                                                   TupleType comp_tup,
                                                                   TupleType val_tup,
                                                                   std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_cas(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)),
                      cuda::std::get<Is>(comp_tup),
                      cuda::std::get<Is>(val_tup))...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_min(Iterator iter,
                                                                   TupleType tup,
                                                                   std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_ref(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)))
      .fetch_min(cuda::std::get<Is>(tup), cuda::std::memory_order_relaxed)...);
}

template <typename Iterator, typename TupleType, std::size_t... Is>
__device__ constexpr TupleType thrust_tuple_elementwise_atomic_max(Iterator iter,
                                                                   TupleType tup,
                                                                   std::index_sequence<Is...>)
{
  return cuda::std::make_tuple(
    device_atomic_ref(thrust::raw_reference_cast(cuda::std::get<Is>(*iter)))
      .fetch_max(cuda::std::get<Is>(tup), cuda::std::memory_order_relaxed)...);
}

}  // namespace detail

template <typename Iterator, typename T>
__device__ std::enable_if_t<cugraph::detail::is_discard_iterator<Iterator>::value, void> atomic_and(
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
  return detail::device_atomic_ref(thrust::raw_reference_cast(*iter))
    .fetch_and(value, cuda::std::memory_order_relaxed);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_and(Iterator iter, T value)
{
  return detail::thrust_tuple_atomic_and(
    iter, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<cugraph::detail::is_discard_iterator<Iterator>::value, void> atomic_or(
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
  return detail::device_atomic_ref(thrust::raw_reference_cast(*iter))
    .fetch_or(value, cuda::std::memory_order_relaxed);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_or(Iterator iter, T value)
{
  return detail::thrust_tuple_atomic_or(
    iter, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<cugraph::detail::is_discard_iterator<Iterator>::value, void> atomic_add(
  Iterator iter, T value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  atomic_add(Iterator iter, T value)
{
  return detail::device_atomic_ref(thrust::raw_reference_cast(*iter))
    .fetch_add(value, cuda::std::memory_order_relaxed);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   T>
  atomic_add(Iterator iter, T value)
{
  static_assert(
    cuda::std::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
    cuda::std::tuple_size<T>::value);
  return detail::thrust_tuple_atomic_add(
    iter, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_arithmetic_v<T> &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  elementwise_atomic_cas(Iterator iter, T compare, T value)
{
  return detail::device_atomic_cas(thrust::raw_reference_cast(*iter), compare, value);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<T>::value &&
                     std::is_same_v<typename thrust::iterator_traits<Iterator>::value_type, T>,
                   T>
  elementwise_atomic_cas(Iterator iter, T compare, T value)
{
  return detail::thrust_tuple_elementwise_atomic_cas(
    iter, compare, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<cugraph::detail::is_discard_iterator<Iterator>::value, void>
elementwise_atomic_min(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   T>
  elementwise_atomic_min(Iterator iter, T const& value)
{
  return detail::device_atomic_ref(thrust::raw_reference_cast(*iter))
    .fetch_min(value, cuda::std::memory_order_relaxed);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   T>
  elementwise_atomic_min(Iterator iter, T const& value)
{
  static_assert(
    cuda::std::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
    cuda::std::tuple_size<T>::value);
  return detail::thrust_tuple_elementwise_atomic_min(
    iter, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ std::enable_if_t<cugraph::detail::is_discard_iterator<Iterator>::value, void>
elementwise_atomic_max(Iterator iter, T const& value)
{
  // no-op
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<std::is_same<typename thrust::iterator_traits<Iterator>::value_type, T>::value &&
                     std::is_arithmetic<T>::value,
                   T>
  elementwise_atomic_max(Iterator iter, T const& value)
{
  return detail::device_atomic_ref(thrust::raw_reference_cast(*iter))
    .fetch_max(value, cuda::std::memory_order_relaxed);
}

template <typename Iterator, typename T>
__device__
  std::enable_if_t<is_thrust_tuple<typename thrust::iterator_traits<Iterator>::value_type>::value &&
                     is_thrust_tuple<T>::value,
                   T>
  elementwise_atomic_max(Iterator iter, T const& value)
{
  static_assert(
    cuda::std::tuple_size<typename thrust::iterator_traits<Iterator>::value_type>::value ==
    cuda::std::tuple_size<T>::value);
  return detail::thrust_tuple_elementwise_atomic_max(
    iter, value, std::make_index_sequence<cuda::std::tuple_size<T>::value>{});
}

template <typename Iterator, typename T>
__device__ void packed_bool_atomic_set(Iterator iter, T offset, bool val)
{
  auto packed_output_offset = packed_bool_offset(offset);
  auto packed_output_mask   = packed_bool_mask(offset);
  if (val) {
    detail::device_atomic_ref(thrust::raw_reference_cast(*(iter + packed_output_offset)))
      .fetch_or(packed_output_mask, cuda::std::memory_order_relaxed);
  } else {
    detail::device_atomic_ref(thrust::raw_reference_cast(*(iter + packed_output_offset)))
      .fetch_and(~packed_output_mask, cuda::std::memory_order_relaxed);
  }
}

}  // namespace CUGRAPH_EXPORT cugraph
