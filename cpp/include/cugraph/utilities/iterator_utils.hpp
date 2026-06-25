/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

namespace detail {

/** Dereferenced value type of @p Iterator (cv stripped), via @c thrust::iterator_traits. */
template <typename Iterator>
using iterator_value_t =
  std::remove_cv_t<typename thrust::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/** True when @p Iterator dereferences to an arithmetic @c cuda::std::tuple (e.g. zip iterator). */
template <typename Iterator>
inline constexpr bool is_thrust_zip_iterator_v =
  is_thrust_tuple_of_arithmetic_v<iterator_value_t<std::remove_cv_t<Iterator>>>;

template <typename Iterator>
struct is_discard_iterator : public std::false_type {};

template <typename System>
struct is_discard_iterator<thrust::discard_iterator<System>> : public std::true_type {};

/// True if \p Iterator is a pointer and its pointee (after cv-removal) is a standard arithmetic
/// type.
template <typename Iterator>
inline constexpr bool is_arithmetic_pointer_v =
  std::is_pointer_v<std::decay_t<Iterator>> &&
  std::is_arithmetic_v<std::remove_cv_t<std::remove_pointer_t<std::decay_t<Iterator>>>>;

template <typename T>
T* iter_to_raw_ptr(T* ptr)
{
  return ptr;
}

template <typename T>
T* iter_to_raw_ptr(thrust::device_ptr<T> ptr)
{
  return thrust::raw_pointer_cast(ptr);
}

template <typename T>
auto iter_to_raw_ptr(thrust::detail::normal_iterator<thrust::device_ptr<T>> iter)
{
  return thrust::raw_pointer_cast(iter.base());
}

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
