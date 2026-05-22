/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/detail/utility_wrappers_device_sort_scalar.hpp>
#include <cugraph/detail/utility_wrappers_device_sort_zip_traits.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

#include <iterator>

namespace cugraph {
namespace detail {

/** Value type of @p Iterator for @ref device_sort constraints (C++17-friendly). */
template <typename Iterator>
using device_sort_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Sort elements in [first, last)
 *
 * For iterators whose value type satisfies @ref device_sort_scalar_value_v, dispatches
 * to @ref device_sort_impl (no Thrust in this header). For other random-access iterators (e.g.
 * lexicographic @c thrust::make_zip_iterator ranges), chooses @ref device_sort_impl when
 * @ref device_sort_supported_v holds and otherwise uses @c thrust::sort.
 *
 * @tparam      ExecutionPolicy       Thrust execution policy (e.g. @c handle.get_thrust_policy())
 * @tparam      RandomAccessIterator  random-access iterator
 *
 * @param[in]   policy       Thrust execution policy
 * @param[in]   first        beginning of the range to sort
 * @param[in]   last         end of the range to sort
 *
 */
template <
  typename ExecutionPolicy,
  typename RandomAccessIterator,
  std::enable_if_t<device_sort_scalar_value_v<device_sort_iterator_value_t<RandomAccessIterator>>,
                   int> = 0>
void device_sort(ExecutionPolicy const& policy,
                 RandomAccessIterator first,
                 RandomAccessIterator last)
{
  device_sort_impl(policy, first, last);
}

/**
 * @ingroup utility_wrappers_cpp
 * @brief Non-scalar-iterator @ref device_sort: @ref device_sort_impl when explicitly
 *        instantiated, otherwise @c thrust::sort.
 */
template <
  typename ExecutionPolicy,
  typename RandomAccessIterator,
  std::enable_if_t<!device_sort_scalar_value_v<device_sort_iterator_value_t<RandomAccessIterator>>,
                   int> = 0>
void device_sort(ExecutionPolicy const& policy,
                 RandomAccessIterator first,
                 RandomAccessIterator last)
{
  if constexpr (device_sort_supported_v<std::remove_cv_t<RandomAccessIterator>>) {
    device_sort_impl(policy, first, last);
  } else {
    thrust::sort(policy, first, last);
  }
}

/**
 * @ingroup utility_wrappers_cpp
 * @brief Sort elements in [first, last) with a custom comparison (delegates to @c thrust::sort).
 */
template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
void device_sort(ExecutionPolicy const& policy,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 Compare compare)
{
  thrust::sort(policy, first, last, compare);
}

}  // namespace detail
}  // namespace cugraph
