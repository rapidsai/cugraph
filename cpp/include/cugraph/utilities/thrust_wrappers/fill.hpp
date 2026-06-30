/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/policy.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief True for value types dispatched to @ref fill_impl. */
template <typename T>
inline constexpr bool fill_supported_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::uint32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>,
                     std::is_same<std::remove_cv_t<T>, float>,
                     std::is_same<std::remove_cv_t<T>, double>>;

/** @brief True when @p Iterator is a pointer to a @ref fill_supported_scalar_v type. */
template <typename Iterator>
inline constexpr bool fill_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  fill_supported_scalar_v<iterator_value_t<Iterator>>;

template <typename ForwardIterator, typename T>
void fill_impl(rmm::exec_policy const& policy,
               ForwardIterator first,
               ForwardIterator last,
               T const& value);

template <typename ForwardIterator, typename T>
void fill_impl(rmm::exec_policy_nosync const& policy,
               ForwardIterator first,
               ForwardIterator last,
               T const& value);

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Assign @p value to every element in [first, last)
 *
 * Similar to @c thrust::fill; dispatches to an explicitly instantiated backend for supported
 * scalar ranges on @c rmm::exec_policy and @c rmm::exec_policy_nosync.
 */
struct fill_t {
  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  void operator()(ExecutionPolicy const& policy,
                  ForwardIterator first,
                  ForwardIterator last,
                  T const& value) const
  {
    using value_t = detail::iterator_value_t<ForwardIterator>;
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::fill_supported_iterator_v<ForwardIterator>) {
      detail::fill_impl(policy, first, last, value_t(value));
    } else {
      thrust::fill(policy, first, last, value);
    }
  }
};

/** @brief Assign @p value to every element in [first, last)
 *
 * This exposes cugraph::fill using the fill_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual fill overloads is provided above.
 */
inline constexpr fill_t fill{};

}  // namespace CUGRAPH_EXPORT cugraph
