/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/policy.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief True for value types dispatched to @ref sequence_impl. */
template <typename T>
inline constexpr bool sequence_supported_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief True when @p Iterator is a pointer to a @ref sequence_supported_scalar_v type. */
template <typename Iterator>
inline constexpr bool sequence_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  sequence_supported_scalar_v<iterator_value_t<Iterator>>;

template <typename ForwardIterator, typename T>
void sequence_impl(
  rmm::exec_policy const& policy, ForwardIterator first, ForwardIterator last, T init, T step);

template <typename ForwardIterator, typename T>
void sequence_impl(rmm::exec_policy_nosync const& policy,
                   ForwardIterator first,
                   ForwardIterator last,
                   T init,
                   T step);

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Fill [first, last) with consecutive integers starting at 0
 *
 * Similar to @c thrust::sequence; dispatches to an explicitly instantiated backend for supported
 * scalar ranges on @c rmm::exec_policy and @c rmm::exec_policy_nosync.
 */
struct sequence_t {
  template <typename ExecutionPolicy, typename ForwardIterator>
  void operator()(ExecutionPolicy const& policy, ForwardIterator first, ForwardIterator last) const
  {
    using value_t = detail::iterator_value_t<ForwardIterator>;
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::sequence_supported_iterator_v<ForwardIterator>) {
      detail::sequence_impl(policy, first, last, value_t{}, value_t{1});
    } else {
      thrust::sequence(policy, first, last);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Fill [first, last) with consecutive integers starting at @p init
   *
   * Similar to @c thrust::sequence.
   */
  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  void operator()(ExecutionPolicy const& policy,
                  ForwardIterator first,
                  ForwardIterator last,
                  T init) const
  {
    using value_t = detail::iterator_value_t<ForwardIterator>;
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::sequence_supported_iterator_v<ForwardIterator>) {
      detail::sequence_impl(policy, first, last, value_t(init), value_t{1});
    } else {
      thrust::sequence(policy, first, last, init);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Fill [first, last) with consecutive integers starting at @p init with step @p step
   *
   * Similar to @c thrust::sequence.
   */
  template <typename ExecutionPolicy, typename ForwardIterator, typename T>
  void operator()(ExecutionPolicy const& policy,
                  ForwardIterator first,
                  ForwardIterator last,
                  T init,
                  T step) const
  {
    using value_t = detail::iterator_value_t<ForwardIterator>;
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::sequence_supported_iterator_v<ForwardIterator>) {
      detail::sequence_impl(policy, first, last, value_t(init), value_t(step));
    } else {
      thrust::sequence(policy, first, last, init, step);
    }
  }
};

/** @brief Fill [first, last) with consecutive integers
 *
 * This exposes cugraph::sequence using the sequence_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual sequence overloads is provided above.
 */
inline constexpr sequence_t sequence{};

}  // namespace CUGRAPH_EXPORT cugraph
