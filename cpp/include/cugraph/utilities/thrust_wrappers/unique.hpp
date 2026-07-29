/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/unique.h>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief True when @p Iterator has an out-of-line @ref unique_impl instantiation.
 *
 * Unique and sort currently have the same supported iterator types.
 */
template <typename Iterator>
inline constexpr bool unique_supported_iterator_v = sort_supported_iterator_v<Iterator>;

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Keep unique elements in [first, last) on the given CUDA stream (out-of-line
 * implementation).
 */
template <typename RandomAccessIterator>
RandomAccessIterator unique_impl(rmm::exec_policy const& policy,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last);

template <typename RandomAccessIterator>
RandomAccessIterator unique_impl(rmm::exec_policy_nosync const& policy,
                                 RandomAccessIterator first,
                                 RandomAccessIterator last);

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Keep unique elements in [first, last)
 *
 * Similar to @c thrust::unique; dispatches to an explicitly instantiated backend for supported
 * scalar and zip-iterator ranges.
 */
struct unique_t {
  template <typename ExecutionPolicy, typename RandomAccessIterator>
  RandomAccessIterator operator()(ExecutionPolicy const& policy,
                                  RandomAccessIterator first,
                                  RandomAccessIterator last) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::unique_supported_iterator_v<RandomAccessIterator>) {
      return detail::unique_impl(policy, first, last);
    } else {
      return thrust::unique(policy, first, last);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Keep unique elements in [first, last) with a custom equivalence predicate
   *
   * Similar to @c thrust::unique.
   */
  template <typename ExecutionPolicy, typename ForwardIterator, typename BinaryPredicate>
  ForwardIterator operator()(ExecutionPolicy const& policy,
                             ForwardIterator first,
                             ForwardIterator last,
                             BinaryPredicate pred) const
  {
    return thrust::unique(policy, first, last, pred);
  }
};

/** @brief Keep unique elements in [first, last)
 *
 * This exposes cugraph::unique using the unique_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual unique overloads is provided above.
 */
inline constexpr unique_t unique{};

}  // namespace CUGRAPH_EXPORT cugraph
