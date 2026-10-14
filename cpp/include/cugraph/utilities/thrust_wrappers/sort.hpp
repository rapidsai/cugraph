/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/policy.hpp>
#include <cugraph/utilities/thrust_wrappers/zip_types.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>

#include <cstdint>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief True for scalar value types dispatched to @ref sort_impl / @ref unique_impl. */
template <typename T>
inline constexpr bool sort_supported_arithmetic_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::uint32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief Whether iterator type @p T has an out-of-line @ref sort_impl lexicographic sort.
 *
 *  Supported @c thrust::make_zip_iterator(...) iterator types are listed in
 *  thrust_wrappers/zip_types.hpp and instantiated in thrust_wrappers/sort.cu. Scalar sorts use
 *  @ref sort_supported_scalar_iterator_v and @ref sort_impl for @c rmm::exec_policy and
 *  @c rmm::exec_policy_nosync.
 *
 *  Keep this in lockstep with those explicit instantiations: add a disjunct only when you add the
 *  matching explicit instantiation (otherwise a call through @ref cugraph::sort can fail at
 *  link time).
 */
template <typename T>
inline constexpr bool sort_supported_zip_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, zip_i32_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i64>,
                     std::is_same<std::remove_cv_t<T>, zip_sz_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_sz_i64>,
                     std::is_same<std::remove_cv_t<T>, zip_f_sz>,
                     std::is_same<std::remove_cv_t<T>, zip_d_sz>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_f>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_d>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_f>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_d>,
                     std::is_same<std::remove_cv_t<T>, zip_sz_i32_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_sz_i64_i64>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_sz>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_sz>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i64_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i32_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_i32>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_sz_i>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_sz_i>,
                     std::is_same<std::remove_cv_t<T>, zip_i32_i32_i32_sz>,
                     std::is_same<std::remove_cv_t<T>, zip_i64_i64_i64_sz>>;

/** @brief True when @p Iterator is a pointer to a @ref sort_supported_arithmetic_scalar_v type. */
template <typename Iterator>
inline constexpr bool sort_supported_scalar_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  sort_supported_arithmetic_scalar_v<iterator_value_t<Iterator>>;

/** @brief True when @p Iterator has an out-of-line @ref sort_impl / @ref unique_impl instantiation.
 */
template <typename Iterator>
inline constexpr bool sort_supported_iterator_v =
  sort_supported_scalar_iterator_v<Iterator> || sort_supported_zip_v<std::remove_cv_t<Iterator>>;

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Sort elements in [first, last) on the given CUDA stream (out-of-line implementation).
 */
template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last);

template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy_nosync const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last);

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Sort elements in [first, last)
 *
 * Similar to @c thrust::sort; dispatches to an explicitly instantiated backend for supported
 * scalar and zip-iterator ranges.
 */
struct sort_t {
  template <typename ExecutionPolicy, typename RandomAccessIterator>
  void operator()(ExecutionPolicy const& policy,
                  RandomAccessIterator first,
                  RandomAccessIterator last) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::sort_supported_iterator_v<RandomAccessIterator>) {
      detail::sort_impl(policy, first, last);
    } else {
      thrust::sort(policy, first, last);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Sort elements in [first, last) with a custom comparison
   *
   * Similar to @c thrust::sort.
   */
  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
  void operator()(ExecutionPolicy const& policy,
                  RandomAccessIterator first,
                  RandomAccessIterator last,
                  Compare compare) const
  {
    thrust::sort(policy, first, last, compare);
  }
};

/** @brief Sort elements in [first, last)
 *
 * This exposes cugraph::sort using the sort_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual sort overloads is provided above.
 */
inline constexpr sort_t sort{};

}  // namespace CUGRAPH_EXPORT cugraph
