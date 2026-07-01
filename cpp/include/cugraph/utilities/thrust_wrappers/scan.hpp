/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/policy.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/scan.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief True for value types dispatched to @ref inclusive_scan_impl / @ref exclusive_scan_impl.
 */
template <typename T>
inline constexpr bool scan_scalar_value_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief True when @p InputIterator and @p OutputIterator are pointers to the same @ref
 * scan_scalar_value_v type. */
template <typename InputIterator, typename OutputIterator>
inline constexpr bool scan_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<InputIterator>> &&
  std::is_pointer_v<std::remove_cv_t<OutputIterator>> &&
  scan_scalar_value_v<iterator_value_t<InputIterator>> &&
  std::is_same_v<iterator_value_t<InputIterator>, iterator_value_t<OutputIterator>>;

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Inclusive prefix sum on [first, last) (out-of-line; default @c plus).
 */
template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result);

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Exclusive prefix sum on [first, last) (out-of-line; default @c plus and zero init).
 */
template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result);

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Exclusive prefix sum on [first, last) with initial value (out-of-line; default @c
 * plus).
 */
template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan_impl(rmm::exec_policy const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   T init);

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan_impl(rmm::exec_policy_nosync const& policy,
                                   InputIterator first,
                                   InputIterator last,
                                   OutputIterator result,
                                   T init);

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Inclusive prefix sum on [first, last)
 *
 * Similar to @c thrust::inclusive_scan; dispatches to an explicitly instantiated backend when
 * @p policy is @c rmm::exec_policy or @c rmm::exec_policy_nosync and @p first, @p last, and
 * @p result are pointers to the same supported scalar type (@c size_t, @c int32_t, or
 * @c int64_t).
 */
struct inclusive_scan_t {
  template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::scan_supported_iterator_v<InputIterator, OutputIterator>) {
      return detail::inclusive_scan_impl(policy, first, last, result);
    } else {
      return thrust::inclusive_scan(policy, first, last, result);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Inclusive prefix sum on [first, last) with a custom associative operator.
   *
   * Similar to @c thrust::inclusive_scan.
   */
  template <typename ExecutionPolicy,
            typename InputIterator,
            typename OutputIterator,
            typename BinaryFunction>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            BinaryFunction binary_op) const
  {
    return thrust::inclusive_scan(policy, first, last, result, binary_op);
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Inclusive prefix sum on [first, last) with initial value and associative operator.
   *
   * Similar to @c thrust::inclusive_scan.
   */
  template <typename ExecutionPolicy,
            typename InputIterator,
            typename OutputIterator,
            typename T,
            typename BinaryFunction>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            T init,
                            BinaryFunction binary_op) const
  {
    return thrust::inclusive_scan(policy, first, last, result, init, binary_op);
  }
};

/** @brief Inclusive prefix sum on [first, last)
 *
 * This exposes cugraph::inclusive_scan using the inclusive_scan_wrapper_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual inclusive_scan overloads is provided above.
 */
inline constexpr inclusive_scan_t inclusive_scan{};

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Exclusive prefix sum on [first, last)
 *
 * Similar to @c thrust::exclusive_scan; dispatches to an explicitly instantiated backend when
 * @p policy is @c rmm::exec_policy or @c rmm::exec_policy_nosync and @p first, @p last, and
 * @p result are pointers to the same supported scalar type (@c size_t, @c int32_t, or
 * @c int64_t).
 */
struct exclusive_scan_t {
  template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::scan_supported_iterator_v<InputIterator, OutputIterator>) {
      return detail::exclusive_scan_impl(policy, first, last, result);
    } else {
      return thrust::exclusive_scan(policy, first, last, result);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Exclusive prefix sum on [first, last) with initial value.
   *
   * Similar to @c thrust::exclusive_scan.
   */
  template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator, typename T>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            T init) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::scan_supported_iterator_v<InputIterator, OutputIterator> &&
                  std::is_same_v<detail::iterator_value_t<InputIterator>, std::remove_cv_t<T>>) {
      return detail::exclusive_scan_impl(policy, first, last, result, init);
    } else {
      return thrust::exclusive_scan(policy, first, last, result, init);
    }
  }

  /**
   * @ingroup utility_wrappers_cpp
   * @brief    Exclusive prefix sum on [first, last) with initial value and associative operator.
   *
   * Similar to @c thrust::exclusive_scan.
   */
  template <typename ExecutionPolicy,
            typename InputIterator,
            typename OutputIterator,
            typename T,
            typename BinaryFunction>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            InputIterator first,
                            InputIterator last,
                            OutputIterator result,
                            T init,
                            BinaryFunction binary_op) const
  {
    return thrust::exclusive_scan(policy, first, last, result, init, binary_op);
  }
};

/** @brief Exclusive prefix sum on [first, last)
 *
 * This exposes cugraph::exclusive_scan using the exclusive_scan_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual exclusive_scan overloads is provided above.
 */
inline constexpr exclusive_scan_t exclusive_scan{};

}  // namespace CUGRAPH_EXPORT cugraph
