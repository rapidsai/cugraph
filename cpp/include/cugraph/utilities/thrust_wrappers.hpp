/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers_zip_types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/** @brief Whether iterator type @p T has an out-of-line @ref sort_impl lexicographic sort. */
template <typename T>
inline constexpr bool sort_supported_arithmetic_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::uint32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief Whether iterator type @p T has an out-of-line @ref sort_impl lexicographic sort.
 *
 *  Supported @c thrust::make_zip_iterator(...) iterator types are listed in
 *  thrust_wrappers_zip_types.hpp and instantiated in thrust_wrappers.cu. Scalar element sorts use
 *  @ref sort_supported_arithmetic_scalar_v and @ref sort_impl for @c rmm::exec_policy and
 *  @c rmm::exec_policy_nosync.
 *
 *  Keep this in lockstep with those explicit instantiations: add a disjunct only when you add the
 *  matching explicit instantiation (otherwise a call through @ref cugraph::sort_wrapper can fail at
 *  link
 *  time).
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

/** @brief True for value types dispatched to @ref inclusive_scan_impl / @ref exclusive_scan_impl.
 */
template <typename T>
inline constexpr bool scan_scalar_value_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief True when @p ExecutionPolicy is @c rmm::exec_policy (e.g. @c handle.get_thrust_policy()).
 */
template <typename ExecutionPolicy>
inline constexpr bool is_rmm_exec_policy_v =
  std::is_same_v<std::remove_cv_t<ExecutionPolicy>, rmm::exec_policy>;

/** Input iterator value type for @ref cugraph::inclusive_scan_wrapper / @ref
 * cugraph::exclusive_scan. */
template <typename Iterator>
using scan_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Inclusive prefix sum on [first, last) (out-of-line; default @c plus).
 */
template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_impl(rmm::exec_policy const& policy,
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

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Sort elements in [first, last) on the given CUDA stream (out-of-line implementation).
 *
 * @tparam      RandomAccessIterator  iterator type; must match an explicit instantiation in
 *                                     @c thrust_wrappers.cu.
 *
 * @param[in]   policy       @c rmm::exec_policy or @c rmm::exec_policy_nosync (e.g.
 *                           @c handle.get_thrust_policy())
 * @param[in]   first        beginning of the range to sort
 * @param[in]   last         end of the range to sort
 *
 */
template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last);

template <typename RandomAccessIterator>
void sort_impl(rmm::exec_policy_nosync const& policy,
               RandomAccessIterator first,
               RandomAccessIterator last);

/** Value type of @p Iterator for @ref cugraph::sort_wrapper constraints (C++17-friendly). */
template <typename Iterator>
using sort_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Sort elements in [first, last)
 *
 * Similar to @c thrust::sort; dispatches to an explicitly instantiated backend for supported
 * scalar and zip-iterator ranges.
 *
 * @tparam      RandomAccessIterator  random-access iterator
 *
 * @param[in]   policy       Thrust execution policy
 * @param[in]   first        beginning of the range to sort
 * @param[in]   last         end of the range to sort
 *
 */
// FIXME: Would like to call this sort, but there's an issue
// with CCCL and nvcc which results in a runtime error.
template <typename ExecutionPolicy, typename RandomAccessIterator>
void sort_wrapper(ExecutionPolicy const& policy,
                  RandomAccessIterator first,
                  RandomAccessIterator last)
{
  using value_t = detail::sort_iterator_value_t<RandomAccessIterator>;
  if constexpr (detail::sort_supported_arithmetic_scalar_v<value_t> ||
                detail::sort_supported_zip_v<std::remove_cv_t<RandomAccessIterator>>) {
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
 *
 * @tparam      RandomAccessIterator  random-access iterator
 * @tparam      Compare               comparison functor
 *
 * @param[in]   policy       Thrust execution policy
 * @param[in]   first        beginning of the range to sort
 * @param[in]   last         end of the range to sort
 * @param[in]   compare      comparison functor
 *
 */
// FIXME: Would like to call this sort, but there's an issue
// with CCCL and nvcc which results in a runtime error.
template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
void sort_wrapper(ExecutionPolicy const& policy,
                  RandomAccessIterator first,
                  RandomAccessIterator last,
                  Compare compare)
{
  thrust::sort(policy, first, last, compare);
}

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Inclusive prefix sum on [first, last)
 *
 * Similar to @c thrust::inclusive_scan; dispatches to an explicitly instantiated backend when
 * @p policy is @c rmm::exec_policy and the input element type is @c size_t, @c int32_t, or
 * @c int64_t.
 */
// FIXME: Would like to call this inclusive_scan, but there's an issue
// with CCCL and nvcc which results in a runtime error.
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan_wrapper(ExecutionPolicy const& policy,
                                      InputIterator first,
                                      InputIterator last,
                                      OutputIterator result)
{
  using value_t = detail::scan_iterator_value_t<InputIterator>;
  if constexpr (detail::is_rmm_exec_policy_v<ExecutionPolicy> &&
                detail::scan_scalar_value_v<value_t>) {
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
OutputIterator inclusive_scan_wrapper(ExecutionPolicy const& policy,
                                      InputIterator first,
                                      InputIterator last,
                                      OutputIterator result,
                                      BinaryFunction binary_op)
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
OutputIterator inclusive_scan_wrapper(ExecutionPolicy const& policy,
                                      InputIterator first,
                                      InputIterator last,
                                      OutputIterator result,
                                      T init,
                                      BinaryFunction binary_op)
{
  return thrust::inclusive_scan(policy, first, last, result, init, binary_op);
}

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Exclusive prefix sum on [first, last)
 *
 * Similar to @c thrust::exclusive_scan; dispatches to an explicitly instantiated backend when
 * @p policy is @c rmm::exec_policy and the input element type is @c size_t, @c int32_t, or
 * @c int64_t.
 */
template <typename ExecutionPolicy, typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(ExecutionPolicy const& policy,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result)
{
  using value_t = detail::scan_iterator_value_t<InputIterator>;
  if constexpr (detail::is_rmm_exec_policy_v<ExecutionPolicy> &&
                detail::scan_scalar_value_v<value_t>) {
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
OutputIterator exclusive_scan(ExecutionPolicy const& policy,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init)
{
  using value_t = detail::scan_iterator_value_t<InputIterator>;
  if constexpr (detail::is_rmm_exec_policy_v<ExecutionPolicy> &&
                detail::scan_scalar_value_v<value_t>) {
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
OutputIterator exclusive_scan(ExecutionPolicy const& policy,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              T init,
                              BinaryFunction binary_op)
{
  return thrust::exclusive_scan(policy, first, last, result, init, binary_op);
}

}  // namespace CUGRAPH_EXPORT cugraph
