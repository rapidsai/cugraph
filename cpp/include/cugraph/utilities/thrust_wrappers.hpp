/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers_zip_types.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

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
 *  thrust_wrappers_zip_types.hpp and instantiated in thrust_wrappers.cu. Scalar sorts use
 *  @ref sort_supported_scalar_iterator_v and @ref sort_impl for @c rmm::exec_policy and
 *  @c rmm::exec_policy_nosync.
 *
 *  Keep this in lockstep with those explicit instantiations: add a disjunct only when you add the
 *  matching explicit instantiation (otherwise a call through @ref cugraph::sort can fail at
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

/** @brief True when @p ExecutionPolicy is @c rmm::exec_policy.
 */
template <typename ExecutionPolicy>
inline constexpr bool is_rmm_exec_policy_v =
  std::is_same_v<std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>, rmm::exec_policy>;

/** @brief True when @p ExecutionPolicy is @c rmm::exec_policy_nosync (e.g. @c
 * handle.get_thrust_policy()).
 */
template <typename ExecutionPolicy>
inline constexpr bool is_rmm_exec_policy_nosync_v =
  std::is_same_v<std::remove_cv_t<std::remove_reference_t<ExecutionPolicy>>,
                 rmm::exec_policy_nosync>;

/** Input iterator value type for @ref cugraph::inclusive_scan / @ref cugraph::exclusive_scan. */
template <typename Iterator>
using scan_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/** @brief True when @p InputIterator and @p OutputIterator are pointers to the same @ref
 * scan_scalar_value_v type. */
template <typename InputIterator, typename OutputIterator>
inline constexpr bool scan_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<InputIterator>> &&
  std::is_pointer_v<std::remove_cv_t<OutputIterator>> &&
  scan_scalar_value_v<scan_iterator_value_t<InputIterator>> &&
  std::is_same_v<scan_iterator_value_t<InputIterator>, scan_iterator_value_t<OutputIterator>>;

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

/** Value type of @p Iterator for @ref cugraph::sort constraints (C++17-friendly). */
template <typename Iterator>
using sort_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/** @brief True when @p Iterator is a pointer to a @ref sort_supported_arithmetic_scalar_v type. */
template <typename Iterator>
inline constexpr bool sort_supported_scalar_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  sort_supported_arithmetic_scalar_v<sort_iterator_value_t<Iterator>>;

/** @brief True when @p Iterator has an out-of-line @ref sort_impl / @ref unique_impl instantiation.
 */
template <typename Iterator>
inline constexpr bool sort_supported_iterator_v =
  sort_supported_scalar_iterator_v<Iterator> || sort_supported_zip_v<std::remove_cv_t<Iterator>>;

/** @brief True for value types dispatched to @ref fill_impl. */
template <typename T>
inline constexpr bool fill_supported_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::uint32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>,
                     std::is_same<std::remove_cv_t<T>, float>,
                     std::is_same<std::remove_cv_t<T>, double>>;

/** @brief True for value types dispatched to @ref sequence_impl. */
template <typename T>
inline constexpr bool sequence_supported_scalar_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** Value type of @p Iterator for @ref cugraph::fill constraints. */
template <typename Iterator>
using fill_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/** Value type of @p Iterator for @ref cugraph::sequence constraints. */
template <typename Iterator>
using sequence_iterator_value_t =
  std::remove_cv_t<typename std::iterator_traits<std::remove_cv_t<Iterator>>::value_type>;

/** @brief True when @p Iterator is a pointer to a @ref fill_supported_scalar_v type. */
template <typename Iterator>
inline constexpr bool fill_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  fill_supported_scalar_v<fill_iterator_value_t<Iterator>>;

/** @brief True when @p Iterator is a pointer to a @ref sequence_supported_scalar_v type. */
template <typename Iterator>
inline constexpr bool sequence_supported_iterator_v =
  std::is_pointer_v<std::remove_cv_t<Iterator>> &&
  sequence_supported_scalar_v<sequence_iterator_value_t<Iterator>>;

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
                  detail::sort_supported_iterator_v<RandomAccessIterator>) {
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
                  std::is_same_v<detail::scan_iterator_value_t<InputIterator>,
                                 std::remove_cv_t<T>>) {
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
    using value_t = detail::fill_iterator_value_t<ForwardIterator>;
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
    using value_t = detail::sequence_iterator_value_t<ForwardIterator>;
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
    using value_t = detail::sequence_iterator_value_t<ForwardIterator>;
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
    using value_t = detail::sequence_iterator_value_t<ForwardIterator>;
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
