/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>
#include <cugraph/utilities/thrust_wrappers/detail/gather_scatter_traits.hpp>
#include <cugraph/utilities/thrust_wrappers/detail/shift_left_map.hpp>
#include <cugraph/utilities/thrust_wrappers/policy.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/scatter.h>

#include <cstddef>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Scatter [input_first, input_last) to @p output_first using @p map_first (out-of-line
 * implementation).
 */
template <typename InputIterator, typename OutputIterator, typename MapType>
void scatter_impl(rmm::exec_policy const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapType const* map_first,
                  OutputIterator output_first);

template <typename InputIterator, typename OutputIterator, typename MapType>
void scatter_impl(rmm::exec_policy_nosync const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapType const* map_first,
                  OutputIterator output_first);

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Scatter using a @c cuda::transform_iterator map with @ref shift_left_t (out-of-line
 * implementation).
 */
template <typename InputIterator, typename OutputIterator, typename MapIterator>
void scatter_shift_left_impl(rmm::exec_policy const& policy,
                             InputIterator input_first,
                             InputIterator input_last,
                             MapIterator map_first,
                             OutputIterator output_first);

template <typename InputIterator, typename OutputIterator, typename MapIterator>
void scatter_shift_left_impl(rmm::exec_policy_nosync const& policy,
                             InputIterator input_first,
                             InputIterator input_last,
                             MapIterator map_first,
                             OutputIterator output_first);

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename MapIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
void scatter_shift_left(ExecutionPolicy const& policy,
                        InputIterator input_first,
                        InputIterator input_last,
                        MapIterator map_first,
                        OutputIterator output_first)
{
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                is_shift_left_transform_map_iterator_v<MapIterator> &&
                scatter_supported_scalar_iterator_pair_v<InputIterator, OutputIterator>) {
    scatter_shift_left_impl(policy, input_first, input_last, map_first, output_first);
  } else {
    thrust::scatter(policy, input_first, input_last, map_first, output_first);
  }
}

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapIterator,
          std::size_t I,
          std::size_t N>
struct scatter_shift_left_zip_split_impl {
  static void run(ExecutionPolicy const& policy,
                  InputZipIterator input_first,
                  InputZipIterator input_last,
                  MapIterator map_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    scatter_shift_left(policy,
                       cuda::std::get<I>(input_tuple),
                       cuda::std::get<I>(input_last_tuple),
                       map_first,
                       cuda::std::get<I>(output_tuple));
    scatter_shift_left_zip_split_impl<InputZipIterator,
                                      OutputZipIterator,
                                      ExecutionPolicy,
                                      MapIterator,
                                      I + 1,
                                      N>::run(policy,
                                              input_first,
                                              input_last,
                                              map_first,
                                              output_first);
  }
};

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapIterator,
          std::size_t I>
struct scatter_shift_left_zip_split_impl<InputZipIterator,
                                         OutputZipIterator,
                                         ExecutionPolicy,
                                         MapIterator,
                                         I,
                                         I> {
  static void run(
    ExecutionPolicy const&, InputZipIterator, InputZipIterator, MapIterator, OutputZipIterator)
  {
  }
};

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename MapIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
void scatter_shift_left(ExecutionPolicy const& policy,
                        InputIterator input_first,
                        InputIterator input_last,
                        MapIterator map_first,
                        OutputIterator output_first)
{
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                is_shift_left_transform_map_iterator_v<MapIterator> &&
                scatter_supported_zip_iterator_v<std::remove_cv_t<InputIterator>> &&
                std::is_same_v<std::remove_cv_t<InputIterator>, std::remove_cv_t<OutputIterator>>) {
    static_assert(cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value ==
                    cuda::std::tuple_size<typename OutputIterator::iterator_tuple>::value,
                  "scatter_shift_left zip overload requires matching tuple arity.");

    constexpr std::size_t tuple_size =
      cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value;

    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    scatter_shift_left(policy,
                       cuda::std::get<0>(input_tuple),
                       cuda::std::get<0>(input_last_tuple),
                       map_first,
                       cuda::std::get<0>(output_tuple));
    if constexpr (tuple_size > 1) {
      scatter_shift_left_zip_split_impl<InputIterator,
                                        OutputIterator,
                                        ExecutionPolicy,
                                        MapIterator,
                                        1,
                                        tuple_size>::run(policy,
                                                         input_first,
                                                         input_last,
                                                         map_first,
                                                         output_first);
    }
  } else {
    thrust::scatter(policy, input_first, input_last, map_first, output_first);
  }
}

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename MapType,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
void scatter(ExecutionPolicy const& policy,
             InputIterator input_first,
             InputIterator input_last,
             MapType const* map_first,
             OutputIterator output_first)
{
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                scatter_supported_map_value_v<MapType> &&
                scatter_supported_scalar_iterator_pair_v<InputIterator, OutputIterator>) {
    scatter_impl(policy, input_first, input_last, map_first, output_first);
  } else {
    thrust::scatter(policy, input_first, input_last, map_first, output_first);
  }
}

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapType,
          std::size_t I,
          std::size_t N>
struct scatter_zip_split_impl {
  static void run(ExecutionPolicy const& policy,
                  InputZipIterator input_first,
                  InputZipIterator input_last,
                  MapType const* map_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    scatter(policy,
            cuda::std::get<I>(input_tuple),
            cuda::std::get<I>(input_last_tuple),
            map_first,
            cuda::std::get<I>(output_tuple));
    scatter_zip_split_impl<InputZipIterator,
                           OutputZipIterator,
                           ExecutionPolicy,
                           MapType,
                           I + 1,
                           N>::run(policy, input_first, input_last, map_first, output_first);
  }
};

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapType,
          std::size_t I>
struct scatter_zip_split_impl<InputZipIterator, OutputZipIterator, ExecutionPolicy, MapType, I, I> {
  static void run(
    ExecutionPolicy const&, InputZipIterator, InputZipIterator, MapType const*, OutputZipIterator)
  {
  }
};

template <typename ExecutionPolicy,
          typename InputIterator,
          typename OutputIterator,
          typename MapType,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
void scatter(ExecutionPolicy const& policy,
             InputIterator input_first,
             InputIterator input_last,
             MapType const* map_first,
             OutputIterator output_first)
{
  static_assert(is_thrust_zip_iterator_v<OutputIterator>,
                "scatter zip overload requires a zip output iterator.");
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                scatter_supported_map_value_v<MapType> &&
                scatter_supported_zip_iterator_v<std::remove_cv_t<InputIterator>> &&
                std::is_same_v<std::remove_cv_t<InputIterator>, std::remove_cv_t<OutputIterator>>) {
    static_assert(cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value ==
                    cuda::std::tuple_size<typename OutputIterator::iterator_tuple>::value,
                  "scatter zip overload requires matching tuple arity.");

    constexpr std::size_t tuple_size =
      cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value;

    auto const& input_tuple      = input_first.get_iterator_tuple();
    auto const& input_last_tuple = input_last.get_iterator_tuple();
    auto const& output_tuple     = output_first.get_iterator_tuple();

    scatter(policy,
            cuda::std::get<0>(input_tuple),
            cuda::std::get<0>(input_last_tuple),
            map_first,
            cuda::std::get<0>(output_tuple));
    if constexpr (tuple_size > 1) {
      scatter_zip_split_impl<InputIterator,
                             OutputIterator,
                             ExecutionPolicy,
                             MapType,
                             1,
                             tuple_size>::run(policy,
                                              input_first,
                                              input_last,
                                              map_first,
                                              output_first);
    }
  } else {
    thrust::scatter(policy, input_first, input_last, map_first, output_first);
  }
}

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Scatter [input_first, input_last) to @p output_first using @p map_first
 *
 * Similar to @c thrust::scatter; dispatches to an explicitly instantiated backend when @p policy is
 * @c rmm::exec_policy or @c rmm::exec_policy_nosync, @p map_first is a pointer to @c size_t,
 * @c int32_t, or @c int64_t, a @c cuda::transform_iterator over @ref shift_left_t, or the
 * input/output ranges are supported scalar or zip iterators.
 */
struct scatter_t {
  template <typename ExecutionPolicy,
            typename InputIterator,
            typename MapIterator,
            typename OutputIterator>
  void operator()(ExecutionPolicy const& policy,
                  InputIterator input_first,
                  InputIterator input_last,
                  MapIterator map_first,
                  OutputIterator output_first) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::scatter_supported_map_iterator_v<MapIterator>) {
      detail::scatter(policy, input_first, input_last, map_first, output_first);
    } else if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                          detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                         detail::is_shift_left_transform_map_iterator_v<MapIterator>) {
      detail::scatter_shift_left(policy, input_first, input_last, map_first, output_first);
    } else {
      thrust::scatter(policy, input_first, input_last, map_first, output_first);
    }
  }
};

/** @brief Scatter [input_first, input_last) to @p output_first using @p map_first
 *
 * This exposes cugraph::scatter using the scatter_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 *
 * Documentation of the individual scatter overloads is provided above.
 */
inline constexpr scatter_t scatter{};

}  // namespace CUGRAPH_EXPORT cugraph
