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
#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/gather.h>

#include <cstddef>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

template <typename InputIterator, typename OutputIterator, typename MapType>
OutputIterator gather_impl(rmm::exec_policy const& policy,
                           MapType const* map_first,
                           MapType const* map_last,
                           InputIterator input_first,
                           OutputIterator output_first);

template <typename InputIterator, typename OutputIterator, typename MapType>
OutputIterator gather_impl(rmm::exec_policy_nosync const& policy,
                           MapType const* map_first,
                           MapType const* map_last,
                           InputIterator input_first,
                           OutputIterator output_first);

template <typename InputIterator, typename OutputIterator, typename MapIterator>
OutputIterator gather_shift_left_impl(rmm::exec_policy const& policy,
                                      MapIterator map_first,
                                      MapIterator map_last,
                                      InputIterator input_first,
                                      OutputIterator output_first);

template <typename InputIterator, typename OutputIterator, typename MapIterator>
OutputIterator gather_shift_left_impl(rmm::exec_policy_nosync const& policy,
                                      MapIterator map_first,
                                      MapIterator map_last,
                                      InputIterator input_first,
                                      OutputIterator output_first);

template <typename ExecutionPolicy,
          typename MapIterator,
          typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator gather_shift_left(ExecutionPolicy const& policy,
                                 MapIterator map_first,
                                 MapIterator map_last,
                                 InputIterator input_first,
                                 OutputIterator output_first)
{
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                is_shift_left_transform_map_iterator_v<MapIterator> &&
                gather_supported_scalar_iterator_pair_v<InputIterator, OutputIterator>) {
    return gather_shift_left_impl(policy, map_first, map_last, input_first, output_first);
  } else {
    return thrust::gather(policy, map_first, map_last, input_first, output_first);
  }
}

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapIterator,
          std::size_t I,
          std::size_t N>
struct gather_shift_left_zip_split_impl {
  static void run(ExecutionPolicy const& policy,
                  MapIterator map_first,
                  MapIterator map_last,
                  InputZipIterator input_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple  = input_first.get_iterator_tuple();
    auto const& output_tuple = output_first.get_iterator_tuple();

    gather_shift_left(
      policy, map_first, map_last, cuda::std::get<I>(input_tuple), cuda::std::get<I>(output_tuple));
    gather_shift_left_zip_split_impl<InputZipIterator,
                                     OutputZipIterator,
                                     ExecutionPolicy,
                                     MapIterator,
                                     I + 1,
                                     N>::run(policy,
                                             map_first,
                                             map_last,
                                             input_first,
                                             output_first);
  }
};

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapIterator,
          std::size_t I>
struct gather_shift_left_zip_split_impl<InputZipIterator,
                                        OutputZipIterator,
                                        ExecutionPolicy,
                                        MapIterator,
                                        I,
                                        I> {
  static void run(
    ExecutionPolicy const&, MapIterator, MapIterator, InputZipIterator, OutputZipIterator)
  {
  }
};

template <typename ExecutionPolicy,
          typename MapIterator,
          typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator gather_shift_left(ExecutionPolicy const& policy,
                                 MapIterator map_first,
                                 MapIterator map_last,
                                 InputIterator input_first,
                                 OutputIterator output_first)
{
  static_assert(is_thrust_zip_iterator_v<OutputIterator>,
                "gather_shift_left zip overload requires a zip output iterator.");
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                is_shift_left_transform_map_iterator_v<MapIterator> &&
                gather_supported_zip_iterator_v<std::remove_cv_t<InputIterator>> &&
                std::is_same_v<std::remove_cv_t<InputIterator>, std::remove_cv_t<OutputIterator>>) {
    static_assert(cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value ==
                    cuda::std::tuple_size<typename OutputIterator::iterator_tuple>::value,
                  "gather_shift_left zip overload requires matching tuple arity.");

    constexpr std::size_t tuple_size =
      cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value;

    auto const& input_tuple  = input_first.get_iterator_tuple();
    auto const& output_tuple = output_first.get_iterator_tuple();

    gather_shift_left(
      policy, map_first, map_last, cuda::std::get<0>(input_tuple), cuda::std::get<0>(output_tuple));
    if constexpr (tuple_size > 1) {
      gather_shift_left_zip_split_impl<InputIterator,
                                       OutputIterator,
                                       ExecutionPolicy,
                                       MapIterator,
                                       1,
                                       tuple_size>::run(policy,
                                                        map_first,
                                                        map_last,
                                                        input_first,
                                                        output_first);
    }
    return thrust::next(output_first, thrust::distance(map_first, map_last));
  } else {
    return thrust::gather(policy, map_first, map_last, input_first, output_first);
  }
}

template <typename ExecutionPolicy,
          typename MapType,
          typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<!is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator gather(ExecutionPolicy const& policy,
                      MapType const* map_first,
                      MapType const* map_last,
                      InputIterator input_first,
                      OutputIterator output_first)
{
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                gather_supported_map_value_v<MapType> &&
                gather_supported_scalar_iterator_pair_v<InputIterator, OutputIterator>) {
    return gather_impl(policy, map_first, map_last, input_first, output_first);
  } else {
    return thrust::gather(policy, map_first, map_last, input_first, output_first);
  }
}

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapType,
          std::size_t I,
          std::size_t N>
struct gather_zip_split_impl {
  static void run(ExecutionPolicy const& policy,
                  MapType const* map_first,
                  MapType const* map_last,
                  InputZipIterator input_first,
                  OutputZipIterator output_first)
  {
    auto const& input_tuple  = input_first.get_iterator_tuple();
    auto const& output_tuple = output_first.get_iterator_tuple();

    gather(
      policy, map_first, map_last, cuda::std::get<I>(input_tuple), cuda::std::get<I>(output_tuple));
    gather_zip_split_impl<InputZipIterator, OutputZipIterator, ExecutionPolicy, MapType, I + 1, N>::
      run(policy, map_first, map_last, input_first, output_first);
  }
};

template <typename InputZipIterator,
          typename OutputZipIterator,
          typename ExecutionPolicy,
          typename MapType,
          std::size_t I>
struct gather_zip_split_impl<InputZipIterator, OutputZipIterator, ExecutionPolicy, MapType, I, I> {
  static void run(
    ExecutionPolicy const&, MapType const*, MapType const*, InputZipIterator, OutputZipIterator)
  {
  }
};

template <typename ExecutionPolicy,
          typename MapType,
          typename InputIterator,
          typename OutputIterator,
          std::enable_if_t<is_thrust_zip_iterator_v<InputIterator>, bool> = true>
OutputIterator gather(ExecutionPolicy const& policy,
                      MapType const* map_first,
                      MapType const* map_last,
                      InputIterator input_first,
                      OutputIterator output_first)
{
  static_assert(is_thrust_zip_iterator_v<OutputIterator>,
                "gather zip overload requires a zip output iterator.");
  if constexpr ((is_rmm_exec_policy_v<ExecutionPolicy> ||
                 is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                gather_supported_map_value_v<MapType> &&
                gather_supported_zip_iterator_v<std::remove_cv_t<InputIterator>> &&
                std::is_same_v<std::remove_cv_t<InputIterator>, std::remove_cv_t<OutputIterator>>) {
    static_assert(cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value ==
                    cuda::std::tuple_size<typename OutputIterator::iterator_tuple>::value,
                  "gather zip overload requires matching tuple arity.");

    constexpr std::size_t tuple_size =
      cuda::std::tuple_size<typename InputIterator::iterator_tuple>::value;

    auto const& input_tuple  = input_first.get_iterator_tuple();
    auto const& output_tuple = output_first.get_iterator_tuple();

    gather(
      policy, map_first, map_last, cuda::std::get<0>(input_tuple), cuda::std::get<0>(output_tuple));
    if constexpr (tuple_size > 1) {
      gather_zip_split_impl<InputIterator,
                            OutputIterator,
                            ExecutionPolicy,
                            MapType,
                            1,
                            tuple_size>::run(policy,
                                             map_first,
                                             map_last,
                                             input_first,
                                             output_first);
    }
    return thrust::next(output_first, thrust::distance(map_first, map_last));
  } else {
    return thrust::gather(policy, map_first, map_last, input_first, output_first);
  }
}

}  // namespace detail

/**
 * @ingroup utility_wrappers_cpp
 * @brief    Gather from @p input_first into @p output_first using @p map_first
 *
 * Similar to @c thrust::gather; dispatches to an explicitly instantiated backend when @p policy is
 * @c rmm::exec_policy or @c rmm::exec_policy_nosync, @p map_first is a pointer to @c size_t,
 * @c int32_t, or @c int64_t, a @c cuda::transform_iterator over @ref shift_left_t, or the
 * input/output ranges are supported scalar or zip iterators.
 */
struct gather_t {
  template <typename ExecutionPolicy,
            typename MapIterator,
            typename InputIterator,
            typename OutputIterator>
  OutputIterator operator()(ExecutionPolicy const& policy,
                            MapIterator map_first,
                            MapIterator map_last,
                            InputIterator input_first,
                            OutputIterator output_first) const
  {
    if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                   detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                  detail::gather_supported_map_iterator_v<MapIterator>) {
      return detail::gather(policy, map_first, map_last, input_first, output_first);
    } else if constexpr ((detail::is_rmm_exec_policy_v<ExecutionPolicy> ||
                          detail::is_rmm_exec_policy_nosync_v<ExecutionPolicy>) &&
                         detail::is_shift_left_transform_map_iterator_v<MapIterator>) {
      return detail::gather_shift_left(policy, map_first, map_last, input_first, output_first);
    } else {
      return thrust::gather(policy, map_first, map_last, input_first, output_first);
    }
  }
};

/** @brief Gather from @p input_first into @p output_first using @p map_first
 *
 * This exposes cugraph::gather using the gather_t functor.
 * This is a workaround to avoid ADL issues with CCCL which result in a runtime error.
 */
inline constexpr gather_t gather{};

}  // namespace CUGRAPH_EXPORT cugraph
