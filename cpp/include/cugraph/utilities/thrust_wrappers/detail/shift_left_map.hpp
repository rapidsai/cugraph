/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/thrust_wrappers/detail/gather_scatter_traits.hpp>

#include <cuda/iterator>

namespace cugraph {
namespace detail {

template <typename T>
inline constexpr bool is_shift_left_functor_v = false;

template <typename T>
inline constexpr bool is_shift_left_functor_v<shift_left_t<T>> = true;

template <typename Fn, typename KeyIterator>
struct shift_left_transform_map_iterator_traits {
  static constexpr bool value = false;
};

template <typename T, typename KeyIterator>
struct shift_left_transform_map_iterator_traits<shift_left_t<T>, KeyIterator> {
  using map_value_type        = T;
  using key_iterator          = KeyIterator;
  static constexpr bool value = scatter_supported_map_value_v<T> &&
                                is_arithmetic_pointer_v<KeyIterator> &&
                                std::is_same_v<iterator_value_t<KeyIterator>, T>;
};

template <typename MapIterator>
inline constexpr bool is_shift_left_transform_map_iterator_v = false;

template <typename Fn, typename KeyIterator>
inline constexpr bool
  is_shift_left_transform_map_iterator_v<cuda::transform_iterator<Fn, KeyIterator>> =
    shift_left_transform_map_iterator_traits<Fn, KeyIterator>::value;

}  // namespace detail
}  // namespace cugraph
