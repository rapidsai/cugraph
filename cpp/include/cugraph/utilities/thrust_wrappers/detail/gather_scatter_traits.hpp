/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace detail {

/** @brief True for value types dispatched to @ref scatter_impl and @ref gather_impl. */
template <typename T>
inline constexpr bool scatter_supported_scalar_value_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>,
                     std::is_same<std::remove_cv_t<T>, float>,
                     std::is_same<std::remove_cv_t<T>, double>,
                     std::is_same<std::remove_cv_t<T>, std::size_t>>;

/** @brief True for index types used as scatter/gather map elements. */
template <typename T>
inline constexpr bool scatter_supported_map_value_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::size_t>,
                     std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

/** @brief True when @p MapIterator is a pointer to a supported scatter map index type. */
template <typename MapIterator>
inline constexpr bool scatter_supported_map_iterator_v =
  std::is_pointer_v<std::remove_cv_t<MapIterator>> &&
  scatter_supported_map_value_v<iterator_value_t<MapIterator>>;

/** @brief True when @p InputIterator and @p OutputIterator are matching supported scalar pointers.
 */
template <typename InputIterator, typename OutputIterator>
inline constexpr bool scatter_supported_scalar_iterator_pair_v =
  std::is_pointer_v<std::remove_cv_t<InputIterator>> &&
  std::is_pointer_v<std::remove_cv_t<OutputIterator>> &&
  std::is_same_v<iterator_value_t<InputIterator>, iterator_value_t<OutputIterator>> &&
  scatter_supported_scalar_value_v<iterator_value_t<InputIterator>>;

/** @brief True when @p Iterator is a supported zip iterator for @ref scatter and @ref gather. */
template <typename Iterator>
inline constexpr bool scatter_supported_zip_iterator_v =
  is_thrust_zip_iterator_v<Iterator> && sort_supported_zip_v<std::remove_cv_t<Iterator>>;

/** @brief True for value types dispatched to @ref gather_impl. */
template <typename T>
inline constexpr bool gather_supported_scalar_value_v = scatter_supported_scalar_value_v<T>;

/** @brief True for index types used as gather map elements in @ref gather_impl. */
template <typename T>
inline constexpr bool gather_supported_map_value_v = scatter_supported_map_value_v<T>;

/** @brief True when @p MapIterator is a pointer to a supported gather map index type. */
template <typename MapIterator>
inline constexpr bool gather_supported_map_iterator_v =
  scatter_supported_map_iterator_v<MapIterator>;

/** @brief True when @p InputIterator and @p OutputIterator are matching supported scalar pointers.
 */
template <typename InputIterator, typename OutputIterator>
inline constexpr bool gather_supported_scalar_iterator_pair_v =
  scatter_supported_scalar_iterator_pair_v<InputIterator, OutputIterator>;

/** @brief True when @p Iterator is a supported zip iterator for @ref gather. */
template <typename Iterator>
inline constexpr bool gather_supported_zip_iterator_v = scatter_supported_zip_iterator_v<Iterator>;

}  // namespace detail
}  // namespace cugraph
