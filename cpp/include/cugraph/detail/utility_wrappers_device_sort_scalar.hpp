/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace detail {

/** @brief True for value types dispatched to @ref device_sort_impl without Thrust in headers. */
template <typename T>
inline constexpr bool device_sort_scalar_value_v =
  std::disjunction_v<std::is_same<std::remove_cv_t<T>, std::int32_t>,
                     std::is_same<std::remove_cv_t<T>, std::uint32_t>,
                     std::is_same<std::remove_cv_t<T>, std::int64_t>>;

}  // namespace detail
}  // namespace cugraph
