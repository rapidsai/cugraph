/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/utility_wrappers_device_sort_zip_types.hpp>

#include <cstdint>
#include <type_traits>

namespace cugraph {
namespace detail {

/** @brief Whether iterator type @p T has an out-of-line @ref device_sort_impl lexicographic sort:
 *         supported @c thrust::make_zip_iterator(...) iterator types (aliases in
 *         utility_wrappers_device_sort_zip_types.hpp, instantiations in
 *         utility_wrappers_zip_device_sort_inst.cu). Scalar element sorts use
 *         @ref device_sort_scalar_value_v and @ref device_sort_impl (for @c rmm::exec_policy and
 *         @c rmm::exec_policy_nosync) via @ref device_sort in @c utility_wrappers_device_sort.cuh.
 *
 *  Keep this in lockstep with those explicit instantiations: add a disjunct only when you add the
 *  matching explicit instantiation (otherwise a call through @ref device_sort can fail at link
 *  time).
 */
template <typename T>
inline constexpr bool device_sort_supported_v =
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

}  // namespace detail
}  // namespace cugraph
