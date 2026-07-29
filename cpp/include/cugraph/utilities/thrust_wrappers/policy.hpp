/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <rmm/exec_policy.hpp>

#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

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

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
