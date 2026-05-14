/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/export.hpp>
#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>

#include <raft/core/device_span.hpp>

namespace CUGRAPH_EXPORT cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Manage device spans on each GPU
 */
template <typename T>
using device_shared_device_span_t = device_shared_wrapper_t<raft::device_span<T>>;

}  // namespace detail
}  // namespace mtmg
}  // namespace CUGRAPH_EXPORT cugraph
