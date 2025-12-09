/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>

#include <raft/core/device_span.hpp>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Manage a tuple of device spans on each GPU
 */
template <typename... Ts>
using device_shared_device_span_tuple_t =
  device_shared_wrapper_t<std::tuple<raft::device_span<Ts>...>>;

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph
