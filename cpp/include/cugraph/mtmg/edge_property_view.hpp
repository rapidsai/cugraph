/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_wrapper.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief Edge property object for each GPU
 */
template <typename edge_t>
using edge_property_view_t =
  detail::device_shared_wrapper_t<cugraph::edge_arithmetic_property_view_t<edge_t>>;

}  // namespace mtmg
}  // namespace cugraph
