/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/detail/device_shared_device_span.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {

/**
 * @brief An MTMG device span for storing a renumber map
 */
template <typename vertex_t>
using renumber_map_view_t = detail::device_shared_device_span_t<vertex_t const>;

}  // namespace mtmg
}  // namespace cugraph
