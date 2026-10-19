/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/export.hpp>
#include <cugraph/mtmg/detail/device_shared_device_span.hpp>
#include <cugraph/mtmg/handle.hpp>

namespace CUGRAPH_EXPORT cugraph {
namespace mtmg {

/**
 * @brief An MTMG device span for storing a renumber map
 *
 * @deprecated This API is deprecated and will be removed in release 27.02.
 */
template <typename vertex_t>
using renumber_map_view_t = detail::device_shared_device_span_t<vertex_t const>;

}  // namespace mtmg
}  // namespace CUGRAPH_EXPORT cugraph
