/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_view_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT class graph_view_t<int32_t, int32_t, true, false>;
template CUGRAPH_EXPORT class graph_view_t<int32_t, int32_t, false, false>;
}  // namespace cugraph
