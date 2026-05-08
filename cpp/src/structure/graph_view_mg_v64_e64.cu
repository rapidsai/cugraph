/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_view_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT class graph_view_t<int64_t, int64_t, true, true>;
template CUGRAPH_EXPORT class graph_view_t<int64_t, int64_t, false, true>;

}  // namespace cugraph
