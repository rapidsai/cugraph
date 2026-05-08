/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT class graph_t<int32_t, int32_t, true, true>;
template CUGRAPH_EXPORT class graph_t<int32_t, int32_t, false, true>;

}  // namespace cugraph
