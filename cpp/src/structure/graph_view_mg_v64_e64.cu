/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_view_impl.cuh"

namespace cugraph {

// MG instantiation

template class graph_view_t<int64_t, int64_t, true, true>;
template class graph_view_t<int64_t, int64_t, false, true>;

}  // namespace cugraph
