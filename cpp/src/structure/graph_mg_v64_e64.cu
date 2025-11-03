/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_impl.cuh"

namespace cugraph {

// MG instantiation

template class graph_t<int64_t, int64_t, true, true>;
template class graph_t<int64_t, int64_t, false, true>;

}  // namespace cugraph
