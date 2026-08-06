/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Built into libcugraph_common.so: sampling common TUs (gather_one_hop / sample_outgoing_edges)
// reference MG graph_view_t::local_edge_partition_* and must not leave those symbols undefined
// for SG consumers that do not load libcugraph_mg.so.

#include "structure/graph_view_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT class graph_view_t<int64_t, int64_t, true, true>;
template CUGRAPH_EXPORT class graph_view_t<int64_t, int64_t, false, true>;

}  // namespace cugraph
