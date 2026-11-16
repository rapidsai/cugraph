/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/leiden_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT leiden_result_t
leiden(raft::handle_t const& handle,
       raft::random::RngState& rng_state,
       graph_view_t<int64_t, int64_t, false, false> const& graph_view,
       std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
       raft::device_span<int64_t> clustering,
       leiden_params_t const& params,
       bool do_expensive_check);

template CUGRAPH_EXPORT leiden_result_t
leiden(raft::handle_t const& handle,
       raft::random::RngState& rng_state,
       graph_view_t<int64_t, int64_t, false, false> const& graph_view,
       std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,
       raft::device_span<int64_t> clustering,
       leiden_params_t const& params,
       bool do_expensive_check);

}  // namespace cugraph
