/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/leiden_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

template CUGRAPH_EXPORT leiden_result_t
leiden(raft::handle_t const& handle,
       raft::random::RngState& rng_state,
       graph_view_t<int32_t, int32_t, false, true> const& graph_view,
       std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
       raft::device_span<int32_t> clustering,
       leiden_params_t const& params,
       bool do_expensive_check);

template CUGRAPH_EXPORT leiden_result_t
leiden(raft::handle_t const& handle,
       raft::random::RngState& rng_state,
       graph_view_t<int32_t, int32_t, false, true> const& graph_view,
       std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
       raft::device_span<int32_t> clustering,
       leiden_params_t const& params,
       bool do_expensive_check);

}  // namespace cugraph
