/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "approx_weighted_matching_impl.cuh"

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, float> approximate_weighted_matching(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template std::tuple<rmm::device_uvector<int64_t>, double> approximate_weighted_matching(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

}  // namespace cugraph
