/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_weight_utils_impl.cuh"

namespace cugraph {

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

}  // namespace cugraph
