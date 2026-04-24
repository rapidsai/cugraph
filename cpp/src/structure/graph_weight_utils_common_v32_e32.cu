/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_weight_utils_impl.cuh"

namespace cugraph {

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

}  // namespace cugraph
