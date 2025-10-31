/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_weight_utils_impl.cuh"

namespace cugraph {

// SG instantiation

// compute_in_weight_sums

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<double>
compute_out_weight_sums<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_max_in_weight_sum<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_max_in_weight_sum<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_max_out_weight_sum<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_max_out_weight_sum<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_total_edge_weight<int32_t, int32_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_total_edge_weight<int32_t, int32_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int32_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int32_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

}  // namespace cugraph
