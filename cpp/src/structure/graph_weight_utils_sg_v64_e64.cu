/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_weight_utils_impl.cuh"

namespace cugraph {

// SG instantiation

// compute_in_weight_sums

template rmm::device_uvector<float> compute_in_weight_sums<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_out_weight_sums

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double>
compute_out_weight_sums<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_max_in_weight_sum

template float compute_max_in_weight_sum<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_in_weight_sum<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_in_weight_sum<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_in_weight_sum<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_max_out_weight_sum

template float compute_max_out_weight_sum<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_out_weight_sum<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_out_weight_sum<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_out_weight_sum<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_total_edge_weight

template float compute_total_edge_weight<int64_t, int64_t, float, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_total_edge_weight<int64_t, int64_t, float, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_total_edge_weight<int64_t, int64_t, double, false, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_total_edge_weight<int64_t, int64_t, double, true, false>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

}  // namespace cugraph
