/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "structure/graph_utils_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation

// compute_in_degrees

template CUGRAPH_EXPORT vertex_property_t<int32_t, int32_t> compute_in_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, int32_t> compute_in_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  bool do_expensive_check);

// compute_out_degrees

template CUGRAPH_EXPORT vertex_property_t<int32_t, int32_t> compute_out_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, int32_t> compute_out_degrees(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  bool do_expensive_check);

// compute_in_weight_sums

template CUGRAPH_EXPORT rmm::device_uvector<float>
compute_in_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<float>
compute_in_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<double>
compute_in_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<double>
compute_in_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
compute_in_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
compute_in_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
compute_in_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
compute_in_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT rmm::device_uvector<float>
compute_out_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<float>
compute_out_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<double>
compute_out_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template CUGRAPH_EXPORT rmm::device_uvector<double>
compute_out_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
compute_out_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, float>
compute_out_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
compute_out_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT vertex_property_t<int32_t, double>
compute_out_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_max_in_weight_sum<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_max_in_weight_sum<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_max_in_weight_sum<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_max_in_weight_sum<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_max_out_weight_sum<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_max_out_weight_sum<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_max_out_weight_sum<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_max_out_weight_sum<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_total_edge_weight<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT float compute_total_edge_weight<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_total_edge_weight<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

template CUGRAPH_EXPORT double compute_total_edge_weight<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view,
  bool do_expensive_check);

}  // namespace cugraph
