/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/graph_helper_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {
namespace c_api {

template CUGRAPH_EXPORT edge_property_t<int32_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  float constant_value);

template CUGRAPH_EXPORT edge_property_t<int64_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  float constant_value);

template CUGRAPH_EXPORT edge_property_t<int32_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  float constant_value);

template CUGRAPH_EXPORT edge_property_t<int64_t, float> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  float constant_value);

template CUGRAPH_EXPORT edge_property_t<int32_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  double constant_value);

template CUGRAPH_EXPORT edge_property_t<int64_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  double constant_value);

template CUGRAPH_EXPORT edge_property_t<int32_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  double constant_value);

template CUGRAPH_EXPORT edge_property_t<int64_t, double> create_constant_edge_property(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  double constant_value);

}  // namespace c_api
}  // namespace cugraph
