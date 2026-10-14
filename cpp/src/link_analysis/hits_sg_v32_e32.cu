/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "link_analysis/hits_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation
template CUGRAPH_EXPORT std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  vertex_property_view_t<int32_t, float*> hubs,
  vertex_property_view_t<int32_t, float*> authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, false> const& graph_view,
  vertex_property_view_t<int32_t, double*> hubs,
  vertex_property_view_t<int32_t, double*> authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

}  // namespace cugraph
