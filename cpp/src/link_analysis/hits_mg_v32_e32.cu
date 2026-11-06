/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "link_analysis/hits_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation
template CUGRAPH_EXPORT std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  float* hubs,
  float* authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  double* hubs,
  double* authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, float>,
                                   vertex_property_t<int32_t, float>,
                                   iterative_algorithm_metadata_t>
hits(raft::handle_t const& handle,
     graph_view_t<int32_t, int32_t, true, true> const& graph_view,
     std::optional<vertex_property_t<int32_t, float>>&& initial_hubs,
     float epsilon,
     size_t max_iterations,
     bool normalize,
     bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<vertex_property_t<int32_t, double>,
                                   vertex_property_t<int32_t, double>,
                                   iterative_algorithm_metadata_t>
hits(raft::handle_t const& handle,
     graph_view_t<int32_t, int32_t, true, true> const& graph_view,
     std::optional<vertex_property_t<int32_t, double>>&& initial_hubs,
     double epsilon,
     size_t max_iterations,
     bool normalize,
     bool do_expensive_check);

}  // namespace cugraph
