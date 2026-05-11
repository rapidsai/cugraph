/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "link_analysis/hits_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// MG instantiation
template CUGRAPH_EXPORT std::tuple<float, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  float* const hubs,
  float* const authorities,
  float epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

template CUGRAPH_EXPORT std::tuple<double, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  double* const hubs,
  double* const authorities,
  double epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check);

}  // namespace cugraph
