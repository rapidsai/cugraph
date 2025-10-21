/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "community/ecg_impl.cuh"

namespace cugraph {
template std::tuple<rmm::device_uvector<int64_t>, size_t, float> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,

  float min_weight,
  size_t ensemble_size,
  size_t max_level,
  float threshold,
  float resolution);

template std::tuple<rmm::device_uvector<int64_t>, size_t, double> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,

  double min_weight,
  size_t ensemble_size,
  size_t max_level,
  double threshold,
  double resolution);

}  // namespace cugraph
