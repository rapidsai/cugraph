/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "community/ecg_impl.cuh"

namespace cugraph {
template std::tuple<rmm::device_uvector<int32_t>, size_t, float> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,

  float min_weight,
  size_t ensemble_size,
  size_t max_level,
  float threshold,
  float resolution);

template std::tuple<rmm::device_uvector<int32_t>, size_t, double> ecg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, double const*>> edge_weight_view,

  double min_weight,
  size_t ensemble_size,
  size_t max_level,
  double threshold,
  double resolution);

}  // namespace cugraph
