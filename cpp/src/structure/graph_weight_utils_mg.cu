/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <structure/graph_weight_utils_impl.cuh>

namespace cugraph {

// SG instantiation

// compute_in_weight_sums

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_in_weight_sums<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_in_weight_sums<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_out_weight_sums

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

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<float> compute_out_weight_sums<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template rmm::device_uvector<double> compute_out_weight_sums<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_max_in_weight_sum

template float compute_max_in_weight_sum<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_max_in_weight_sum<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_max_in_weight_sum<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_in_weight_sum<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_in_weight_sum<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template float compute_max_in_weight_sum<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_in_weight_sum<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_in_weight_sum<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_in_weight_sum<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_max_out_weight_sum

template float compute_max_out_weight_sum<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_max_out_weight_sum<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_max_out_weight_sum<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_out_weight_sum<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_out_weight_sum<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template float compute_max_out_weight_sum<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_max_out_weight_sum<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_max_out_weight_sum<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_max_out_weight_sum<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

// compute_total_edge_weight

template float compute_total_edge_weight<int32_t, int32_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template float compute_total_edge_weight<int32_t, int32_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, float const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int32_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int32_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, true, true> const& graph_view,
  edge_property_view_t<int32_t, double const*> edge_weight_view);

template float compute_total_edge_weight<int32_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_total_edge_weight<int32_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_total_edge_weight<int32_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template float compute_total_edge_weight<int64_t, int64_t, float, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template float compute_total_edge_weight<int64_t, int64_t, float, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_weight_view);

template double compute_total_edge_weight<int64_t, int64_t, double, false, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

template double compute_total_edge_weight<int64_t, int64_t, double, true, true>(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, true, true> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_weight_view);

}  // namespace cugraph
