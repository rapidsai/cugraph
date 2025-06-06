/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "neighbor_sampling_impl.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/sampling_functions.hpp>

namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
biased_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  edge_property_view_t<int32_t, float const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
biased_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  edge_property_view_t<int32_t, double const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
heterogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  edge_property_view_t<int32_t, int32_t const*> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  int32_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
heterogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  edge_property_view_t<int32_t, int32_t const*> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  int32_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
heterogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  edge_property_view_t<int32_t, int32_t const*> edge_type_view,
  edge_property_view_t<int32_t, double const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  int32_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
heterogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  edge_property_view_t<int32_t, int32_t const*> edge_type_view,
  edge_property_view_t<int32_t, float const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  int32_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
homogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
homogeneous_uniform_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
homogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  edge_property_view_t<int32_t, double const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
homogeneous_biased_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_type_view,
  edge_property_view_t<int32_t, float const*> edge_bias_view,
  raft::device_span<int32_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check);

}  // namespace cugraph
