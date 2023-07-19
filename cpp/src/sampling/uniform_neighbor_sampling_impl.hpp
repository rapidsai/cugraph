/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#pragma once

#include <sampling/detail/sampling_utils.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
uniform_neighbor_sample_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> this_frontier_vertices,
  std::optional<raft::device_span<label_t const>> this_frontier_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  raft::random::RngState& rng_state,
  bool do_expensive_check)
{
#ifdef NO_CUGRAPH_OPS
  CUGRAPH_FAIL(
    "uniform_neighbor_sample_impl not supported in this configuration, built with NO_CUGRAPH_OPS");
#else
  CUGRAPH_EXPECTS(fan_out.size() > 0, "Invalid input argument: number of levels must be non-zero.");
  CUGRAPH_EXPECTS(
    fan_out.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                              // to store hops

  if constexpr (!multi_gpu) {
    CUGRAPH_EXPECTS(!label_to_output_comm_rank,
                    "cannot specify output GPU mapping in SG implementation");
  }

  CUGRAPH_EXPECTS(
    !label_to_output_comm_rank || this_frontier_vertex_labels,
    "cannot specify output GPU mapping without also specifying this_frontier_vertex_labels");

  if (do_expensive_check) {
    if (label_to_output_comm_rank) {
      CUGRAPH_EXPECTS(cugraph::detail::is_sorted(handle, std::get<0>(*label_to_output_comm_rank)),
                      "Labels in label_to_output_comm_rank must be sorted");
    }
  }

  std::vector<rmm::device_uvector<vertex_t>> level_result_src_vectors{};
  std::vector<rmm::device_uvector<vertex_t>> level_result_dst_vectors{};
  auto level_result_weight_vectors =
    edge_weight_view ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                     : std::nullopt;
  auto level_result_edge_id_vectors =
    edge_id_view ? std::make_optional(std::vector<rmm::device_uvector<edge_t>>{}) : std::nullopt;
  auto level_result_edge_type_vectors =
    edge_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>{})
                   : std::nullopt;
  auto level_result_label_vectors =
    this_frontier_vertex_labels ? std::make_optional(std::vector<rmm::device_uvector<label_t>>{})
                                : std::nullopt;

  level_result_src_vectors.reserve(fan_out.size());
  level_result_dst_vectors.reserve(fan_out.size());
  if (level_result_weight_vectors) { (*level_result_weight_vectors).reserve(fan_out.size()); }
  if (level_result_edge_id_vectors) { (*level_result_edge_id_vectors).reserve(fan_out.size()); }
  if (level_result_edge_type_vectors) { (*level_result_edge_type_vectors).reserve(fan_out.size()); }
  if (level_result_label_vectors) { (*level_result_label_vectors).reserve(fan_out.size()); }

  rmm::device_uvector<vertex_t> frontier_vertices(0, handle.get_stream());
  auto frontier_vertex_labels =
    this_frontier_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
      : std::nullopt;

  std::optional<
    std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<label_t>>>>
    vertex_used_as_source{std::nullopt};

  if (prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(
      std::make_tuple(rmm::device_uvector<vertex_t>{0, handle.get_stream()},
                      this_frontier_vertex_labels
                        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
                        : std::nullopt));
  }

  std::vector<size_t> level_sizes{};
  int32_t hop{0};
  for (auto&& k_level : fan_out) {
    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
    std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

    if (k_level > 0) {
      std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) =
        sample_edges(handle,
                     graph_view,
                     edge_weight_view,
                     edge_id_view,
                     edge_type_view,
                     rng_state,
                     this_frontier_vertices,
                     this_frontier_vertex_labels,
                     static_cast<size_t>(k_level),
                     with_replacement);
    } else {
      std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) =
        gather_one_hop_edgelist(handle,
                                graph_view,
                                edge_weight_view,
                                edge_id_view,
                                edge_type_view,
                                this_frontier_vertices,
                                this_frontier_vertex_labels);
    }

    level_sizes.push_back(srcs.size());

    level_result_src_vectors.push_back(std::move(srcs));
    level_result_dst_vectors.push_back(std::move(dsts));
    if (weights) { (*level_result_weight_vectors).push_back(std::move(*weights)); }
    if (edge_ids) { (*level_result_edge_id_vectors).push_back(std::move(*edge_ids)); }
    if (edge_types) { (*level_result_edge_type_vectors).push_back(std::move(*edge_types)); }
    if (labels) { (*level_result_label_vectors).push_back(std::move(*labels)); }

    ++hop;
    if (hop < fan_out.size()) {
      // FIXME:  We should modify vertex_partition_range_lasts to return a raft::host_span
      //  rather than making a copy.
      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
      std::tie(frontier_vertices, frontier_vertex_labels, vertex_used_as_source) =
        prepare_next_frontier(
          handle,
          this_frontier_vertices,
          this_frontier_vertex_labels,
          raft::device_span<vertex_t const>{level_result_dst_vectors.back().data(),
                                            level_result_dst_vectors.back().size()},
          frontier_vertex_labels ? std::make_optional(raft::device_span<label_t const>(
                                     level_result_label_vectors->back().data(),
                                     level_result_label_vectors->back().size()))
                                 : std::nullopt,
          std::move(vertex_used_as_source),
          graph_view.local_vertex_partition_view(),
          vertex_partition_range_lasts,
          prior_sources_behavior,
          dedupe_sources,
          do_expensive_check);

      this_frontier_vertices =
        raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size());

      if (frontier_vertex_labels) {
        this_frontier_vertex_labels = raft::device_span<label_t const>(
          frontier_vertex_labels->data(), frontier_vertex_labels->size());
      }
    }
  }

  auto result_size = std::reduce(level_sizes.begin(), level_sizes.end());
  size_t output_offset{};

  rmm::device_uvector<vertex_t> result_srcs(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_src_vectors.size(); ++i) {
    raft::copy(result_srcs.begin() + output_offset,
               level_result_src_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_src_vectors.clear();
  level_result_src_vectors.shrink_to_fit();

  rmm::device_uvector<vertex_t> result_dsts(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < level_result_dst_vectors.size(); ++i) {
    raft::copy(result_dsts.begin() + output_offset,
               level_result_dst_vectors[i].begin(),
               level_sizes[i],
               handle.get_stream());
    output_offset += level_sizes[i];
  }
  level_result_dst_vectors.clear();
  level_result_dst_vectors.shrink_to_fit();

  auto result_weights =
    level_result_weight_vectors
      ? std::make_optional(rmm::device_uvector<weight_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_weights) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_weight_vectors).size(); ++i) {
      raft::copy((*result_weights).begin() + output_offset,
                 (*level_result_weight_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_weight_vectors = std::nullopt;
  }

  auto result_edge_ids =
    level_result_edge_id_vectors
      ? std::make_optional(rmm::device_uvector<edge_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_ids) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_id_vectors).size(); ++i) {
      raft::copy((*result_edge_ids).begin() + output_offset,
                 (*level_result_edge_id_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_id_vectors = std::nullopt;
  }

  auto result_edge_types =
    level_result_edge_type_vectors
      ? std::make_optional(rmm::device_uvector<edge_type_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_types) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_edge_type_vectors).size(); ++i) {
      raft::copy((*result_edge_types).begin() + output_offset,
                 (*level_result_edge_type_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_edge_type_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};

  if (return_hops) {
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < fan_out.size(); ++i) {
      scalar_fill(
        handle, result_hops->data() + output_offset, level_sizes[i], static_cast<int32_t>(i));
      output_offset += level_sizes[i];
    }
  }

  auto result_labels =
    level_result_label_vectors
      ? std::make_optional(rmm::device_uvector<label_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_labels) {
    output_offset = 0;
    for (size_t i = 0; i < (*level_result_label_vectors).size(); ++i) {
      raft::copy((*result_labels).begin() + output_offset,
                 (*level_result_label_vectors)[i].begin(),
                 level_sizes[i],
                 handle.get_stream());
      output_offset += level_sizes[i];
    }
    level_result_label_vectors = std::nullopt;
  }

  return detail::shuffle_and_organize_output(handle,
                                             std::move(result_srcs),
                                             std::move(result_dsts),
                                             std::move(result_weights),
                                             std::move(result_edge_ids),
                                             std::move(result_edge_types),
                                             std::move(result_hops),
                                             std::move(result_labels),
                                             label_to_output_comm_rank);
#endif
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::uniform_neighbor_sample_impl(handle,
                                              graph_view,
                                              edge_weight_view,
                                              edge_id_view,
                                              edge_type_view,
                                              starting_vertices,
                                              starting_vertex_labels,
                                              label_to_output_comm_rank,
                                              fan_out,
                                              return_hops,
                                              with_replacement,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              rng_state,
                                              do_expensive_check);
}

}  // namespace cugraph
