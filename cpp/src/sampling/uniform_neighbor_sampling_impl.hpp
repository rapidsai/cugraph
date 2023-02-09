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

#include <sampling/detail/graph_functions.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<int32_t>>>
uniform_neighbor_sample_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  rmm::device_uvector<vertex_t>&& seed_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& seed_vertex_labels,
  raft::host_span<int32_t const> h_fan_out,
  bool with_replacement,
  raft::random::RngState& rng_state)
{
#ifdef NO_CUGRAPH_OPS
  CUGRAPH_FAIL(
    "uniform_neighbor_sample_impl not supported in this configuration, built with NO_CUGRAPH_OPS");
#else
  CUGRAPH_EXPECTS(h_fan_out.size() > 0,
                  "Invalid input argument: number of levels must be non-zero.");
  CUGRAPH_EXPECTS(
    h_fan_out.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                              // to store hops

  std::vector<rmm::device_uvector<vertex_t>> level_result_src_vectors{};
  std::vector<rmm::device_uvector<vertex_t>> level_result_dst_vectors{};
  auto level_result_weight_vectors =
    edge_weight_view ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                     : std::nullopt;
  auto level_result_edge_id_vectors =
    edge_id_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_t>>{})
                      : std::nullopt;
  auto level_result_edge_type_vectors =
    edge_id_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>{})
                      : std::nullopt;
  auto level_result_label_vectors =
    seed_vertex_labels ? std::make_optional(std::vector<rmm::device_uvector<int32_t>>{})
                       : std::nullopt;

  level_result_src_vectors.reserve(h_fan_out.size());
  level_result_dst_vectors.reserve(h_fan_out.size());
  if (level_result_weight_vectors) { (*level_result_weight_vectors).reserve(h_fan_out.size()); }
  if (level_result_edge_id_vectors) { (*level_result_edge_id_vectors).reserve(h_fan_out.size()); }
  if (level_result_edge_type_vectors) {
    (*level_result_edge_type_vectors).reserve(h_fan_out.size());
  }
  if (level_result_label_vectors) { (*level_result_label_vectors).reserve(h_fan_out.size()); }

  std::vector<size_t> level_sizes{};
  int32_t hop{0};
  for (auto&& k_level : h_fan_out) {
    // prep step for extracting out-degs(sources):
    if constexpr (multi_gpu) {
      if (seed_vertex_labels) {
        std::tie(seed_vertices, *seed_vertex_labels) =
          shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
            handle,
            std::move(seed_vertices),
            std::move(*seed_vertex_labels),
            graph_view.vertex_partition_range_lasts());
      } else {
        seed_vertices = shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
          handle, std::move(seed_vertices), graph_view.vertex_partition_range_lasts());
      }
    }

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
                     edge_id_type_view,
                     rng_state,
                     seed_vertices,
                     seed_vertex_labels,
                     static_cast<size_t>(k_level),
                     with_replacement);
    } else {
      std::tie(srcs, dsts, weights, edge_ids, edge_types, labels) = gather_one_hop_edgelist(
        handle, graph_view, edge_weight_view, edge_id_type_view, seed_vertices, seed_vertex_labels);
    }

    level_sizes.push_back(srcs.size());

    level_result_src_vectors.push_back(std::move(srcs));
    level_result_dst_vectors.push_back(std::move(dsts));
    if (weights) { (*level_result_weight_vectors).push_back(std::move(*weights)); }
    if (edge_ids) { (*level_result_edge_id_vectors).push_back(std::move(*edge_ids)); }
    if (edge_types) { (*level_result_edge_type_vectors).push_back(std::move(*edge_types)); }
    if (labels) { (*level_result_label_vectors).push_back(std::move(*labels)); }

    ++hop;
    if (hop < h_fan_out.size()) {
      seed_vertices.resize(level_sizes.back(), handle.get_stream());
      raft::copy(seed_vertices.data(),
                 level_result_dst_vectors.back().data(),
                 level_sizes.back(),
                 handle.get_stream());
      if (seed_vertex_labels) {
        (*seed_vertex_labels).resize(level_sizes.back(), handle.get_stream());
        raft::copy((*seed_vertex_labels).data(),
                   (*level_result_label_vectors).back().data(),
                   level_sizes.back(),
                   handle.get_stream());
      }
    }
  }

  seed_vertices.resize(0, handle.get_stream());
  seed_vertices.shrink_to_fit(handle.get_stream());
  if (seed_vertex_labels) { seed_vertex_labels = std::nullopt; }

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

  rmm::device_uvector<int32_t> result_hops(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < h_fan_out.size(); ++i) {
    scalar_fill(
      handle, result_hops.data() + output_offset, level_sizes[i], static_cast<int32_t>(i));
    output_offset += level_sizes[i];
  }

  auto result_labels =
    level_result_label_vectors
      ? std::make_optional(rmm::device_uvector<int32_t>(result_size, handle.get_stream()))
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

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_weights),
                         std::move(result_edge_ids),
                         std::move(result_edge_types),
                         std::move(result_hops),
                         std::move(result_labels));
#endif
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<int32_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  rmm::device_uvector<vertex_t>&& starting_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& starting_labels,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool with_replacement)
{
  return detail::uniform_neighbor_sample_impl(handle,
                                              graph_view,
                                              edge_weight_view,
                                              edge_id_type_view,
                                              std::move(starting_vertices),
                                              std::move(starting_labels),
                                              fan_out,
                                              with_replacement,
                                              rng_state);
}

}  // namespace cugraph
