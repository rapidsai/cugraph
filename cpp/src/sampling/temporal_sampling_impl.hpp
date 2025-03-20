/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cugraph/utilities/error.hpp"
#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

// #include <limits>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename bias_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
temporal_neighbor_sample_impl(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  std::optional<edge_type_t> num_edge_types,  // valid if heterogeneous sampling
  bool return_hops,
  bool with_replacement,
  prior_sources_behavior_t prior_sources_behavior,
  bool dedupe_sources,
  bool do_expensive_check)
{
  static_assert(std::is_floating_point_v<bias_t>);
  static_assert(std::is_same_v<bias_t, weight_t>);

  if constexpr (!multi_gpu) {
    CUGRAPH_EXPECTS(!label_to_output_comm_rank,
                    "cannot specify output GPU mapping in SG implementation");
  }

  std::cout << "temporal_neighbor_sample_impl" << std::endl;

  CUGRAPH_EXPECTS(
    !label_to_output_comm_rank || starting_vertex_labels,
    "cannot specify output GPU mapping without also specifying starting_vertex_labels");

  if (do_expensive_check) {
    if (edge_bias_view) {
      auto [num_negative_edge_weights, num_overflows] =
        check_edge_bias_values(handle, graph_view, *edge_bias_view);

      CUGRAPH_EXPECTS(
        num_negative_edge_weights == 0,
        "Invalid input argument: input edge bias values should have non-negative values.");
      CUGRAPH_EXPECTS(num_overflows == 0,
                      "Invalid input argument: sum of neighboring edge bias values should not "
                      "exceed std::numeric_limits<bias_t>::max() for any vertex.");
    }

#if 0
    // Do we have a function to do this?  Seems like a common validation check
    auto vertex_partition =
      vertex_partition_device_view_t<vertex_t, multi_gpu>(graph_view.local_vertex_partition_view());
    auto num_invalid_vertices =
      thrust::count_if(handle.get_thrust_policy(),
                       starting_vertices.begin(),
                       starting_vertices.end(),
                       [vertex_partition] __device__(auto v) {
                         return !(vertex_partition.is_valid_vertex(v) &&
                                  vertex_partition.in_local_vertex_partition_range_nocheck(v));
                       });
    if constexpr (multi_gpu) {
      num_invalid_vertices = cugraph::host_scalar_allreduce(
        handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
    }

    CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                    "Invalid input arguments: there are invalid input vertices.");
#endif
  }

  CUGRAPH_EXPECTS(fan_out.size() > 0, "Invalid input argument: number of levels must be non-zero.");
  CUGRAPH_EXPECTS(
    fan_out.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                              // to store hops

  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> temporal_graph_view = graph_view;
  edge_time_t STARTING_TIME{std::numeric_limits<edge_time_t>::min()};

  // Get the number of hop.
  auto num_hops = raft::div_rounding_up_safe(
    fan_out.size(), static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));

  std::vector<rmm::device_uvector<vertex_t>> result_src_vectors{};
  std::vector<rmm::device_uvector<vertex_t>> result_dst_vectors{};

  auto result_weight_vectors = edge_weight_view
                                 ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                                 : std::nullopt;
  auto result_edge_id_vectors =
    edge_id_view ? std::make_optional(std::vector<rmm::device_uvector<edge_t>>{}) : std::nullopt;
  auto result_edge_type_vectors =
    edge_type_view ? std::make_optional(std::vector<rmm::device_uvector<edge_type_t>>{})
                   : std::nullopt;
  auto result_edge_start_time_vectors =
    std::make_optional(std::vector<rmm::device_uvector<edge_time_t>>{});
  auto result_edge_end_time_vectors =
    edge_end_time_view ? std::make_optional(std::vector<rmm::device_uvector<edge_time_t>>{})
                       : std::nullopt;
  auto result_label_vectors = starting_vertex_labels
                                ? std::make_optional(std::vector<rmm::device_uvector<label_t>>{})
                                : std::nullopt;

  result_src_vectors.reserve(num_hops);
  result_dst_vectors.reserve(num_hops);

  if (result_weight_vectors) { (*result_weight_vectors).reserve(num_hops); }
  if (result_edge_id_vectors) { (*result_edge_id_vectors).reserve(num_hops); }
  if (result_edge_type_vectors) { (*result_edge_type_vectors).reserve(num_hops); }
  if (result_edge_start_time_vectors) { (*result_edge_start_time_vectors).reserve(num_hops); }
  if (result_edge_end_time_vectors) { (*result_edge_end_time_vectors).reserve(num_hops); }
  if (result_label_vectors) {
    std::cout << "allocated result_label_vectors" << std::endl;
    (*result_label_vectors).reserve(num_hops);
  }

  rmm::device_uvector<vertex_t> frontier_vertices(0, handle.get_stream());
  auto frontier_vertex_times = std::make_optional<rmm::device_uvector<edge_time_t>>(
    starting_vertices.size(), handle.get_stream());

  auto frontier_vertex_labels =
    starting_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
      : std::nullopt;

  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>,
                           std::optional<rmm::device_uvector<edge_time_t>>>>
    vertex_used_as_source{std::nullopt};

  if (prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(std::make_tuple(
      rmm::device_uvector<vertex_t>{0, handle.get_stream()},
      starting_vertex_labels
        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
        : std::nullopt,
      std::make_optional<rmm::device_uvector<edge_time_t>>(0, handle.get_stream())));
  }

  std::vector<size_t> result_sizes{};

  cugraph::edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, bool>
    edge_time_mask(handle, graph_view);

  cugraph::fill_edge_property(
    handle, temporal_graph_view, edge_time_mask.mutable_view(), bool{true});
  temporal_graph_view.attach_edge_mask(edge_time_mask.view());

  edge_src_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, edge_time_t>
    edge_src_times(handle, graph_view);
  fill_edge_src_property(handle, graph_view, edge_src_times.mutable_view(), STARTING_TIME);

  raft::device_span<vertex_t const> frontier_vertices_view{starting_vertices.data(),
                                                           starting_vertices.size()};
  raft::device_span<edge_time_t const> frontier_vertex_times_view{nullptr, size_t{0}};

  std::optional<raft::device_span<label_t const>> frontier_vertex_labels_view{
    starting_vertex_labels};
  rmm::device_uvector<vertex_t> frontier_vertices_current_subset(0, handle.get_stream());
  rmm::device_uvector<edge_time_t> frontier_vertex_times_current_subset(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> frontier_vertex_labels_current_subset{std::nullopt};

  for (size_t hop = 0; hop < num_hops; ++hop) {
    std::cout << "hop loop, hop = " << hop << std::endl;

    std::optional<std::vector<size_t>> level_Ks{std::nullopt};
    std::optional<std::vector<uint8_t>> gather_flags{std::nullopt};
    std::vector<raft::device_span<vertex_t const>> next_frontier_vertex_spans;
    auto next_frontier_vertex_label_spans =
      starting_vertex_labels ? std::make_optional<std::vector<raft::device_span<label_t const>>>()
                             : std::nullopt;
    auto next_frontier_vertex_time_spans =
      std::make_optional<std::vector<raft::device_span<edge_time_t const>>>();

    auto start_offset = hop * (num_edge_types ? *num_edge_types : edge_type_t{1});
    auto end_offset =
      start_offset +
      std::min(static_cast<size_t>((num_edge_types ? *num_edge_types : edge_type_t{1})),
               fan_out.size() -
                 hop * static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));
    for (size_t i = start_offset; i < end_offset; ++i) {
      if (fan_out[i] > 0) {
        if (!level_Ks) {
          level_Ks = std::vector<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}, 0);
        }
        (*level_Ks)[i - start_offset] = fan_out[i];
      } else if (fan_out[i] < 0) {
        if (!gather_flags) {
          gather_flags = std::vector<uint8_t>(num_edge_types ? *num_edge_types : edge_type_t{1},
                                              static_cast<uint8_t>(false));
        }
        (*gather_flags)[i - start_offset] = static_cast<uint8_t>(true);
      }
    }

    std::cout << "frontier_vertices_view_size: " << frontier_vertices_view.size() << std::endl;
    while (frontier_vertices_view.size() > 0) {
      rmm::device_uvector<vertex_t> frontier_vertices_next_subset(0, handle.get_stream());
      rmm::device_uvector<edge_time_t> frontier_vertex_times_next_subset(0, handle.get_stream());
      std::optional<rmm::device_uvector<label_t>> frontier_vertex_labels_next_subset{std::nullopt};

      if (hop > 0) {
        // It's possible for a vertex to appear in the frontier multiple times with different time
        // stamps (after the first hop).  To handle that situation, we need to partition the vertex
        // set.
        std::tie(frontier_vertices_current_subset,
                 frontier_vertex_times_current_subset,
                 frontier_vertex_labels_current_subset,
                 frontier_vertices_next_subset,
                 frontier_vertex_times_next_subset,
                 frontier_vertex_labels_next_subset) =
          temporal_partition_vertices(handle,
                                      frontier_vertices_view,
                                      frontier_vertex_times_view,
                                      frontier_vertex_labels_view);

        frontier_vertices_view = raft::device_span<vertex_t const>{
          frontier_vertices_current_subset.data(), frontier_vertices_current_subset.size()};
        if (frontier_vertex_labels_current_subset)
          frontier_vertex_labels_view =
            raft::device_span<label_t const>{frontier_vertex_labels_current_subset->data(),
                                             frontier_vertex_labels_current_subset->size()};

        fill_edge_src_property(handle, graph_view, edge_src_times.mutable_view(), STARTING_TIME);
        update_edge_src_property(handle,
                                 graph_view,
                                 frontier_vertices_view.begin(),
                                 frontier_vertices_view.end(),
                                 frontier_vertex_times_current_subset.begin(),
                                 edge_src_times.mutable_view());

        // FIXME: Need to use transform_e, would be more efficient to pass in vertex frontier,
        //   but that doesn't exist today
        // NOTE: using graph_view as input here since to avoid read/write conflicts on edge mask
        cugraph::transform_e(
          handle,
          graph_view,
          edge_src_times.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          edge_start_time_view,
          [] __device__(auto src, auto dst, auto src_time, auto, auto edge_start_time) {
            return edge_start_time > src_time;
          },
          edge_time_mask.mutable_view(),
          false);
      }

      if (level_Ks) {
        std::cout << "calling sample_edges, frontier view size = " << frontier_vertices_view.size()
                  << std::endl;

        auto [srcs, dsts, weights, edge_ids, edge_types, edge_start_times, edge_end_times, labels] =
          sample_edges(handle,
                       temporal_graph_view,
                       edge_weight_view,
                       edge_id_view,
                       edge_type_view,
                       std::make_optional(edge_start_time_view),
                       edge_end_time_view,
                       edge_bias_view,
                       rng_state,
                       frontier_vertices_view,
                       frontier_vertex_labels_view,
                       raft::host_span<size_t const>(level_Ks->data(), level_Ks->size()),
                       with_replacement);

#if 0
        raft::print_device_vector("      frontier_vertices",
                                  frontier_vertices_view.data(),
                                  frontier_vertices_view.size(),
                                  std::cout);
        raft::print_device_vector("      srcs", srcs.data(), srcs.size(), std::cout);
        raft::print_device_vector("      dsts", dsts.data(), dsts.size(), std::cout);
        raft::print_device_vector(
          "      edge_start_times", edge_start_times->data(), edge_start_times->size(), std::cout);
        if (weights)
          raft::print_device_vector("      weights", weights->data(), weights->size(), std::cout);
        if (edge_ids)
          raft::print_device_vector(
            "      edge_ids", edge_ids->data(), edge_ids->size(), std::cout);
        if (edge_types)
          raft::print_device_vector(
            "      edge_types", edge_types->data(), edge_types->size(), std::cout);
        if (edge_end_times)
          raft::print_device_vector(
            "      edge_end_times", edge_end_times->data(), edge_end_times->size(), std::cout);
        if (labels)
          raft::print_device_vector("      labels", labels->data(), labels->size(), std::cout);

#endif

        result_sizes.push_back(srcs.size());
        result_src_vectors.push_back(std::move(srcs));
        result_dst_vectors.push_back(std::move(dsts));

        if (weights) { (*result_weight_vectors).push_back(std::move(*weights)); }
        if (edge_ids) { (*result_edge_id_vectors).push_back(std::move(*edge_ids)); }
        if (edge_types) { (*result_edge_type_vectors).push_back(std::move(*edge_types)); }
        if (edge_start_times) {
          (*result_edge_start_time_vectors).push_back(std::move(*edge_start_times));
        }
        if (edge_end_times) {
          (*result_edge_end_time_vectors).push_back(std::move(*edge_end_times));
        }
        if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }

        next_frontier_vertex_spans.push_back(raft::device_span<vertex_t const>{
          result_dst_vectors.back().data(), result_dst_vectors.back().size()});
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<edge_time_t const>{result_edge_start_time_vectors->back().data(),
                                               result_edge_start_time_vectors->back().size()});
        if (next_frontier_vertex_label_spans)
          next_frontier_vertex_label_spans->push_back(raft::device_span<label_t const>{
            result_label_vectors->back().data(), result_label_vectors->back().size()});
      }

      if (gather_flags) {
        auto [srcs, dsts, weights, edge_ids, edge_types, edge_start_times, edge_end_times, labels] =
          gather_one_hop_edgelist(handle,
                                  temporal_graph_view,
                                  edge_weight_view,
                                  edge_id_view,
                                  edge_type_view,
                                  std::make_optional(edge_start_time_view),
                                  edge_end_time_view,
                                  frontier_vertices_view,
                                  frontier_vertex_labels_view,
                                  num_edge_types
                                    ? std::make_optional<raft::host_span<uint8_t const>>(
                                        gather_flags->data(), gather_flags->size())
                                    : std::nullopt);

        result_sizes.push_back(srcs.size());
        result_src_vectors.push_back(std::move(srcs));
        result_dst_vectors.push_back(std::move(dsts));

        if (weights) { (*result_weight_vectors).push_back(std::move(*weights)); }
        if (edge_ids) { (*result_edge_id_vectors).push_back(std::move(*edge_ids)); }
        if (edge_types) { (*result_edge_type_vectors).push_back(std::move(*edge_types)); }
        if (edge_start_times) {
          (*result_edge_start_time_vectors).push_back(std::move(*edge_start_times));
        }
        if (edge_end_times) {
          (*result_edge_end_time_vectors).push_back(std::move(*edge_end_times));
        }
        if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }

        next_frontier_vertex_spans.push_back(raft::device_span<vertex_t const>{
          result_dst_vectors.back().data(), result_dst_vectors.back().size()});
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<edge_time_t const>{result_edge_start_time_vectors->back().data(),
                                               result_edge_start_time_vectors->back().size()});
        if (next_frontier_vertex_label_spans)
          next_frontier_vertex_label_spans->push_back(raft::device_span<label_t const>{
            result_label_vectors->back().data(), result_label_vectors->back().size()});
      }

      frontier_vertices_current_subset      = std::move(frontier_vertices_next_subset);
      frontier_vertex_labels_current_subset = std::move(frontier_vertex_labels_next_subset);
      frontier_vertex_times_current_subset  = std::move(frontier_vertex_times_next_subset);

      frontier_vertices_view = raft::device_span<vertex_t const>{
        frontier_vertices_current_subset.data(), frontier_vertices_current_subset.size()};
      if (frontier_vertex_labels)
        frontier_vertex_labels_view =
          raft::device_span<label_t const>{frontier_vertex_labels_current_subset->begin(),
                                           frontier_vertex_labels_current_subset->size()};
      frontier_vertex_times_view = raft::device_span<edge_time_t const>{
        frontier_vertex_times_current_subset.begin(), frontier_vertex_times_current_subset.size()};
    }

    std::cout << "calling prepare_next_frontier" << std::endl;

    std::tie(
      frontier_vertices, frontier_vertex_labels, frontier_vertex_times, vertex_used_as_source) =
      prepare_next_frontier(
        handle,
        hop == 0
          ? starting_vertices
          : raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
        hop == 0 ? starting_vertex_labels
        : starting_vertex_labels
          ? std::make_optional(raft::device_span<label_t const>(frontier_vertex_labels->data(),
                                                                frontier_vertex_labels->size()))
          : std::nullopt,
        std::make_optional<raft::device_span<edge_time_t const>>(frontier_vertex_times->data(),
                                                                 frontier_vertex_times->size()),
        raft::host_span<raft::device_span<vertex_t const>>{next_frontier_vertex_spans.data(),
                                                           next_frontier_vertex_spans.size()},
        next_frontier_vertex_label_spans
          ? std::make_optional(raft::host_span<raft::device_span<label_t const>>{
              next_frontier_vertex_label_spans->data(), next_frontier_vertex_label_spans->size()})
          : std::nullopt,
        std::make_optional(raft::host_span<raft::device_span<edge_time_t const>>{
          next_frontier_vertex_time_spans->data(), next_frontier_vertex_time_spans->size()}),
        std::move(vertex_used_as_source),
        graph_view.local_vertex_partition_view(),
        graph_view.vertex_partition_range_lasts(),
        prior_sources_behavior,
        dedupe_sources,
        do_expensive_check);

    frontier_vertices_view =
      raft::device_span<vertex_t const>{frontier_vertices.data(), frontier_vertices.size()};
    frontier_vertex_times_view = raft::device_span<edge_time_t const>{
      frontier_vertex_times->data(), frontier_vertex_times->size()};
    if (frontier_vertex_labels)
      frontier_vertex_labels_view = raft::device_span<label_t const>{
        frontier_vertex_labels->data(), frontier_vertex_labels->size()};

#if 0
        raft::print_device_vector("      frontier_vertices",
          frontier_vertices.data(),
          frontier_vertices.size(),
          std::cout);
          raft::print_device_vector("      frontier_vertex_times",
            frontier_vertex_times->data(),
            frontier_vertex_times->size(),
            std::cout);
#endif
  }

  auto result_size = std::reduce(result_sizes.begin(), result_sizes.end());
  size_t output_offset{};

  rmm::device_uvector<vertex_t> result_srcs(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < result_src_vectors.size(); ++i) {
    raft::copy(result_srcs.begin() + output_offset,
               result_src_vectors[i].begin(),
               result_sizes[i],
               handle.get_stream());
    output_offset += result_sizes[i];
  }
  result_src_vectors.clear();
  result_src_vectors.shrink_to_fit();

  rmm::device_uvector<vertex_t> result_dsts(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < result_dst_vectors.size(); ++i) {
    raft::copy(result_dsts.begin() + output_offset,
               result_dst_vectors[i].begin(),
               result_sizes[i],
               handle.get_stream());
    output_offset += result_sizes[i];
  }
  result_dst_vectors.clear();
  result_dst_vectors.shrink_to_fit();

  auto result_weights =
    result_weight_vectors
      ? std::make_optional(rmm::device_uvector<weight_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_weights) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_weight_vectors).size(); ++i) {
      raft::copy((*result_weights).begin() + output_offset,
                 (*result_weight_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_weight_vectors = std::nullopt;
  }

  auto result_edge_ids =
    result_edge_id_vectors
      ? std::make_optional(rmm::device_uvector<edge_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_ids) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_edge_id_vectors).size(); ++i) {
      raft::copy((*result_edge_ids).begin() + output_offset,
                 (*result_edge_id_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_edge_id_vectors = std::nullopt;
  }

  auto result_edge_types =
    result_edge_type_vectors
      ? std::make_optional(rmm::device_uvector<edge_type_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_types) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_edge_type_vectors).size(); ++i) {
      raft::copy((*result_edge_types).begin() + output_offset,
                 (*result_edge_type_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_edge_type_vectors = std::nullopt;
  }

  auto result_edge_start_times =
    result_edge_start_time_vectors
      ? std::make_optional(rmm::device_uvector<edge_time_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_start_times) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_edge_start_time_vectors).size(); ++i) {
      raft::copy((*result_edge_start_times).begin() + output_offset,
                 (*result_edge_start_time_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_edge_start_time_vectors = std::nullopt;
  }

  auto result_edge_end_times =
    result_edge_end_time_vectors
      ? std::make_optional(rmm::device_uvector<edge_time_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_edge_end_times) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_edge_end_time_vectors).size(); ++i) {
      raft::copy((*result_edge_end_times).begin() + output_offset,
                 (*result_edge_end_time_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_edge_end_time_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};

  if (return_hops) {
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < num_hops; ++i) {
      scalar_fill(
        handle, result_hops->data() + output_offset, result_sizes[i], static_cast<int32_t>(i));
      output_offset += result_sizes[i];
    }
  }

  auto result_labels =
    result_label_vectors
      ? std::make_optional(rmm::device_uvector<label_t>(result_size, handle.get_stream()))
      : std::nullopt;
  if (result_labels) {
    output_offset = 0;
    for (size_t i = 0; i < (*result_label_vectors).size(); ++i) {
      raft::copy((*result_labels).begin() + output_offset,
                 (*result_label_vectors)[i].begin(),
                 result_sizes[i],
                 handle.get_stream());
      output_offset += result_sizes[i];
    }
    result_label_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<size_t>> result_offsets{std::nullopt};

  std::tie(result_srcs,
           result_dsts,
           result_weights,
           result_edge_ids,
           result_edge_types,
           result_edge_start_times,
           result_edge_end_times,
           result_hops,
           result_labels,
           result_offsets) = detail::shuffle_and_organize_output(handle,
                                                                 std::move(result_srcs),
                                                                 std::move(result_dsts),
                                                                 std::move(result_weights),
                                                                 std::move(result_edge_ids),
                                                                 std::move(result_edge_types),
                                                                 std::move(result_edge_start_times),
                                                                 std::move(result_edge_end_times),
                                                                 std::move(result_hops),
                                                                 std::move(result_labels),
                                                                 label_to_output_comm_rank);

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_weights),
                         std::move(result_edge_ids),
                         std::move(result_edge_types),
                         std::move(result_edge_start_times),
                         std::move(result_edge_end_times),
                         std::move(result_hops),
                         std::move(result_offsets));
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_uniform_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  using bias_t = weight_t;  // dummy

  return detail::
    temporal_neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      edge_type_view,
      edge_start_time_view,
      edge_end_time_view,
      std::optional<edge_property_view_t<edge_t, bias_t const*>>{
        std::nullopt},  // Optional edge_bias_view
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_uniform_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  using bias_t = weight_t;  // dummy

  return detail::
    temporal_neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      std::make_optional(edge_type_view),
      edge_start_time_view,
      edge_end_time_view,
      std::optional<edge_property_view_t<edge_t, bias_t const*>>{
        std::nullopt},  // Optional edge_bias_view
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_biased_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  return detail::
    temporal_neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      edge_type_view,
      edge_start_time_view,
      edge_end_time_view,
      std::make_optional(edge_bias_view),
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_biased_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  return detail::
    temporal_neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t, bias_t>(
      handle,
      rng_state,
      graph_view,
      edge_weight_view,
      edge_id_view,
      std::make_optional(edge_type_view),
      edge_start_time_view,
      edge_end_time_view,
      std::make_optional(edge_bias_view),
      starting_vertices,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags.return_hops,
      sampling_flags.with_replacement,
      sampling_flags.prior_sources_behavior,
      sampling_flags.dedupe_sources,
      do_expensive_check);
}

}  // namespace cugraph
