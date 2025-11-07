/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/shuffle_wrappers.hpp"
#include "sampling/detail/sampling_utils.hpp"
#include "utilities/validation_checks.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cstddef>

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
  std::optional<raft::device_span<edge_time_t const>> starting_vertex_times,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  std::optional<edge_type_t> num_edge_types,  // valid if heterogeneous sampling
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  static_assert(std::is_floating_point_v<bias_t>);
  static_assert(std::is_same_v<bias_t, weight_t>);

  // FIXME: Add support for a graph_view that already has an edge mask
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(),
                  "Can't currently support a graph view with an existing edge mask");

  if constexpr (!multi_gpu) {
    CUGRAPH_EXPECTS(!label_to_output_comm_rank,
                    "cannot specify output GPU mapping in SG implementation");
  }

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

    CUGRAPH_EXPECTS(
      cugraph::count_invalid_vertices(
        handle,
        graph_view,
        raft::device_span<vertex_t const>{starting_vertices.data(), starting_vertices.size()}) == 0,
      "Invalid input arguments: there are invalid input vertices.");
  }

  CUGRAPH_EXPECTS(fan_out.size() > 0, "Invalid input argument: number of levels must be non-zero.");
  CUGRAPH_EXPECTS(
    fan_out.size() <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
    "Invalid input argument: number of levels should not overflow int32_t");  // as we use int32_t
                                                                              // to store hops

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
  if (result_label_vectors) { (*result_label_vectors).reserve(num_hops); }

  rmm::device_uvector<vertex_t> frontier_vertices(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_time_t>> frontier_vertex_times{std::nullopt};

  auto frontier_vertex_labels =
    starting_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
      : std::nullopt;

  if (starting_vertex_times) {
    frontier_vertices.resize(starting_vertices.size(), handle.get_stream());
    raft::copy(frontier_vertices.data(),
               starting_vertices.data(),
               starting_vertices.size(),
               handle.get_stream());
    frontier_vertex_times =
      rmm::device_uvector<edge_time_t>(starting_vertex_times->size(), handle.get_stream());
    raft::copy(frontier_vertex_times->data(),
               starting_vertex_times->data(),
               starting_vertex_times->size(),
               handle.get_stream());

    if (frontier_vertex_labels) {
      frontier_vertex_labels->resize(starting_vertex_labels->size(), handle.get_stream());
      raft::copy(frontier_vertex_labels->data(),
                 starting_vertex_labels->data(),
                 starting_vertex_labels->size(),
                 handle.get_stream());
    }
  }

  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>,
                           std::optional<rmm::device_uvector<edge_time_t>>>>
    vertex_used_as_source{std::nullopt};

  if (sampling_flags.prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(std::make_tuple(
      rmm::device_uvector<vertex_t>{0, handle.get_stream()},
      starting_vertex_labels
        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
        : std::nullopt,
      std::make_optional<rmm::device_uvector<edge_time_t>>(0, handle.get_stream())));
  }

  std::vector<size_t> result_vector_sizes{};
  std::vector<int32_t> result_vector_hops{};

  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> temporal_graph_view{graph_view};

  cugraph::edge_property_t<edge_t, bool> edge_time_mask(handle, graph_view);

  rmm::device_uvector<vertex_t> frontier_vertices_no_duplicates(0, handle.get_stream());
  rmm::device_uvector<edge_time_t> frontier_vertex_times_no_duplicates(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> frontier_vertex_labels_no_duplicates{std::nullopt};
  rmm::device_uvector<vertex_t> frontier_vertices_has_duplicates(0, handle.get_stream());
  rmm::device_uvector<edge_time_t> frontier_vertex_times_has_duplicates(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> frontier_vertex_labels_has_duplicates{std::nullopt};

  for (size_t hop = 0; hop < num_hops; ++hop) {
    std::optional<std::vector<size_t>> level_Ks{std::nullopt};
    std::optional<std::vector<uint8_t>> gather_flags{std::nullopt};
    std::vector<raft::device_span<vertex_t const>> next_frontier_vertex_spans{};
    auto next_frontier_vertex_label_spans =
      starting_vertex_labels ? std::make_optional<std::vector<raft::device_span<label_t const>>>()
                             : std::nullopt;
    auto next_frontier_vertex_time_spans =
      std::make_optional<std::vector<raft::device_span<edge_time_t const>>>();
    size_t no_duplicates_size{0};
    size_t has_duplicates_size{0};

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

    if (frontier_vertex_times) {
      // It's possible for a vertex to appear in the frontier multiple times with different time
      // stamps.  To handle that situation, we need to partition the vertex set.
      std::tie(frontier_vertices_no_duplicates,
               frontier_vertex_times_no_duplicates,
               frontier_vertex_labels_no_duplicates,
               frontier_vertices_has_duplicates,
               frontier_vertex_times_has_duplicates,
               frontier_vertex_labels_has_duplicates) =
        temporal_partition_vertices(
          handle,
          raft::device_span<vertex_t const>{frontier_vertices.data(), frontier_vertices.size()},
          raft::device_span<edge_time_t const>{frontier_vertex_times->data(),
                                               frontier_vertex_times->size()},
          frontier_vertex_labels
            ? std::make_optional(raft::device_span<label_t const>{frontier_vertex_labels->data(),
                                                                  frontier_vertex_labels->size()})
            : std::nullopt);

      no_duplicates_size  = frontier_vertices_no_duplicates.size();
      has_duplicates_size = frontier_vertices_has_duplicates.size();

      if constexpr (multi_gpu) {
        no_duplicates_size = host_scalar_allreduce(
          handle.get_comms(), no_duplicates_size, raft::comms::op_t::SUM, handle.get_stream());
        has_duplicates_size = host_scalar_allreduce(
          handle.get_comms(), has_duplicates_size, raft::comms::op_t::SUM, handle.get_stream());
      }

      if (no_duplicates_size > 0) {
        update_temporal_edge_mask(
          handle,
          graph_view,
          edge_start_time_view,
          raft::device_span<vertex_t const>{frontier_vertices_no_duplicates.data(),
                                            frontier_vertices_no_duplicates.size()},
          raft::device_span<edge_time_t const>{frontier_vertex_times_no_duplicates.data(),
                                               frontier_vertex_times_no_duplicates.size()},
          edge_time_mask.mutable_view(),
          sampling_flags.temporal_sampling_comparison);

        temporal_graph_view.attach_edge_mask(edge_time_mask.view());
      }
    } else {
      // This can only happen in the first hop if starting_vertex_times is not provided.
      frontier_vertices_no_duplicates.resize(starting_vertices.size(), handle.get_stream());
      raft::copy(frontier_vertices_no_duplicates.data(),
                 starting_vertices.data(),
                 starting_vertices.size(),
                 handle.get_stream());
      if (starting_vertex_labels) {
        if (frontier_vertex_labels_no_duplicates) {
          frontier_vertex_labels_no_duplicates->resize(starting_vertices.size(),
                                                       handle.get_stream());
        } else {
          frontier_vertex_labels_no_duplicates = std::make_optional(
            rmm::device_uvector<label_t>{starting_vertex_labels->size(), handle.get_stream()});
        }
        raft::copy(frontier_vertex_labels_no_duplicates->data(),
                   starting_vertex_labels->data(),
                   starting_vertex_labels->size(),
                   handle.get_stream());
      }

      no_duplicates_size = frontier_vertices_no_duplicates.size();
      if constexpr (multi_gpu) {
        no_duplicates_size = host_scalar_allreduce(
          handle.get_comms(), no_duplicates_size, raft::comms::op_t::SUM, handle.get_stream());
      }
    }

    if (level_Ks) {
      if (no_duplicates_size > 0) {
        std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};

        if (edge_weight_view) edge_property_views.push_back(*edge_weight_view);
        if (edge_id_view) edge_property_views.push_back(*edge_id_view);
        if (edge_type_view) edge_property_views.push_back(*edge_type_view);
        edge_property_views.push_back(edge_start_time_view);
        if (edge_end_time_view) edge_property_views.push_back(*edge_end_time_view);

        auto [srcs, dsts, sampled_edge_properties, labels] = sample_edges(
          handle,
          rng_state,
          temporal_graph_view,
          raft::host_span<edge_arithmetic_property_view_t<edge_t>>{edge_property_views.data(),
                                                                   edge_property_views.size()},
          edge_type_view
            ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_type_view)
            : std::nullopt,
          edge_bias_view
            ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_bias_view)
            : std::nullopt,
          raft::device_span<vertex_t const>{frontier_vertices_no_duplicates.data(),
                                            frontier_vertices_no_duplicates.size()},
          frontier_vertex_labels_no_duplicates
            ? std::make_optional(
                raft::device_span<label_t const>{frontier_vertex_labels_no_duplicates->data(),
                                                 frontier_vertex_labels_no_duplicates->size()})
            : std::nullopt,
          raft::host_span<size_t const>(level_Ks->data(), level_Ks->size()),
          sampling_flags.with_replacement);

        result_vector_sizes.push_back(srcs.size());
        result_vector_hops.push_back(hop);
        result_src_vectors.push_back(std::move(srcs));
        result_dst_vectors.push_back(std::move(dsts));

        size_t pos{0};
        auto weights =
          (edge_weight_view)
            ? std::make_optional(
                std::move(std::get<rmm::device_uvector<weight_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;
        auto edge_ids =
          (edge_id_view) ? std::make_optional(std::move(
                             std::get<rmm::device_uvector<edge_t>>(sampled_edge_properties[pos++])))
                         : std::nullopt;
        auto edge_types =
          (edge_type_view)
            ? std::make_optional(std::move(
                std::get<rmm::device_uvector<edge_type_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;
        auto edge_start_times =
          std::move(std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++]));
        auto edge_end_times =
          (edge_end_time_view)
            ? std::make_optional(std::move(
                std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;

        if (weights) { (*result_weight_vectors).push_back(std::move(*weights)); }
        if (edge_ids) { (*result_edge_id_vectors).push_back(std::move(*edge_ids)); }
        if (edge_types) { (*result_edge_type_vectors).push_back(std::move(*edge_types)); }
        (*result_edge_start_time_vectors).push_back(std::move(edge_start_times));
        if (edge_end_times) {
          (*result_edge_end_time_vectors).push_back(std::move(*edge_end_times));
        }
        if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }

        next_frontier_vertex_spans.push_back(raft::device_span<vertex_t const>{
          result_dst_vectors.back().data(), result_dst_vectors.back().size()});
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<edge_time_t const>{result_edge_start_time_vectors->back().data(),
                                               result_edge_start_time_vectors->back().size()});
        if (next_frontier_vertex_label_spans) {
          next_frontier_vertex_label_spans->push_back(raft::device_span<label_t const>{
            result_label_vectors->back().data(), result_label_vectors->back().size()});
        }
      }

      if (has_duplicates_size > 0) {
        std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};

        if (edge_weight_view) edge_property_views.push_back(*edge_weight_view);
        if (edge_id_view) edge_property_views.push_back(*edge_id_view);
        if (edge_type_view) edge_property_views.push_back(*edge_type_view);
        edge_property_views.push_back(edge_start_time_view);
        if (edge_end_time_view) edge_property_views.push_back(*edge_end_time_view);

        auto [srcs, dsts, sampled_edge_properties, labels] =
          temporal_sample_edges<vertex_t, edge_t, edge_time_t, multi_gpu>(
            handle,
            rng_state,
            graph_view,
            raft::host_span<edge_arithmetic_property_view_t<edge_t>>{edge_property_views.data(),
                                                                     edge_property_views.size()},
            edge_start_time_view,
            edge_type_view
              ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_type_view)
              : std::nullopt,
            edge_bias_view
              ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_bias_view)
              : std::nullopt,
            raft::device_span<vertex_t const>{frontier_vertices_has_duplicates.data(),
                                              frontier_vertices_has_duplicates.size()},
            raft::device_span<edge_time_t const>{frontier_vertex_times_has_duplicates.data(),
                                                 frontier_vertex_times_has_duplicates.size()},
            frontier_vertex_labels_has_duplicates
              ? std::make_optional(
                  raft::device_span<label_t const>{frontier_vertex_labels_has_duplicates->data(),
                                                   frontier_vertex_labels_has_duplicates->size()})
              : std::nullopt,
            raft::host_span<size_t const>(level_Ks->data(), level_Ks->size()),
            sampling_flags.with_replacement,
            sampling_flags.temporal_sampling_comparison);

        size_t pos{0};
        auto weights =
          (edge_weight_view)
            ? std::make_optional(
                std::move(std::get<rmm::device_uvector<weight_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;
        auto edge_ids =
          (edge_id_view) ? std::make_optional(std::move(
                             std::get<rmm::device_uvector<edge_t>>(sampled_edge_properties[pos++])))
                         : std::nullopt;
        auto edge_types =
          (edge_type_view)
            ? std::make_optional(std::move(
                std::get<rmm::device_uvector<edge_type_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;
        auto edge_start_times =
          std::move(std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++]));
        auto edge_end_times =
          (edge_end_time_view)
            ? std::make_optional(std::move(
                std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])))
            : std::nullopt;

        if (srcs.size() > 0) {
          result_vector_sizes.push_back(srcs.size());
          result_vector_hops.push_back(hop);
          result_src_vectors.push_back(std::move(srcs));
          result_dst_vectors.push_back(std::move(dsts));

          if (weights) { (*result_weight_vectors).push_back(std::move(*weights)); }
          if (edge_ids) { (*result_edge_id_vectors).push_back(std::move(*edge_ids)); }
          if (edge_types) { (*result_edge_type_vectors).push_back(std::move(*edge_types)); }
          (*result_edge_start_time_vectors).push_back(std::move(edge_start_times));
          if (edge_end_times) {
            (*result_edge_end_time_vectors).push_back(std::move(*edge_end_times));
          }
          if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }

          next_frontier_vertex_spans.push_back(raft::device_span<vertex_t const>{
            result_dst_vectors.back().data(), result_dst_vectors.back().size()});
          next_frontier_vertex_time_spans->push_back(
            raft::device_span<edge_time_t const>{result_edge_start_time_vectors->back().data(),
                                                 result_edge_start_time_vectors->back().size()});
          if (next_frontier_vertex_label_spans) {
            next_frontier_vertex_label_spans->push_back(raft::device_span<label_t const>{
              result_label_vectors->back().data(), result_label_vectors->back().size()});
          }
        }
      }
    }

    if (gather_flags) {
      rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
      raft::update_device(
        d_gather_flags.data(), gather_flags->data(), gather_flags->size(), handle.get_stream());
      auto gather_flags_span = gather_flags->size() > 1
                                 ? std::make_optional(raft::device_span<uint8_t const>{
                                     d_gather_flags.data(), d_gather_flags.size()})
                                 : std::nullopt;

      if (no_duplicates_size > 0) {
        std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};

        if (edge_weight_view) edge_property_views.push_back(*edge_weight_view);
        if (edge_id_view) edge_property_views.push_back(*edge_id_view);
        if (edge_type_view) edge_property_views.push_back(*edge_type_view);
        edge_property_views.push_back(edge_start_time_view);
        if (edge_end_time_view) edge_property_views.push_back(*edge_end_time_view);

        auto [srcs, dsts, sampled_edge_properties, labels] = gather_one_hop_edgelist(
          handle,
          temporal_graph_view,
          raft::host_span<edge_arithmetic_property_view_t<edge_t>>{edge_property_views.data(),
                                                                   edge_property_views.size()},
          edge_type_view,
          raft::device_span<vertex_t const>{frontier_vertices_no_duplicates.data(),
                                            frontier_vertices_no_duplicates.size()},
          frontier_vertex_labels_no_duplicates
            ? std::make_optional(
                raft::device_span<label_t const>{frontier_vertex_labels_no_duplicates->data(),
                                                 frontier_vertex_labels_no_duplicates->size()})
            : std::nullopt,
          gather_flags_span,
          do_expensive_check);

        if (srcs.size() > 0) {
          result_vector_sizes.push_back(srcs.size());
          result_vector_hops.push_back(hop);
          result_src_vectors.push_back(std::move(srcs));
          result_dst_vectors.push_back(std::move(dsts));

          size_t pos{0};
          if (edge_weight_view) {
            (*result_weight_vectors)
              .push_back(
                std::move(std::get<rmm::device_uvector<weight_t>>(sampled_edge_properties[pos++])));
          }
          if (edge_id_view) {
            (*result_edge_id_vectors)
              .push_back(
                std::move(std::get<rmm::device_uvector<edge_t>>(sampled_edge_properties[pos++])));
          }
          if (edge_type_view) {
            (*result_edge_type_vectors)
              .push_back(std::move(
                std::get<rmm::device_uvector<edge_type_t>>(sampled_edge_properties[pos++])));
          }
          (*result_edge_start_time_vectors)
            .push_back(std::move(
              std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])));
          if (edge_end_time_view) {
            (*result_edge_end_time_vectors)
              .push_back(std::move(
                std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])));
          }
          if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }

          next_frontier_vertex_spans.push_back(raft::device_span<vertex_t const>{
            result_dst_vectors.back().data(), result_dst_vectors.back().size()});
          next_frontier_vertex_time_spans->push_back(
            raft::device_span<edge_time_t const>{result_edge_start_time_vectors->back().data(),
                                                 result_edge_start_time_vectors->back().size()});
          if (next_frontier_vertex_label_spans) {
            next_frontier_vertex_label_spans->push_back(raft::device_span<label_t const>{
              result_label_vectors->back().data(), result_label_vectors->back().size()});
          }
        }
      }

      if (has_duplicates_size > 0) {
        std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};

        if (edge_weight_view) edge_property_views.push_back(*edge_weight_view);
        if (edge_id_view) edge_property_views.push_back(*edge_id_view);
        if (edge_type_view) edge_property_views.push_back(*edge_type_view);
        edge_property_views.push_back(edge_start_time_view);
        if (edge_end_time_view) edge_property_views.push_back(*edge_end_time_view);

        auto [srcs, dsts, sampled_edge_properties, labels] = temporal_gather_one_hop_edgelist(
          handle,
          graph_view,
          raft::host_span<edge_arithmetic_property_view_t<edge_t>>{edge_property_views.data(),
                                                                   edge_property_views.size()},
          edge_start_time_view,
          edge_type_view,
          raft::device_span<vertex_t const>{frontier_vertices_has_duplicates.data(),
                                            frontier_vertices_has_duplicates.size()},

          raft::device_span<edge_time_t const>{frontier_vertex_times_has_duplicates.data(),
                                               frontier_vertex_times_has_duplicates.size()},
          frontier_vertex_labels_has_duplicates
            ? std::make_optional(
                raft::device_span<label_t const>{frontier_vertex_labels_has_duplicates->data(),
                                                 frontier_vertex_labels_has_duplicates->size()})
            : std::nullopt,
          gather_flags_span,
          sampling_flags.temporal_sampling_comparison,
          do_expensive_check);

        if (srcs.size() > 0) {
          result_vector_sizes.push_back(srcs.size());
          result_vector_hops.push_back(hop);
          result_src_vectors.push_back(std::move(srcs));
          result_dst_vectors.push_back(std::move(dsts));

          size_t pos{0};
          if (edge_weight_view) {
            (*result_weight_vectors)
              .push_back(
                std::move(std::get<rmm::device_uvector<weight_t>>(sampled_edge_properties[pos++])));
          }
          if (edge_id_view) {
            (*result_edge_id_vectors)
              .push_back(
                std::move(std::get<rmm::device_uvector<edge_t>>(sampled_edge_properties[pos++])));
          }
          if (edge_type_view) {
            (*result_edge_type_vectors)
              .push_back(std::move(
                std::get<rmm::device_uvector<edge_type_t>>(sampled_edge_properties[pos++])));
          }
          (*result_edge_start_time_vectors)
            .push_back(std::move(
              std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])));
          if (edge_end_time_view) {
            (*result_edge_end_time_vectors)
              .push_back(std::move(
                std::get<rmm::device_uvector<edge_time_t>>(sampled_edge_properties[pos++])));
          }
          if (labels) { (*result_label_vectors).push_back(std::move(*labels)); }
        }
      }
    }

    std::tie(
      frontier_vertices, frontier_vertex_labels, frontier_vertex_times, vertex_used_as_source) =
      prepare_next_frontier(
        handle,
        raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
        frontier_vertex_labels ? std::make_optional(raft::device_span<label_t const>(
                                   frontier_vertex_labels->data(), frontier_vertex_labels->size()))
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
        graph_view.vertex_partition_range_lasts(),
        sampling_flags.prior_sources_behavior,
        sampling_flags.dedupe_sources,
        multi_gpu,
        do_expensive_check);
  }

  auto result_size = std::reduce(result_vector_sizes.begin(), result_vector_sizes.end());
  size_t output_offset{};

  rmm::device_uvector<vertex_t> result_srcs(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < result_src_vectors.size(); ++i) {
    raft::copy(result_srcs.begin() + output_offset,
               result_src_vectors[i].begin(),
               result_vector_sizes[i],
               handle.get_stream());
    output_offset += result_vector_sizes[i];
  }
  result_src_vectors.clear();
  result_src_vectors.shrink_to_fit();

  rmm::device_uvector<vertex_t> result_dsts(result_size, handle.get_stream());
  output_offset = 0;
  for (size_t i = 0; i < result_dst_vectors.size(); ++i) {
    raft::copy(result_dsts.begin() + output_offset,
               result_dst_vectors[i].begin(),
               result_vector_sizes[i],
               handle.get_stream());
    output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
    }
    result_edge_end_time_vectors = std::nullopt;
  }

  std::optional<rmm::device_uvector<int32_t>> result_hops{std::nullopt};

  if (sampling_flags.return_hops) {
    // FIX THIS!!!  It's possible for some labels to end before num_hops values... so we need to
    // Populate the result_hops vector some other way...
    result_hops   = rmm::device_uvector<int32_t>(result_size, handle.get_stream());
    output_offset = 0;
    for (size_t i = 0; i < result_vector_hops.size(); ++i) {
      scalar_fill(
        handle, result_hops->data() + output_offset, result_vector_sizes[i], result_vector_hops[i]);
      output_offset += result_vector_sizes[i];
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
                 result_vector_sizes[i],
                 handle.get_stream());
      output_offset += result_vector_sizes[i];
    }
    result_label_vectors = std::nullopt;
  }

  std::vector<cugraph::arithmetic_device_uvector_t> property_edges{};

  property_edges.push_back(std::move(result_srcs));
  property_edges.push_back(std::move(result_dsts));
  if (result_weights) property_edges.push_back(std::move(*result_weights));
  if (result_edge_ids) property_edges.push_back(std::move(*result_edge_ids));
  if (result_edge_types) property_edges.push_back(std::move(*result_edge_types));
  if (result_edge_start_times) property_edges.push_back(std::move(*result_edge_start_times));
  if (result_edge_end_times) property_edges.push_back(std::move(*result_edge_end_times));

  std::optional<rmm::device_uvector<size_t>> result_offsets{std::nullopt};

  std::tie(property_edges, result_labels, result_hops, result_offsets) =
    shuffle_and_organize_output(
      handle,
      std::move(property_edges),
      std::move(result_labels),
      std::move(result_hops),
      starting_vertex_labels,
      sampling_flags.return_hops ? std::make_optional<int32_t>(num_hops) : std::nullopt,
      label_to_output_comm_rank);

  size_t pos  = 0;
  result_srcs = std::move(std::get<rmm::device_uvector<vertex_t>>(property_edges[pos++]));
  result_dsts = std::move(std::get<rmm::device_uvector<vertex_t>>(property_edges[pos++]));
  if (result_weights)
    result_weights = std::move(std::get<rmm::device_uvector<weight_t>>(property_edges[pos++]));
  if (result_edge_ids)
    result_edge_ids = std::move(std::get<rmm::device_uvector<edge_t>>(property_edges[pos++]));
  if (result_edge_types)
    result_edge_types =
      std::move(std::get<rmm::device_uvector<edge_type_t>>(property_edges[pos++]));
  if (result_edge_start_times)
    result_edge_start_times =
      std::move(std::get<rmm::device_uvector<edge_time_t>>(property_edges[pos++]));
  if (result_edge_end_times)
    result_edge_end_times =
      std::move(std::get<rmm::device_uvector<edge_time_t>>(property_edges[pos++]));

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
  std::optional<raft::device_span<edge_time_t const>> starting_vertex_times,
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
      starting_vertex_times,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags,
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
  std::optional<raft::device_span<edge_time_t const>> starting_vertex_times,
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
      starting_vertex_times,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags,
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
  std::optional<raft::device_span<edge_time_t const>> starting_vertex_times,
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
      starting_vertex_times,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{std::nullopt},
      sampling_flags,
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
  std::optional<raft::device_span<edge_time_t const>> starting_vertex_times,
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
      starting_vertex_times,
      starting_vertex_labels,
      label_to_output_comm_rank,
      fan_out,
      std::optional<edge_type_t>{num_edge_types},
      sampling_flags,
      do_expensive_check);
}

}  // namespace cugraph
