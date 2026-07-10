/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sample_edges_one_property.cuh"
#include "sampling_utils.hpp"

#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
sample_edges(raft::handle_t const& handle,
             raft::random::RngState& rng_state,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             size_t number_of_edge_properties,
             std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
             std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
             raft::device_span<vertex_t const> active_majors,
             std::optional<raft::device_span<int32_t const>> active_major_labels,
             raft::host_span<size_t const> Ks,
             bool with_replacement)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");

  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  auto& bucket0 = vertex_frontier.bucket(0);
  bucket0.insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> sample_labels{std::nullopt};

  auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false>(
    handle, raft::device_span<vertex_t const>(bucket0.begin(), bucket0.size()));

  if (number_of_edge_properties == 0) {
    std::tie(majors, minors, std::ignore, sample_labels) =
      sample_with_one_property(handle,
                               rng_state,
                               graph_view,
                               cugraph::edge_dummy_property_view_t{},
                               edge_type_view,
                               edge_bias_view,
                               active_bucket_view,
                               Ks,
                               active_major_labels,
                               with_replacement);
  } else {
    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(majors, minors, tmp_edge_indices, sample_labels) =
        sample_with_one_property(handle,
                                 rng_state,
                                 graph_view,
                                 multi_index_property.view(),
                                 edge_type_view,
                                 edge_bias_view,
                                 active_bucket_view,
                                 Ks,
                                 active_major_labels,
                                 with_replacement);

    } else {
      std::tie(majors, minors, std::ignore, sample_labels) =
        sample_with_one_property(handle,
                                 rng_state,
                                 graph_view,
                                 cugraph::edge_dummy_property_view_t{},
                                 edge_type_view,
                                 edge_bias_view,
                                 active_bucket_view,
                                 Ks,
                                 active_major_labels,
                                 with_replacement);
    }
  }

  labels = std::move(sample_labels);

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(tmp_edge_indices), std::move(labels));
}

template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      size_t number_of_edge_properties,
                      edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
                      std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
                      std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
                      raft::device_span<vertex_t const> active_majors,
                      raft::device_span<time_stamp_t const> active_major_times,
                      std::optional<raft::device_span<int32_t const>> active_major_labels,
                      raft::host_span<size_t const> Ks,
                      bool with_replacement,
                      temporal_sampling_comparison_t temporal_sampling_comparison)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");
  CUGRAPH_EXPECTS(number_of_edge_properties > 0,
                  "Temporal sampling requires at least a time as a property");

  using tag_t = time_stamp_t;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  auto& bucket0 = vertex_frontier.bucket(0);
  bucket0.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_times.begin()),
                 thrust::make_zip_iterator(active_majors.end(), active_major_times.end()));

  auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false>(
    handle,
    raft::device_span<vertex_t const>(bucket0.vertex_begin(), bucket0.size()),
    raft::device_span<tag_t const>(bucket0.tag_begin(), bucket0.size()));

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

  if (graph_view.is_multigraph()) {
    cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle, graph_view);

    std::tie(majors, minors, tmp_edge_indices, labels) =
      temporal_sample_with_one_property(handle,
                                        rng_state,
                                        graph_view,
                                        multi_index_property.view(),
                                        edge_time_view,
                                        edge_type_view,
                                        edge_bias_view,
                                        active_bucket_view,
                                        Ks,
                                        with_replacement,
                                        active_major_labels,
                                        temporal_sampling_comparison);

  } else {
    std::tie(majors, minors, std::ignore, labels) =
      temporal_sample_with_one_property(handle,
                                        rng_state,
                                        graph_view,
                                        cugraph::edge_dummy_property_view_t{},
                                        edge_time_view,
                                        edge_type_view,
                                        edge_bias_view,
                                        active_bucket_view,
                                        Ks,
                                        with_replacement,
                                        active_major_labels,
                                        temporal_sampling_comparison);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(tmp_edge_indices), std::move(labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_edges_to_unvisited_neighbors(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  size_t number_of_edge_properties,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  raft::host_span<size_t const> Ks,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  bool with_replacement)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");
  CUGRAPH_EXPECTS(active_major_labels.has_value() == visited_minor_labels.has_value(),
                  "Active major labels and visited vertex labels must both be specified or both "
                  "be unspecified");

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (active_major_labels) {
    using tag_t = int32_t;  // label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    auto& bucket0 = vertex_frontier.bucket(0);
    bucket0.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
                   thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, tag_t, multi_gpu, false>(
      handle,
      raft::device_span<vertex_t const>(bucket0.vertex_begin(), bucket0.size()),
      raft::device_span<tag_t const>(bucket0.tag_begin(), bucket0.size()));

    if (number_of_edge_properties == 0) {
      std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           active_bucket_view,
                                           std::move(visited_minors),
                                           std::move(visited_minor_labels),
                                           Ks,
                                           active_major_labels,
                                           with_replacement);
    } else {
      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);

        std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             multi_index_property.view(),
                                             edge_type_view,
                                             edge_bias_view,
                                             active_bucket_view,
                                             std::move(visited_minors),
                                             std::move(visited_minor_labels),
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      } else {
        std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             cugraph::edge_dummy_property_view_t{},
                                             edge_type_view,
                                             edge_bias_view,
                                             active_bucket_view,
                                             std::move(visited_minors),
                                             std::move(visited_minor_labels),
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      }
    }

  } else {
    using tag_t = void;  // no label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    auto& bucket0 = vertex_frontier.bucket(0);
    bucket0.insert(active_majors.begin(), active_majors.end());

    auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false>(
      handle, raft::device_span<vertex_t const>(bucket0.begin(), bucket0.size()));

    if (number_of_edge_properties == 0) {
      std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           active_bucket_view,
                                           std::move(visited_minors),
                                           std::move(visited_minor_labels),
                                           Ks,
                                           active_major_labels,
                                           with_replacement);
    } else {
      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);
        std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             multi_index_property.view(),
                                             edge_type_view,
                                             edge_bias_view,
                                             active_bucket_view,
                                             std::move(visited_minors),
                                             std::move(visited_minor_labels),
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      } else {
        std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             cugraph::edge_dummy_property_view_t{},
                                             edge_type_view,
                                             edge_bias_view,
                                             active_bucket_view,
                                             std::move(visited_minors),
                                             std::move(visited_minor_labels),
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      }
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(tmp_edge_indices),
                         std::move(labels),
                         std::move(visited_minors),
                         std::move(visited_minor_labels));
}

}  // namespace detail
}  // namespace cugraph
