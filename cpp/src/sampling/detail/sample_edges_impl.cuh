/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sample_outgoing_edges.hpp"
#include "sampling_utils.hpp"

#include <cugraph/detail/device_comm_wrapper.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include <limits>
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

  std::tie(majors, minors, tmp_edge_indices, sample_labels) =
    sample_outgoing_edges(handle,
                          rng_state,
                          graph_view,
                          number_of_edge_properties > 0,
                          edge_type_view,
                          edge_bias_view,
                          active_bucket_view,
                          Ks,
                          active_major_labels,
                          with_replacement);

  labels = std::move(sample_labels);

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

    std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
      sample_unvisited_outgoing_edges(handle,
                                      rng_state,
                                      graph_view,
                                      number_of_edge_properties > 0,
                                      edge_type_view,
                                      edge_bias_view,
                                      active_bucket_view,
                                      std::move(visited_minors),
                                      std::move(visited_minor_labels),
                                      Ks,
                                      active_major_labels,
                                      with_replacement);

  } else {
    using tag_t = void;  // no label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    auto& bucket0 = vertex_frontier.bucket(0);
    bucket0.insert(active_majors.begin(), active_majors.end());

    auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false>(
      handle, raft::device_span<vertex_t const>(bucket0.begin(), bucket0.size()));

    std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
      sample_unvisited_outgoing_edges(handle,
                                      rng_state,
                                      graph_view,
                                      number_of_edge_properties > 0,
                                      edge_type_view,
                                      edge_bias_view,
                                      active_bucket_view,
                                      std::move(visited_minors),
                                      std::move(visited_minor_labels),
                                      Ks,
                                      active_major_labels,
                                      with_replacement);
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(tmp_edge_indices),
                         std::move(labels),
                         std::move(visited_minors),
                         std::move(visited_minor_labels));
}

namespace {

struct temporal_side_table_window_reduce_t {
  bool increasing{};
  template <typename time_stamp_t>
  __device__ cuda::std::tuple<time_stamp_t, time_stamp_t> operator()(
    cuda::std::tuple<time_stamp_t, time_stamp_t> a,
    cuda::std::tuple<time_stamp_t, time_stamp_t> b) const
  {
    auto const time_a = cuda::std::get<0>(a);
    auto const time_b = cuda::std::get<0>(b);
    if (increasing) { return time_a >= time_b ? a : b; }
    return time_a <= time_b ? a : b;
  }
};

// minor_comm allgather can concatenate duplicate (major[, label]) keys with different times.  Keep
// the extremum required by the temporal mode so the bias operator's lookup sees one entry per key.
template <typename vertex_t, typename time_stamp_t>
void dedupe_sorted_temporal_mg_side_table(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& sorted_majors,
  rmm::device_uvector<time_stamp_t>& sorted_times,
  std::optional<rmm::device_uvector<time_stamp_t>>& sorted_window_ends,
  std::optional<rmm::device_uvector<int32_t>>& sorted_labels,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  auto const n = sorted_majors.size();
  if (n < 2) { return; }

  bool const increasing =
    temporal_sampling_comparison == temporal_sampling_comparison_t::MONOTONICALLY_INCREASING ||
    temporal_sampling_comparison == temporal_sampling_comparison_t::STRICTLY_INCREASING;

  if (sorted_labels && sorted_window_ends) {
    rmm::device_uvector<vertex_t> out_majors(n, handle.get_stream());
    rmm::device_uvector<int32_t> out_labels(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_window_ends(n, handle.get_stream());

    temporal_side_table_window_reduce_t reducer{increasing};
    auto ends = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(sorted_majors.begin(), sorted_labels->begin()),
      thrust::make_zip_iterator(sorted_majors.end(), sorted_labels->end()),
      thrust::make_zip_iterator(sorted_times.begin(), sorted_window_ends->begin()),
      thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()),
      thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
      thrust::equal_to<cuda::std::tuple<vertex_t, int32_t>>{},
      reducer);
    auto const new_size = static_cast<size_t>(cuda::std::distance(
      thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()), ends.first));

    sorted_majors       = std::move(out_majors);
    *sorted_labels      = std::move(out_labels);
    sorted_times        = std::move(out_times);
    *sorted_window_ends = std::move(out_window_ends);
    sorted_majors.resize(new_size, handle.get_stream());
    sorted_labels->resize(new_size, handle.get_stream());
    sorted_times.resize(new_size, handle.get_stream());
    sorted_window_ends->resize(new_size, handle.get_stream());
  } else if (sorted_labels) {
    rmm::device_uvector<vertex_t> out_majors(n, handle.get_stream());
    rmm::device_uvector<int32_t> out_labels(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());

    size_t new_size{};
    if (increasing) {
      auto ends = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(sorted_majors.begin(), sorted_labels->begin()),
        thrust::make_zip_iterator(sorted_majors.end(), sorted_labels->end()),
        sorted_times.begin(),
        thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()),
        out_times.begin(),
        thrust::equal_to<cuda::std::tuple<vertex_t, int32_t>>{},
        thrust::maximum<time_stamp_t>{});
      new_size = static_cast<size_t>(cuda::std::distance(
        thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()), ends.first));
    } else {
      auto ends = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(sorted_majors.begin(), sorted_labels->begin()),
        thrust::make_zip_iterator(sorted_majors.end(), sorted_labels->end()),
        sorted_times.begin(),
        thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()),
        out_times.begin(),
        thrust::equal_to<cuda::std::tuple<vertex_t, int32_t>>{},
        thrust::minimum<time_stamp_t>{});
      new_size = static_cast<size_t>(cuda::std::distance(
        thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()), ends.first));
    }

    sorted_majors  = std::move(out_majors);
    *sorted_labels = std::move(out_labels);
    sorted_times   = std::move(out_times);
    sorted_majors.resize(new_size, handle.get_stream());
    sorted_labels->resize(new_size, handle.get_stream());
    sorted_times.resize(new_size, handle.get_stream());
  } else if (sorted_window_ends) {
    rmm::device_uvector<vertex_t> out_majors(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_window_ends(n, handle.get_stream());

    temporal_side_table_window_reduce_t reducer{increasing};
    auto ends = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      sorted_majors.begin(),
      sorted_majors.end(),
      thrust::make_zip_iterator(sorted_times.begin(), sorted_window_ends->begin()),
      out_majors.begin(),
      thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
      thrust::equal_to<vertex_t>{},
      reducer);
    auto const new_size = static_cast<size_t>(cuda::std::distance(out_majors.begin(), ends.first));

    sorted_majors       = std::move(out_majors);
    sorted_times        = std::move(out_times);
    *sorted_window_ends = std::move(out_window_ends);
    sorted_majors.resize(new_size, handle.get_stream());
    sorted_times.resize(new_size, handle.get_stream());
    sorted_window_ends->resize(new_size, handle.get_stream());
  } else {
    rmm::device_uvector<vertex_t> out_majors(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());

    size_t new_size{};
    if (increasing) {
      auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        sorted_majors.begin(),
                                        sorted_majors.end(),
                                        sorted_times.begin(),
                                        out_majors.begin(),
                                        out_times.begin(),
                                        thrust::equal_to<vertex_t>{},
                                        thrust::maximum<time_stamp_t>{});
      new_size  = static_cast<size_t>(cuda::std::distance(out_majors.begin(), ends.first));
    } else {
      auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        sorted_majors.begin(),
                                        sorted_majors.end(),
                                        sorted_times.begin(),
                                        out_majors.begin(),
                                        out_times.begin(),
                                        thrust::equal_to<vertex_t>{},
                                        thrust::minimum<time_stamp_t>{});
      new_size  = static_cast<size_t>(cuda::std::distance(out_majors.begin(), ends.first));
    }

    sorted_majors = std::move(out_majors);
    sorted_times  = std::move(out_times);
    sorted_majors.resize(new_size, handle.get_stream());
    sorted_times.resize(new_size, handle.get_stream());
  }
}

}  // namespace

// Temporal sampling to unvisited neighbors.  Combines the per-source temporal window filter with
// the disjoint (unvisited) constraint by reusing sample_unvisited_outgoing_edges's resample loop
// with a temporal-aware bias operator.  The per-source (time, window_end) values are looked up
// inside the bias operator from side spans sorted by (major) when unlabeled and by (major, label)
// when labeled; under always-disjoint sampling that key is unique.
template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges_to_unvisited_neighbors(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  size_t number_of_edge_properties,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<time_stamp_t const> active_major_times,
  std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  raft::host_span<size_t const> Ks,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  bool with_replacement,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");
  CUGRAPH_EXPECTS(number_of_edge_properties > 0,
                  "Temporal sampling requires at least a time as a property");
  CUGRAPH_EXPECTS(active_major_labels.has_value() == visited_minor_labels.has_value(),
                  "Active major labels and visited vertex labels must both be specified or both "
                  "be unspecified");

  // Build side spans sorted so the bias operator can recover (time, window_end) by (major[,
  // label]).  On MG, per_v_random_select_transform_outgoing_e allgathers frontier keys across
  // minor_comm before invoking the bias op, so these side tables must contain every key that
  // minor_comm may present — not only this rank's local frontier.
  rmm::device_uvector<vertex_t> sorted_majors(active_majors.size(), handle.get_stream());
  rmm::device_uvector<time_stamp_t> sorted_times(active_major_times.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), active_majors.begin(), active_majors.end(), sorted_majors.begin());
  thrust::copy(handle.get_thrust_policy(),
               active_major_times.begin(),
               active_major_times.end(),
               sorted_times.begin());

  std::optional<rmm::device_uvector<time_stamp_t>> sorted_window_ends{std::nullopt};
  if (active_major_window_ends) {
    sorted_window_ends =
      rmm::device_uvector<time_stamp_t>(active_major_window_ends->size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 active_major_window_ends->begin(),
                 active_major_window_ends->end(),
                 sorted_window_ends->begin());
  }

  std::optional<rmm::device_uvector<int32_t>> sorted_labels{std::nullopt};
  if (active_major_labels) {
    sorted_labels = rmm::device_uvector<int32_t>(active_major_labels->size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 active_major_labels->begin(),
                 active_major_labels->end(),
                 sorted_labels->begin());
  }

  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    if (minor_comm.get_size() > 1) {
      sorted_majors = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<vertex_t const>{sorted_majors.data(), sorted_majors.size()});
      sorted_times = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<time_stamp_t const>{sorted_times.data(), sorted_times.size()});
      if (sorted_window_ends) {
        sorted_window_ends =
          device_allgatherv(handle,
                            minor_comm,
                            raft::device_span<time_stamp_t const>{sorted_window_ends->data(),
                                                                  sorted_window_ends->size()});
      }
      if (sorted_labels) {
        sorted_labels = device_allgatherv(
          handle,
          minor_comm,
          raft::device_span<int32_t const>{sorted_labels->data(), sorted_labels->size()});
      }
    }
  }

  if (sorted_labels) {
    if (sorted_window_ends) {
      cugraph::sort(handle.get_thrust_policy(),
                    thrust::make_zip_iterator(sorted_majors.begin(),
                                              sorted_labels->begin(),
                                              sorted_times.begin(),
                                              sorted_window_ends->begin()),
                    thrust::make_zip_iterator(sorted_majors.end(),
                                              sorted_labels->end(),
                                              sorted_times.end(),
                                              sorted_window_ends->end()));
    } else {
      cugraph::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(
          sorted_majors.begin(), sorted_labels->begin(), sorted_times.begin()),
        thrust::make_zip_iterator(sorted_majors.end(), sorted_labels->end(), sorted_times.end()));
    }
  } else if (sorted_window_ends) {
    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(
                    sorted_majors.begin(), sorted_times.begin(), sorted_window_ends->begin()),
                  thrust::make_zip_iterator(
                    sorted_majors.end(), sorted_times.end(), sorted_window_ends->end()));
  } else {
    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(sorted_majors.begin(), sorted_times.begin()),
                  thrust::make_zip_iterator(sorted_majors.end(), sorted_times.end()));
  }

  if constexpr (multi_gpu) {
    dedupe_sorted_temporal_mg_side_table<vertex_t, time_stamp_t>(handle,
                                                                 sorted_majors,
                                                                 sorted_times,
                                                                 sorted_window_ends,
                                                                 sorted_labels,
                                                                 temporal_sampling_comparison);
  }

  temporal_unvisited_params_t<vertex_t, edge_t, time_stamp_t> temporal_params{
    edge_time_view,
    raft::device_span<vertex_t const>{sorted_majors.data(), sorted_majors.size()},
    raft::device_span<time_stamp_t const>{sorted_times.data(), sorted_times.size()},
    sorted_window_ends ? cuda::std::make_optional(raft::device_span<time_stamp_t const>{
                           sorted_window_ends->data(), sorted_window_ends->size()})
                       : cuda::std::nullopt,
    sorted_labels ? cuda::std::make_optional(raft::device_span<int32_t const>{
                      sorted_labels->data(), sorted_labels->size()})
                  : cuda::std::nullopt,
    temporal_sampling_comparison};

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

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

    std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
      sample_unvisited_outgoing_edges(handle,
                                      rng_state,
                                      graph_view,
                                      number_of_edge_properties > 0,
                                      edge_type_view,
                                      edge_bias_view,
                                      active_bucket_view,
                                      std::move(visited_minors),
                                      std::move(visited_minor_labels),
                                      Ks,
                                      active_major_labels,
                                      with_replacement,
                                      temporal_params);
  } else {
    using tag_t = void;  // no label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    auto& bucket0 = vertex_frontier.bucket(0);
    bucket0.insert(active_majors.begin(), active_majors.end());

    auto active_bucket_view = cugraph::key_bucket_view_t<vertex_t, void, multi_gpu, false>(
      handle, raft::device_span<vertex_t const>(bucket0.begin(), bucket0.size()));

    std::tie(majors, minors, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
      sample_unvisited_outgoing_edges(handle,
                                      rng_state,
                                      graph_view,
                                      number_of_edge_properties > 0,
                                      edge_type_view,
                                      edge_bias_view,
                                      active_bucket_view,
                                      std::move(visited_minors),
                                      std::move(visited_minor_labels),
                                      Ks,
                                      active_major_labels,
                                      with_replacement,
                                      temporal_params);
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
