/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/gather_sampled_properties.cuh"
#include "detail/output_labels.cuh"
#include "detail/sampling_result_utils.hpp"
#include "detail/sampling_utils.hpp"
#include "detail/shuffle_wrappers.hpp"
#include "detail/temporal_sampling_utils.cuh"
#include "utilities/validation_checks.hpp"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace detail {

template <typename time_stamp_t>
void validate_starting_vertex_time_windows(
  raft::handle_t const& handle,
  raft::device_span<time_stamp_t const> starting_vertex_start_times,
  raft::device_span<time_stamp_t const> starting_vertex_end_times)
{
  auto n = starting_vertex_start_times.size();
  std::vector<time_stamp_t> h_start_times(n);
  std::vector<time_stamp_t> h_end_times(n);
  raft::copy(h_start_times.data(), starting_vertex_start_times.data(), n, handle.get_stream());
  raft::copy(h_end_times.data(), starting_vertex_end_times.data(), n, handle.get_stream());
  handle.sync_stream();
  for (size_t i = 0; i < n; ++i) {
    CUGRAPH_EXPECTS(
      h_start_times[i] <= h_end_times[i],
      "Invalid input argument: starting_vertex_start_times must be less than or equal to "
      "starting_vertex_end_times for each starting vertex.");
  }
}

// Always-disjoint temporal sampling requires that each starting vertex appear at most once per
// label (or once overall when unlabeled).  Duplicate seeds would make the per-source time /
// window-end ambiguous, so reject them up front with a clear error.
template <typename vertex_t, typename label_t>
void validate_no_duplicate_seeds(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels)
{
  auto const n = starting_vertices.size();
  if (n < 2) { return; }

  rmm::device_uvector<vertex_t> sorted_vertices(n, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               starting_vertices.begin(),
               starting_vertices.end(),
               sorted_vertices.begin());

  size_t num_duplicates{0};
  if (starting_vertex_labels) {
    rmm::device_uvector<label_t> sorted_labels(n, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 starting_vertex_labels->begin(),
                 starting_vertex_labels->end(),
                 sorted_labels.begin());
    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(sorted_vertices.begin(), sorted_labels.begin()),
                  thrust::make_zip_iterator(sorted_vertices.end(), sorted_labels.end()));
    num_duplicates = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(1),
      thrust::make_counting_iterator<size_t>(n),
      [vertices = sorted_vertices.data(), labels = sorted_labels.data()] __device__(size_t i) {
        return (vertices[i] == vertices[i - 1]) && (labels[i] == labels[i - 1]);
      });
  } else {
    cugraph::sort(handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end());
    num_duplicates = thrust::count_if(handle.get_thrust_policy(),
                                      thrust::make_counting_iterator<size_t>(1),
                                      thrust::make_counting_iterator<size_t>(n),
                                      [vertices = sorted_vertices.data()] __device__(size_t i) {
                                        return vertices[i] == vertices[i - 1];
                                      });
  }

  CUGRAPH_EXPECTS(num_duplicates == 0,
                  "Invalid input argument: temporal neighbor sampling requires disjoint sampling, "
                  "which does not allow duplicate starting vertices per label.");
}

struct temporal_time_window_reduce_t {
  bool increasing{};
  // FIXED_WINDOW keeps every seed's original (window_start, window_end) fixed, so duplicate
  // entries for the same key are expected to carry identical values; simply keep the first.
  bool fixed_window{};
  template <typename time_stamp_t>
  __device__ cuda::std::tuple<time_stamp_t, time_stamp_t> operator()(
    cuda::std::tuple<time_stamp_t, time_stamp_t> a,
    cuda::std::tuple<time_stamp_t, time_stamp_t> b) const
  {
    if (fixed_window) { return a; }
    auto const time_a = cuda::std::get<0>(a);
    auto const time_b = cuda::std::get<0>(b);
    if (increasing) { return time_a >= time_b ? a : b; }
    return time_a <= time_b ? a : b;
  }
};

// FIXED_WINDOW analogue of thrust::maximum / thrust::minimum used to dedupe scalar frontier times:
// duplicate entries for the same key are expected to be identical, so keep the first.
struct keep_first_temporal_value_t {
  template <typename time_stamp_t>
  __device__ time_stamp_t operator()(time_stamp_t a, time_stamp_t const&) const
  {
    return a;
  }
};

// prepare_next_frontier can emit the same (vertex[, label]) multiple times when several edge lists
// are concatenated for a hop.  Keep the frontier time extremum required by the temporal mode so
// downstream sampling/gather filters use the tightest valid bound.
template <typename vertex_t, typename time_stamp_t, typename label_t>
void dedupe_temporal_frontier(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& frontier_vertices,
  std::optional<rmm::device_uvector<label_t>>& frontier_vertex_labels,
  std::optional<rmm::device_uvector<time_stamp_t>>& frontier_vertex_times,
  std::optional<rmm::device_uvector<time_stamp_t>>& frontier_vertex_window_ends,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  if (!frontier_vertex_times || frontier_vertices.size() < 2) { return; }

  auto const n            = frontier_vertices.size();
  bool const increasing   = !is_temporal_decreasing(temporal_sampling_comparison);
  bool const fixed_window = is_fixed_window(temporal_sampling_comparison);

  if (frontier_vertex_labels && frontier_vertex_window_ends) {
    rmm::device_uvector<vertex_t> out_vertices(n, handle.get_stream());
    rmm::device_uvector<label_t> out_labels(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_window_ends(n, handle.get_stream());

    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(frontier_vertices.begin(),
                                            frontier_vertex_labels->begin(),
                                            frontier_vertex_times->begin(),
                                            frontier_vertex_window_ends->begin()),
                  thrust::make_zip_iterator(frontier_vertices.end(),
                                            frontier_vertex_labels->end(),
                                            frontier_vertex_times->end(),
                                            frontier_vertex_window_ends->end()));

    temporal_time_window_reduce_t reducer{increasing, fixed_window};
    auto ends = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin()),
      thrust::make_zip_iterator(frontier_vertices.end(), frontier_vertex_labels->end()),
      thrust::make_zip_iterator(frontier_vertex_times->begin(),
                                frontier_vertex_window_ends->begin()),
      thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()),
      thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
      thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
      reducer);
    auto const new_size = static_cast<size_t>(cuda::std::distance(
      thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()), ends.first));

    frontier_vertices            = std::move(out_vertices);
    *frontier_vertex_labels      = std::move(out_labels);
    *frontier_vertex_times       = std::move(out_times);
    *frontier_vertex_window_ends = std::move(out_window_ends);
    frontier_vertices.resize(new_size, handle.get_stream());
    frontier_vertex_labels->resize(new_size, handle.get_stream());
    frontier_vertex_times->resize(new_size, handle.get_stream());
    frontier_vertex_window_ends->resize(new_size, handle.get_stream());
  } else if (frontier_vertex_labels) {
    rmm::device_uvector<vertex_t> out_vertices(n, handle.get_stream());
    rmm::device_uvector<label_t> out_labels(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());

    cugraph::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        frontier_vertices.begin(), frontier_vertex_labels->begin(), frontier_vertex_times->begin()),
      thrust::make_zip_iterator(
        frontier_vertices.end(), frontier_vertex_labels->end(), frontier_vertex_times->end()));

    size_t new_size{};
    if (fixed_window) {
      auto ends = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin()),
        thrust::make_zip_iterator(frontier_vertices.end(), frontier_vertex_labels->end()),
        frontier_vertex_times->begin(),
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()),
        out_times.begin(),
        thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
        keep_first_temporal_value_t{});
      new_size = static_cast<size_t>(cuda::std::distance(
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()), ends.first));
    } else if (increasing) {
      auto ends = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin()),
        thrust::make_zip_iterator(frontier_vertices.end(), frontier_vertex_labels->end()),
        frontier_vertex_times->begin(),
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()),
        out_times.begin(),
        thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
        thrust::maximum<time_stamp_t>{});
      new_size = static_cast<size_t>(cuda::std::distance(
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()), ends.first));
    } else {
      auto ends = thrust::reduce_by_key(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_labels->begin()),
        thrust::make_zip_iterator(frontier_vertices.end(), frontier_vertex_labels->end()),
        frontier_vertex_times->begin(),
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()),
        out_times.begin(),
        thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
        thrust::minimum<time_stamp_t>{});
      new_size = static_cast<size_t>(cuda::std::distance(
        thrust::make_zip_iterator(out_vertices.begin(), out_labels.begin()), ends.first));
    }

    frontier_vertices       = std::move(out_vertices);
    *frontier_vertex_labels = std::move(out_labels);
    *frontier_vertex_times  = std::move(out_times);
    frontier_vertices.resize(new_size, handle.get_stream());
    frontier_vertex_labels->resize(new_size, handle.get_stream());
    frontier_vertex_times->resize(new_size, handle.get_stream());
  } else if (frontier_vertex_window_ends) {
    rmm::device_uvector<vertex_t> out_vertices(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_window_ends(n, handle.get_stream());

    cugraph::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(),
                                frontier_vertex_times->begin(),
                                frontier_vertex_window_ends->begin()),
      thrust::make_zip_iterator(
        frontier_vertices.end(), frontier_vertex_times->end(), frontier_vertex_window_ends->end()));

    temporal_time_window_reduce_t reducer{increasing, fixed_window};
    auto ends =
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            frontier_vertices.begin(),
                            frontier_vertices.end(),
                            thrust::make_zip_iterator(frontier_vertex_times->begin(),
                                                      frontier_vertex_window_ends->begin()),
                            out_vertices.begin(),
                            thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
                            thrust::equal_to<vertex_t>{},
                            reducer);
    auto const new_size =
      static_cast<size_t>(cuda::std::distance(out_vertices.begin(), ends.first));

    frontier_vertices            = std::move(out_vertices);
    *frontier_vertex_times       = std::move(out_times);
    *frontier_vertex_window_ends = std::move(out_window_ends);
    frontier_vertices.resize(new_size, handle.get_stream());
    frontier_vertex_times->resize(new_size, handle.get_stream());
    frontier_vertex_window_ends->resize(new_size, handle.get_stream());
  } else {
    rmm::device_uvector<vertex_t> out_vertices(n, handle.get_stream());
    rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());

    cugraph::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(frontier_vertices.begin(), frontier_vertex_times->begin()),
      thrust::make_zip_iterator(frontier_vertices.end(), frontier_vertex_times->end()));

    size_t new_size{};
    if (fixed_window) {
      auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        frontier_vertices.begin(),
                                        frontier_vertices.end(),
                                        frontier_vertex_times->begin(),
                                        out_vertices.begin(),
                                        out_times.begin(),
                                        thrust::equal_to<vertex_t>{},
                                        keep_first_temporal_value_t{});
      new_size  = static_cast<size_t>(cuda::std::distance(out_vertices.begin(), ends.first));
    } else if (increasing) {
      auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        frontier_vertices.begin(),
                                        frontier_vertices.end(),
                                        frontier_vertex_times->begin(),
                                        out_vertices.begin(),
                                        out_times.begin(),
                                        thrust::equal_to<vertex_t>{},
                                        thrust::maximum<time_stamp_t>{});
      new_size  = static_cast<size_t>(cuda::std::distance(out_vertices.begin(), ends.first));
    } else {
      auto ends = thrust::reduce_by_key(handle.get_thrust_policy(),
                                        frontier_vertices.begin(),
                                        frontier_vertices.end(),
                                        frontier_vertex_times->begin(),
                                        out_vertices.begin(),
                                        out_times.begin(),
                                        thrust::equal_to<vertex_t>{},
                                        thrust::minimum<time_stamp_t>{});
      new_size  = static_cast<size_t>(cuda::std::distance(out_vertices.begin(), ends.first));
    }

    frontier_vertices      = std::move(out_vertices);
    *frontier_vertex_times = std::move(out_times);
    frontier_vertices.resize(new_size, handle.get_stream());
    frontier_vertex_times->resize(new_size, handle.get_stream());
  }
}

// Propagate a per-seed frontier time bound (window-end for all modes, or window-start under
// FIXED_WINDOW) from the current frontier to each sampled edge by looking it up from the source
// vertex.  Under always-disjoint temporal sampling each vertex appears at most once per label in
// the frontier, so the key is (src) when unlabeled and (src, label) when labeled; both are unique,
// making the lookup unambiguous.
template <typename vertex_t, typename time_stamp_t, typename label_t>
rmm::device_uvector<time_stamp_t> lookup_src_time_from_frontier(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> frontier_vertices,
  std::optional<raft::device_span<label_t const>> frontier_labels,
  raft::device_span<time_stamp_t const> frontier_times,
  raft::device_span<vertex_t const> srcs,
  std::optional<raft::device_span<label_t const>> src_labels)
{
  auto const n = frontier_vertices.size();

  rmm::device_uvector<vertex_t> sorted_vertices(n, handle.get_stream());
  rmm::device_uvector<time_stamp_t> sorted_times(n, handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               frontier_vertices.begin(),
               frontier_vertices.end(),
               sorted_vertices.begin());
  thrust::copy(
    handle.get_thrust_policy(), frontier_times.begin(), frontier_times.end(), sorted_times.begin());

  rmm::device_uvector<time_stamp_t> edge_times(srcs.size(), handle.get_stream());

  if (frontier_labels) {
    rmm::device_uvector<label_t> sorted_labels(n, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 frontier_labels->begin(),
                 frontier_labels->end(),
                 sorted_labels.begin());

    // (vertex, label) is unique so sorting the full tuple (which also carries the time) still
    // orders by the key.
    cugraph::sort(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(
        sorted_vertices.begin(), sorted_labels.begin(), sorted_times.begin()),
      thrust::make_zip_iterator(sorted_vertices.end(), sorted_labels.end(), sorted_times.end()));

    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(srcs.size()),
      edge_times.begin(),
      [sorted_vertices = sorted_vertices.data(),
       sorted_labels   = sorted_labels.data(),
       sorted_times    = sorted_times.data(),
       srcs            = srcs.data(),
       src_labels      = src_labels->data(),
       num             = n] __device__(size_t idx) {
        auto begin = thrust::make_zip_iterator(sorted_vertices, sorted_labels);
        auto it    = thrust::lower_bound(
          thrust::seq, begin, begin + num, cuda::std::make_tuple(srcs[idx], src_labels[idx]));
        return sorted_times[static_cast<size_t>(cuda::std::distance(begin, it))];
      });
  } else {
    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(sorted_vertices.begin(), sorted_times.begin()),
                  thrust::make_zip_iterator(sorted_vertices.end(), sorted_times.end()));

    thrust::transform(handle.get_thrust_policy(),
                      srcs.begin(),
                      srcs.end(),
                      edge_times.begin(),
                      [sorted_vertices = sorted_vertices.data(),
                       sorted_times    = sorted_times.data(),
                       num             = n] __device__(vertex_t src) {
                        auto it = thrust::lower_bound(
                          thrust::seq, sorted_vertices, sorted_vertices + num, src);
                        return sorted_times[static_cast<size_t>(it - sorted_vertices)];
                      });
  }

  return edge_times;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          typename bias_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
temporal_neighbor_sample_impl(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
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

  CUGRAPH_EXPECTS(
    !starting_vertex_start_times || starting_vertex_start_times->size() == starting_vertices.size(),
    "Invalid input argument: starting_vertex_start_times should have the same size as "
    "starting_vertices.");
  CUGRAPH_EXPECTS(
    !starting_vertex_end_times || starting_vertex_end_times->size() == starting_vertices.size(),
    "Invalid input argument: starting_vertex_end_times should have the same size as "
    "starting_vertices.");

  if (starting_vertex_start_times && starting_vertex_end_times) {
    validate_starting_vertex_time_windows(
      handle, *starting_vertex_start_times, *starting_vertex_end_times);
  }

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

  CUGRAPH_EXPECTS(
    sampling_flags.disjoint_sampling,
    "Invalid input argument: temporal neighbor sampling requires disjoint sampling; set "
    "sampling_flags.disjoint_sampling to true.");
  CUGRAPH_EXPECTS(
    sampling_flags.temporal_sampling_comparison.has_value(),
    "Invalid input argument: temporal neighbor sampling requires temporal_sampling_comparison.");
  auto const temporal_sampling_comparison = *sampling_flags.temporal_sampling_comparison;

  CUGRAPH_EXPECTS((sampling_flags.neighbor_selection == neighbor_selection_t::RANDOM) ||
                    !sampling_flags.with_replacement,
                  "Invalid input argument: LAST neighbor selection does not support sampling with "
                  "replacement.");
  CUGRAPH_EXPECTS(
    (sampling_flags.neighbor_selection == neighbor_selection_t::RANDOM) || !edge_bias_view,
    "Invalid input argument: LAST neighbor selection does not accept edge "
    "biases.");

  validate_no_duplicate_seeds<vertex_t, label_t>(handle, starting_vertices, starting_vertex_labels);

  // Get the number of hop.
  auto num_hops = raft::div_rounding_up_safe(
    fan_out.size(), static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));

  rmm::device_uvector<vertex_t> frontier_vertices(starting_vertices.size(), handle.get_stream());
  raft::copy(frontier_vertices.data(),
             starting_vertices.data(),
             starting_vertices.size(),
             handle.get_stream());

  std::optional<rmm::device_uvector<time_stamp_t>> frontier_vertex_times{std::nullopt};
  std::optional<rmm::device_uvector<time_stamp_t>> frontier_vertex_window_ends{std::nullopt};

  auto frontier_vertex_labels = starting_vertex_labels
                                  ? std::make_optional(rmm::device_uvector<label_t>{
                                      starting_vertex_labels->size(), handle.get_stream()})
                                  : std::nullopt;
  if (frontier_vertex_labels) {
    raft::copy(frontier_vertex_labels->data(),
               starting_vertex_labels->data(),
               starting_vertex_labels->size(),
               handle.get_stream());
  }

  auto const decreasing = is_temporal_decreasing(temporal_sampling_comparison);
  auto const initial_window_starts =
    decreasing ? starting_vertex_end_times : starting_vertex_start_times;
  auto const initial_window_ends =
    decreasing ? starting_vertex_start_times : starting_vertex_end_times;

  if (initial_window_starts) {
    frontier_vertex_times =
      rmm::device_uvector<time_stamp_t>(initial_window_starts->size(), handle.get_stream());
    raft::copy(frontier_vertex_times->data(),
               initial_window_starts->data(),
               initial_window_starts->size(),
               handle.get_stream());
  }

  if (initial_window_ends) {
    frontier_vertex_window_ends =
      rmm::device_uvector<time_stamp_t>(initial_window_ends->size(), handle.get_stream());
    raft::copy(frontier_vertex_window_ends->data(),
               initial_window_ends->data(),
               initial_window_ends->size(),
               handle.get_stream());
  }

  // Temporal neighbor sampling is always disjoint: seed the visited sets with the starting vertices
  // so they are never revisited, and thread the sets through every hop.
  std::optional<rmm::device_uvector<vertex_t>> visited_minors =
    std::make_optional(rmm::device_uvector<vertex_t>(0, handle.get_stream()));
  std::optional<rmm::device_uvector<label_t>> visited_minor_labels =
    starting_vertex_labels
      ? std::make_optional(rmm::device_uvector<label_t>(0, handle.get_stream()))
      : std::nullopt;
  {
    auto [updated_visited_minors, updated_visited_minor_labels] =
      detail::update_dst_visited_vertices_and_labels<vertex_t, edge_t, multi_gpu>(
        handle,
        graph_view,
        std::move(*visited_minors),
        std::move(visited_minor_labels),
        starting_vertices,
        starting_vertex_labels ? std::make_optional(*starting_vertex_labels) : std::nullopt);
    visited_minors       = std::make_optional(std::move(updated_visited_minors));
    visited_minor_labels = std::move(updated_visited_minor_labels);
  }

  std::optional<std::tuple<rmm::device_uvector<vertex_t>,
                           std::optional<rmm::device_uvector<label_t>>,
                           std::optional<rmm::device_uvector<time_stamp_t>>>>
    vertex_used_as_source{std::nullopt};

  if (sampling_flags.prior_sources_behavior == prior_sources_behavior_t::EXCLUDE) {
    vertex_used_as_source = std::make_optional(std::make_tuple(
      rmm::device_uvector<vertex_t>{0, handle.get_stream()},
      starting_vertex_labels
        ? std::make_optional(rmm::device_uvector<label_t>{0, handle.get_stream()})
        : std::nullopt,
      std::make_optional<rmm::device_uvector<time_stamp_t>>(0, handle.get_stream())));
  }

  // Edge property views and span are unchanged across hop iterations; build once and reuse.  The
  // push order below defines the fixed schema of every sampled edge list's property-column vector
  // (see produced_edge_list_t): [weight?, edge_id?, edge_type?, edge_start_time, edge_end_time?].
  std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};
  if (edge_weight_view) edge_property_views.push_back(*edge_weight_view);
  if (edge_id_view) edge_property_views.push_back(*edge_id_view);
  if (edge_type_view) edge_property_views.push_back(*edge_type_view);
  edge_property_views.push_back(edge_start_time_view);
  if (edge_end_time_view) edge_property_views.push_back(*edge_end_time_view);
  auto const edge_prop_span = raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
    edge_property_views.data(), edge_property_views.size()};
  size_t const n_edge_props = edge_property_views.size();

  // Fixed index of each edge-property column within a produced edge list's property vector,
  // matching the edge_property_views push order above.  edge_start_time is always present on the
  // temporal path; the others are present iff their edge view is set.
  size_t prop_pos = 0;
  auto const weight_prop_idx =
    edge_weight_view ? std::make_optional(prop_pos++) : std::optional<size_t>{std::nullopt};
  auto const edge_id_prop_idx =
    edge_id_view ? std::make_optional(prop_pos++) : std::optional<size_t>{std::nullopt};
  auto const edge_type_prop_idx =
    edge_type_view ? std::make_optional(prop_pos++) : std::optional<size_t>{std::nullopt};
  size_t const edge_start_time_prop_idx = prop_pos++;
  auto const edge_end_time_prop_idx =
    edge_end_time_view ? std::make_optional(prop_pos++) : std::optional<size_t>{std::nullopt};

  // Edge types are only a filter key for heterogeneous sampling. Homogeneous temporal sampling may
  // still carry edge types as an output property, so keep edge_type_view in edge_property_views.
  auto const edge_type_filter_view = num_edge_types ? edge_type_view : std::nullopt;

  // Every sampled edge list produced across all hops, in output order.  Each entry owns its source
  // / destination vertices, the property columns (schema above), the optional per-edge label, and
  // the hop it was produced at.  The final result is simply the concatenation of these entries, so
  // this is the single home for sampled-edge data between sampling and output assembly.
  using produced_edge_list_t = std::tuple<rmm::device_uvector<vertex_t>,
                                          rmm::device_uvector<vertex_t>,
                                          std::vector<cugraph::arithmetic_device_uvector_t>,
                                          std::optional<rmm::device_uvector<label_t>>,
                                          int32_t>;
  std::vector<produced_edge_list_t> produced_edge_lists{};
  produced_edge_lists.reserve(num_hops * 2);  // at most a biased/uniform + a gather list per hop

  for (size_t hop = 0; hop < num_hops; ++hop) {
    std::optional<std::vector<size_t>> level_Ks{std::nullopt};
    std::unique_ptr<bool[]> gather_flags{};
    std::vector<raft::device_span<vertex_t const>> next_frontier_vertex_spans{};
    auto next_frontier_vertex_label_spans =
      starting_vertex_labels ? std::make_optional<std::vector<raft::device_span<label_t const>>>()
                             : std::nullopt;
    auto next_frontier_vertex_time_spans =
      std::make_optional<std::vector<raft::device_span<time_stamp_t const>>>();
    auto next_frontier_vertex_window_end_spans =
      frontier_vertex_window_ends
        ? std::make_optional<std::vector<raft::device_span<time_stamp_t const>>>()
        : std::nullopt;
    std::vector<rmm::device_uvector<time_stamp_t>> next_frontier_window_end_vectors{};
    if (next_frontier_vertex_window_end_spans) {
      // At most one push for the biased/uniform (level_Ks) branch and one for the gather branch.
      next_frontier_window_end_vectors.reserve(2);
    }
    // FIXED_WINDOW re-derives each sampled edge's frontier window-start from the seed's original
    // bound (rather than the sampled edge's start time), so those lookups need storage of their
    // own; unlike the other modes' window starts, they are not spans into produced_edge_lists.
    bool const fixed_window_starts =
      is_fixed_window(temporal_sampling_comparison) && frontier_vertex_times.has_value();
    std::vector<rmm::device_uvector<time_stamp_t>> next_frontier_window_start_vectors{};
    if (fixed_window_starts) { next_frontier_window_start_vectors.reserve(2); }

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
          gather_flags = std::make_unique<bool[]>(
            static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1}));
        }
        gather_flags[i - start_offset] = true;
      }
    }

    auto const active_majors =
      raft::device_span<vertex_t const>{frontier_vertices.data(), frontier_vertices.size()};
    auto const active_labels =
      frontier_vertex_labels ? std::make_optional(raft::device_span<label_t const>{
                                 frontier_vertex_labels->data(), frontier_vertex_labels->size()})
                             : std::nullopt;
    auto const active_window_starts =
      frontier_vertex_times ? std::make_optional(raft::device_span<time_stamp_t const>{
                                frontier_vertex_times->data(), frontier_vertex_times->size()})
                            : std::nullopt;
    auto const active_window_ends =
      frontier_vertex_window_ends
        ? std::make_optional(raft::device_span<time_stamp_t const>{
            frontier_vertex_window_ends->data(), frontier_vertex_window_ends->size()})
        : std::nullopt;

    if (level_Ks) {
      rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
      std::vector<cugraph::arithmetic_device_uvector_t> props{};
      std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

      cugraph::arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
      std::tie(srcs, dsts, tmp_edge_indices, labels, visited_minors, visited_minor_labels) =
        temporal_sample_edges_to_unvisited_neighbors<vertex_t, edge_t, time_stamp_t, multi_gpu>(
          handle,
          rng_state,
          graph_view,
          n_edge_props,
          edge_start_time_view,
          edge_type_filter_view
            ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_type_filter_view)
            : std::nullopt,
          edge_bias_view
            ? std::make_optional<edge_arithmetic_property_view_t<edge_t>>(*edge_bias_view)
            : std::nullopt,
          active_majors,
          active_window_starts,
          active_window_ends,
          active_labels,
          raft::host_span<size_t const>(level_Ks->data(), level_Ks->size()),
          std::move(*visited_minors),
          std::move(visited_minor_labels),
          sampling_flags.with_replacement,
          temporal_sampling_comparison,
          sampling_flags.neighbor_selection);
      if (n_edge_props > 0) {
        std::tie(srcs, dsts, props) = gather_sampled_properties(handle,
                                                                graph_view,
                                                                std::move(srcs),
                                                                std::move(dsts),
                                                                std::move(tmp_edge_indices),
                                                                edge_prop_span);
      }

      // Publish this edge list's device spans as next-hop frontier inputs, then move it into
      // produced_edge_lists.  Moving an rmm::device_uvector preserves its device pointer, so the
      // spans stay valid after the move; produced_edge_lists outlives prepare_next_frontier (the
      // span consumer) and the final output assembly.
      next_frontier_vertex_spans.push_back(
        raft::device_span<vertex_t const>{dsts.data(), dsts.size()});
      if (fixed_window_starts) {
        // FIXED_WINDOW never replaces a seed's window-start with a sampled edge time: look it up
        // from the current frontier by (source[, label]) instead, same as the window-end lookup.
        next_frontier_window_start_vectors.push_back(
          lookup_src_time_from_frontier<vertex_t, time_stamp_t, label_t>(
            handle,
            active_majors,
            active_labels,
            *active_window_starts,
            raft::device_span<vertex_t const>{srcs.data(), srcs.size()},
            labels
              ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
              : std::nullopt));
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<time_stamp_t const>{next_frontier_window_start_vectors.back().data(),
                                                next_frontier_window_start_vectors.back().size()});
      } else {
        auto const& edge_start_times =
          std::get<rmm::device_uvector<time_stamp_t>>(props[edge_start_time_prop_idx]);
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<time_stamp_t const>{edge_start_times.data(), edge_start_times.size()});
      }
      if (next_frontier_vertex_label_spans) {
        next_frontier_vertex_label_spans->push_back(
          raft::device_span<label_t const>{labels->data(), labels->size()});
      }
      // Propagate the per-seed window-end bound: look it up from the current frontier by
      // (source[, label]) and store it so it outlives this hop, then publish it as a span aligned
      // with the destination-vertex span recorded above.
      if (next_frontier_vertex_window_end_spans) {
        next_frontier_window_end_vectors.push_back(
          lookup_src_time_from_frontier<vertex_t, time_stamp_t, label_t>(
            handle,
            active_majors,
            active_labels,
            *active_window_ends,
            raft::device_span<vertex_t const>{srcs.data(), srcs.size()},
            labels
              ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
              : std::nullopt));
        next_frontier_vertex_window_end_spans->push_back(
          raft::device_span<time_stamp_t const>{next_frontier_window_end_vectors.back().data(),
                                                next_frontier_window_end_vectors.back().size()});
      }
      produced_edge_lists.emplace_back(std::move(srcs),
                                       std::move(dsts),
                                       std::move(props),
                                       std::move(labels),
                                       static_cast<int32_t>(hop));
    }

    if (gather_flags) {
      auto const n_gather_flags =
        static_cast<size_t>(num_edge_types ? *num_edge_types : edge_type_t{1});
      rmm::device_uvector<bool> d_gather_flags(n_gather_flags, handle.get_stream());
      raft::update_device(
        d_gather_flags.data(), gather_flags.get(), n_gather_flags, handle.get_stream());
      auto gather_flags_span = std::make_optional<raft::device_span<bool const>>(
        d_gather_flags.data(), d_gather_flags.size());

      rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
      rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
      std::vector<cugraph::arithmetic_device_uvector_t> props{};
      std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

      std::tie(srcs, dsts, props, labels, visited_minors, visited_minor_labels) =
        temporal_gather_one_hop_edgelist_to_unvisited_neighbors<vertex_t,
                                                                edge_t,
                                                                time_stamp_t,
                                                                multi_gpu>(
          handle,
          graph_view,
          edge_prop_span,
          edge_start_time_view,
          edge_type_filter_view,
          active_majors,
          active_window_starts,
          active_window_ends,
          active_labels,
          gather_flags_span,
          std::move(*visited_minors),
          std::move(visited_minor_labels),
          temporal_sampling_comparison,
          do_expensive_check);

      // See the level_Ks branch above: publish next-hop frontier spans, then store the edge list.
      next_frontier_vertex_spans.push_back(
        raft::device_span<vertex_t const>{dsts.data(), dsts.size()});
      if (fixed_window_starts) {
        // FIXED_WINDOW never replaces a seed's window-start with a sampled edge time: look it up
        // from the current frontier by (source[, label]) instead, same as the window-end lookup.
        next_frontier_window_start_vectors.push_back(
          lookup_src_time_from_frontier<vertex_t, time_stamp_t, label_t>(
            handle,
            active_majors,
            active_labels,
            *active_window_starts,
            raft::device_span<vertex_t const>{srcs.data(), srcs.size()},
            labels
              ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
              : std::nullopt));
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<time_stamp_t const>{next_frontier_window_start_vectors.back().data(),
                                                next_frontier_window_start_vectors.back().size()});
      } else {
        auto const& edge_start_times =
          std::get<rmm::device_uvector<time_stamp_t>>(props[edge_start_time_prop_idx]);
        next_frontier_vertex_time_spans->push_back(
          raft::device_span<time_stamp_t const>{edge_start_times.data(), edge_start_times.size()});
      }
      if (next_frontier_vertex_label_spans) {
        next_frontier_vertex_label_spans->push_back(
          raft::device_span<label_t const>{labels->data(), labels->size()});
      }
      // Propagate the per-seed window-end bound: look it up from the current frontier by
      // (source[, label]) and store it so it outlives this hop, then publish it as a span aligned
      // with the destination-vertex span recorded above.
      if (next_frontier_vertex_window_end_spans) {
        next_frontier_window_end_vectors.push_back(
          lookup_src_time_from_frontier<vertex_t, time_stamp_t, label_t>(
            handle,
            active_majors,
            active_labels,
            *active_window_ends,
            raft::device_span<vertex_t const>{srcs.data(), srcs.size()},
            labels
              ? std::make_optional(raft::device_span<label_t const>{labels->data(), labels->size()})
              : std::nullopt));
        next_frontier_vertex_window_end_spans->push_back(
          raft::device_span<time_stamp_t const>{next_frontier_window_end_vectors.back().data(),
                                                next_frontier_window_end_vectors.back().size()});
      }
      produced_edge_lists.emplace_back(std::move(srcs),
                                       std::move(dsts),
                                       std::move(props),
                                       std::move(labels),
                                       static_cast<int32_t>(hop));
    }

    // CARRY_OVER can mix unbounded current sources with destinations that have edge-derived
    // window starts. Materialize sentinels only for that mixed frontier; otherwise preserve the
    // missing column all the way to the temporal device filter.
    std::optional<rmm::device_uvector<time_stamp_t>> carry_over_window_starts{std::nullopt};
    auto sampled_src_window_starts = active_window_starts;
    if (sampling_flags.prior_sources_behavior == prior_sources_behavior_t::CARRY_OVER &&
        !sampled_src_window_starts && next_frontier_vertex_time_spans &&
        std::any_of(next_frontier_vertex_time_spans->begin(),
                    next_frontier_vertex_time_spans->end(),
                    [](auto span) { return span.size() > 0; })) {
      carry_over_window_starts =
        rmm::device_uvector<time_stamp_t>(frontier_vertices.size(), handle.get_stream());
      cugraph::fill(handle.get_thrust_policy(),
                    carry_over_window_starts->begin(),
                    carry_over_window_starts->end(),
                    unbounded_temporal_window_start<time_stamp_t>(temporal_sampling_comparison));
      sampled_src_window_starts = raft::device_span<time_stamp_t const>{
        carry_over_window_starts->data(), carry_over_window_starts->size()};
    }
    if (vertex_used_as_source && sampled_src_window_starts) {
      auto& [used_vertices, used_labels, used_window_starts] = *vertex_used_as_source;
      static_cast<void>(used_labels);
      if (used_window_starts->size() < used_vertices.size()) {
        auto const old_size = used_window_starts->size();
        used_window_starts->resize(used_vertices.size(), handle.get_stream());
        cugraph::fill(handle.get_thrust_policy(),
                      used_window_starts->begin() + old_size,
                      used_window_starts->end(),
                      unbounded_temporal_window_start<time_stamp_t>(temporal_sampling_comparison));
      }
    }

    std::tie(frontier_vertices,
             frontier_vertex_labels,
             frontier_vertex_times,
             frontier_vertex_window_ends,
             vertex_used_as_source) =
      prepare_next_frontier(
        handle,
        raft::device_span<vertex_t const>(frontier_vertices.data(), frontier_vertices.size()),
        frontier_vertex_labels ? std::make_optional(raft::device_span<label_t const>(
                                   frontier_vertex_labels->data(), frontier_vertex_labels->size()))
                               : std::nullopt,
        sampled_src_window_starts,
        frontier_vertex_window_ends
          ? std::make_optional<raft::device_span<time_stamp_t const>>(
              frontier_vertex_window_ends->data(), frontier_vertex_window_ends->size())
          : std::nullopt,
        raft::host_span<raft::device_span<vertex_t const>>{next_frontier_vertex_spans.data(),
                                                           next_frontier_vertex_spans.size()},
        next_frontier_vertex_label_spans
          ? std::make_optional(raft::host_span<raft::device_span<label_t const>>{
              next_frontier_vertex_label_spans->data(), next_frontier_vertex_label_spans->size()})
          : std::nullopt,
        next_frontier_vertex_time_spans
          ? std::make_optional(raft::host_span<raft::device_span<time_stamp_t const>>{
              next_frontier_vertex_time_spans->data(), next_frontier_vertex_time_spans->size()})
          : std::nullopt,
        next_frontier_vertex_window_end_spans
          ? std::make_optional(raft::host_span<raft::device_span<time_stamp_t const>>{
              next_frontier_vertex_window_end_spans->data(),
              next_frontier_vertex_window_end_spans->size()})
          : std::nullopt,
        std::move(vertex_used_as_source),
        graph_view.vertex_partition_range_lasts(),
        sampling_flags.prior_sources_behavior,
        sampling_flags.dedupe_sources,
        multi_gpu,
        do_expensive_check);
    dedupe_temporal_frontier<vertex_t, time_stamp_t, label_t>(handle,
                                                              frontier_vertices,
                                                              frontier_vertex_labels,
                                                              frontier_vertex_times,
                                                              frontier_vertex_window_ends,
                                                              temporal_sampling_comparison);
  }

  // Assemble the output by concatenating every produced edge list, in order.
  auto [result_srcs, result_dsts, result_properties, result_labels, result_hops] =
    concatenate_produced_edge_list_properties<vertex_t, label_t>(
      handle, std::move(produced_edge_lists), sampling_flags.return_hops);

  std::vector<cugraph::arithmetic_device_uvector_t> property_edges{};

  property_edges.push_back(std::move(result_srcs));
  property_edges.push_back(std::move(result_dsts));
  std::for_each(
    result_properties.begin(), result_properties.end(), [&property_edges](auto& property) {
      property_edges.push_back(std::move(property));
    });

  std::optional<rmm::device_uvector<size_t>> result_offsets{std::nullopt};

  auto output_labels = compute_output_labels<label_t, multi_gpu>(
    handle, starting_vertex_labels, label_to_output_comm_rank);

  std::tie(property_edges, result_labels, result_hops, result_offsets) =
    shuffle_and_organize_output(
      handle,
      std::move(property_edges),
      std::move(result_labels),
      std::move(result_hops),
      sampling_flags.return_hops ? std::make_optional<int32_t>(num_hops) : std::nullopt,
      output_labels ? std::make_optional(raft::device_span<label_t const>{output_labels->data(),
                                                                          output_labels->size()})
                    : std::nullopt,
      label_to_output_comm_rank);

  result_srcs = std::move(std::get<rmm::device_uvector<vertex_t>>(property_edges[0]));
  result_dsts = std::move(std::get<rmm::device_uvector<vertex_t>>(property_edges[1]));

  auto result_weights =
    weight_prop_idx
      ? std::make_optional(
          std::move(std::get<rmm::device_uvector<weight_t>>(property_edges[2 + *weight_prop_idx])))
      : std::nullopt;
  auto result_edge_ids =
    edge_id_prop_idx
      ? std::make_optional(
          std::move(std::get<rmm::device_uvector<edge_t>>(property_edges[2 + *edge_id_prop_idx])))
      : std::nullopt;
  auto result_edge_types =
    edge_type_prop_idx ? std::make_optional(std::move(std::get<rmm::device_uvector<edge_type_t>>(
                           property_edges[2 + *edge_type_prop_idx])))
                       : std::nullopt;
  auto result_edge_start_times = std::move(
    std::get<rmm::device_uvector<time_stamp_t>>(property_edges[2 + edge_start_time_prop_idx]));
  auto result_edge_end_times =
    edge_end_time_prop_idx
      ? std::make_optional(std::move(
          std::get<rmm::device_uvector<time_stamp_t>>(property_edges[2 + *edge_end_time_prop_idx])))
      : std::nullopt;

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
          typename time_stamp_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_uniform_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  // Specialized temporal APIs historically defaulted temporal_sampling_comparison to
  // STRICTLY_INCREASING; keep that when the optional is unset.
  if (!sampling_flags.temporal_sampling_comparison.has_value()) {
    sampling_flags.temporal_sampling_comparison =
      temporal_sampling_comparison_t::STRICTLY_INCREASING;
  }
  return neighbor_sample<vertex_t,
                         edge_t,
                         weight_t,
                         edge_type_t,
                         time_stamp_t,
                         store_transposed,
                         multi_gpu>(handle,
                                    rng_state,
                                    graph_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    edge_type_view,
                                    std::make_optional(edge_start_time_view),
                                    edge_end_time_view,
                                    std::nullopt,  // edge_bias_view
                                    starting_vertices,
                                    starting_vertex_start_times,
                                    starting_vertex_end_times,
                                    starting_vertex_labels,
                                    label_to_output_comm_rank,
                                    fan_out,
                                    std::nullopt,  // num_edge_types
                                    sampling_flags,
                                    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_uniform_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  if (!sampling_flags.temporal_sampling_comparison.has_value()) {
    sampling_flags.temporal_sampling_comparison =
      temporal_sampling_comparison_t::STRICTLY_INCREASING;
  }
  return neighbor_sample<vertex_t,
                         edge_t,
                         weight_t,
                         edge_type_t,
                         time_stamp_t,
                         store_transposed,
                         multi_gpu>(handle,
                                    rng_state,
                                    graph_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    std::make_optional(edge_type_view),
                                    std::make_optional(edge_start_time_view),
                                    edge_end_time_view,
                                    std::nullopt,  // edge_bias_view
                                    starting_vertices,
                                    starting_vertex_start_times,
                                    starting_vertex_end_times,
                                    starting_vertex_labels,
                                    label_to_output_comm_rank,
                                    fan_out,
                                    std::make_optional(num_edge_types),
                                    sampling_flags,
                                    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
homogeneous_biased_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  static_assert(std::is_same_v<bias_t, weight_t>);

  if (!sampling_flags.temporal_sampling_comparison.has_value()) {
    sampling_flags.temporal_sampling_comparison =
      temporal_sampling_comparison_t::STRICTLY_INCREASING;
  }
  return neighbor_sample<vertex_t,
                         edge_t,
                         weight_t,
                         edge_type_t,
                         time_stamp_t,
                         store_transposed,
                         multi_gpu>(handle,
                                    rng_state,
                                    graph_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    edge_type_view,
                                    std::make_optional(edge_start_time_view),
                                    edge_end_time_view,
                                    std::make_optional(edge_bias_view),
                                    starting_vertices,
                                    starting_vertex_start_times,
                                    starting_vertex_end_times,
                                    starting_vertex_labels,
                                    label_to_output_comm_rank,
                                    fan_out,
                                    std::nullopt,  // num_edge_types
                                    sampling_flags,
                                    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          typename bias_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
heterogeneous_biased_temporal_neighbor_sample(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_start_times,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_end_times,
  std::optional<raft::device_span<int32_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  edge_type_t num_edge_types,
  sampling_flags_t sampling_flags,
  bool do_expensive_check)
{
  static_assert(std::is_same_v<bias_t, weight_t>);

  if (!sampling_flags.temporal_sampling_comparison.has_value()) {
    sampling_flags.temporal_sampling_comparison =
      temporal_sampling_comparison_t::STRICTLY_INCREASING;
  }
  return neighbor_sample<vertex_t,
                         edge_t,
                         weight_t,
                         edge_type_t,
                         time_stamp_t,
                         store_transposed,
                         multi_gpu>(handle,
                                    rng_state,
                                    graph_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    std::make_optional(edge_type_view),
                                    std::make_optional(edge_start_time_view),
                                    edge_end_time_view,
                                    std::make_optional(edge_bias_view),
                                    starting_vertices,
                                    starting_vertex_start_times,
                                    starting_vertex_end_times,
                                    starting_vertex_labels,
                                    label_to_output_comm_rank,
                                    fan_out,
                                    std::make_optional(num_edge_types),
                                    sampling_flags,
                                    do_expensive_check);
}

}  // namespace cugraph
