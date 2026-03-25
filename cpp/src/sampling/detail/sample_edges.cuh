/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gather_sampled_properties.hpp"
#include "prims/edge_bucket.cuh"
#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/transform_gather_e.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "sampling_utils.hpp"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/sort.h>

#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {
namespace detail {

struct return_edge_property_t {
  template <typename key_t, typename vertex_t, typename T>
  T __device__
  operator()(key_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, T edge_property) const
  {
    return edge_property;
  }
};

template <typename vertex_t, typename edge_properties_t>
struct sample_edges_op_t {
  using return_type = std::conditional_t<std::is_same_v<edge_properties_t, cuda::std::nullopt_t>,
                                         cuda::std::tuple<vertex_t, vertex_t>,
                                         cuda::std::tuple<vertex_t, vertex_t, edge_properties_t>>;

  template <typename key_t>
  return_type __device__ operator()(key_t optionally_tagged_major,
                                    vertex_t minor,
                                    cuda::std::nullopt_t,
                                    cuda::std::nullopt_t,
                                    edge_properties_t edge_properties) const
  {
    vertex_t major{};

    if constexpr (std::is_same_v<key_t, vertex_t>)
      major = optionally_tagged_major;
    else
      major = cuda::std::get<0>(optionally_tagged_major);

    if constexpr (std::is_same_v<edge_properties_t, cuda::std::nullopt_t>) {
      return cuda::std::make_tuple(major, minor);
    } else {
      return cuda::std::make_tuple(major, minor, edge_properties);
    }
  }
};

template <typename vertex_t, typename bias_t>
struct sample_edge_biases_op_t {
  template <typename key_t>
  bias_t __device__
  operator()(key_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, bias_t bias) const
  {
    return bias;
  }
};

template <typename vertex_t, typename bias_t>
struct sample_unvisited_edge_biases_op_t {
  raft::device_span<vertex_t const> visited_minors{};
  cuda::std::optional<raft::device_span<int32_t const>> visited_minor_labels{};

  template <typename edge_property_t>
  bias_t __device__ operator()(vertex_t,
                               vertex_t minor,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               edge_property_t edge_property) const
  {
    bias_t bias{1};
    if constexpr (std::is_same_v<edge_property_t, bias_t>) { bias = edge_property; }

    return thrust::binary_search(thrust::seq, visited_minors.begin(), visited_minors.end(), minor)
             ? bias_t{0}
             : bias;
  }

  template <typename edge_property_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, int32_t> tagged_major,
                               vertex_t minor,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               edge_property_t edge_property) const
  {
    bias_t bias{1};
    if constexpr (std::is_same_v<edge_property_t, bias_t>) { bias = edge_property; }

    return thrust::binary_search(
             thrust::seq,
             thrust::make_zip_iterator(visited_minors.begin(), visited_minor_labels->begin()),
             thrust::make_zip_iterator(visited_minors.end(), visited_minor_labels->end()),
             cuda::std::make_tuple(minor, cuda::std::get<1>(tagged_major)))
             ? bias_t{0}
             : bias;
  }
};

template <typename vertex_t, typename bias_t>
struct temporal_sample_edge_biases_op_t {
  temporal_sampling_comparison_t temporal_sampling_comparison{};

  template <typename time_stamp_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, time_stamp_t> tagged_major,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               bias_t) const
  {
    // Should not happen at runtime
    return bias_t{0};
  }

  template <typename time_stamp_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, time_stamp_t> tagged_major,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               time_stamp_t edge_time) const
  {
    switch (temporal_sampling_comparison) {
      case temporal_sampling_comparison_t::STRICTLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) < edge_time) ? bias_t{1} : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) <= edge_time) ? bias_t{1} : bias_t{0};
      case temporal_sampling_comparison_t::STRICTLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) > edge_time) ? bias_t{1} : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) >= edge_time) ? bias_t{1} : bias_t{0};
    }
    return bias_t{0};
  }

  template <typename time_stamp_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, time_stamp_t> tagged_major,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               cuda::std::tuple<bias_t, time_stamp_t> bias_and_time) const
  {
    switch (temporal_sampling_comparison) {
      case temporal_sampling_comparison_t::STRICTLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) < cuda::std::get<1>(bias_and_time))
                 ? cuda::std::get<0>(bias_and_time)
                 : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) <= cuda::std::get<1>(bias_and_time))
                 ? cuda::std::get<0>(bias_and_time)
                 : bias_t{0};
      case temporal_sampling_comparison_t::STRICTLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) > cuda::std::get<1>(bias_and_time))
                 ? cuda::std::get<0>(bias_and_time)
                 : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) >= cuda::std::get<1>(bias_and_time))
                 ? cuda::std::get<0>(bias_and_time)
                 : bias_t{0};
    }
    return bias_t{0};
  }

  template <typename time_stamp_t,
            typename edge_type_t,
            typename std::enable_if_t<std::is_integral_v<edge_type_t>>* = nullptr>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, time_stamp_t> tagged_major,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               cuda::std::tuple<time_stamp_t, edge_type_t> time_and_type) const
  {
    switch (temporal_sampling_comparison) {
      case temporal_sampling_comparison_t::STRICTLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) < cuda::std::get<0>(time_and_type)) ? bias_t{1}
                                                                                    : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) <= cuda::std::get<0>(time_and_type)) ? bias_t{1}
                                                                                     : bias_t{0};
      case temporal_sampling_comparison_t::STRICTLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) > cuda::std::get<0>(time_and_type)) ? bias_t{1}
                                                                                    : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) >= cuda::std::get<0>(time_and_type)) ? bias_t{1}
                                                                                     : bias_t{0};
    }
    return bias_t{0};
  }

  template <typename time_stamp_t, typename edge_type_t>
  bias_t __device__
  operator()(cuda::std::tuple<vertex_t, time_stamp_t> tagged_major,
             vertex_t,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             cuda::std::tuple<bias_t, time_stamp_t, edge_type_t> bias_time_and_type) const
  {
    switch (temporal_sampling_comparison) {
      case temporal_sampling_comparison_t::STRICTLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) < cuda::std::get<1>(bias_time_and_type))
                 ? cuda::std::get<0>(bias_time_and_type)
                 : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
        return (cuda::std::get<1>(tagged_major) <= cuda::std::get<1>(bias_time_and_type))
                 ? cuda::std::get<0>(bias_time_and_type)
                 : bias_t{0};
      case temporal_sampling_comparison_t::STRICTLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) > cuda::std::get<1>(bias_time_and_type))
                 ? cuda::std::get<0>(bias_time_and_type)
                 : bias_t{0};
      case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
        return (cuda::std::get<1>(tagged_major) >= cuda::std::get<1>(bias_time_and_type))
                 ? cuda::std::get<0>(bias_time_and_type)
                 : bias_t{0};
    }
    return bias_t{0};
  }
};

struct segmented_fill_t {
  raft::device_span<int32_t const> fill_values{};
  raft::device_span<size_t const> segment_offsets{};
  raft::device_span<int32_t> output_values{};

  __device__ void operator()(size_t i) const
  {
    thrust::fill(thrust::seq,
                 output_values.begin() + segment_offsets[i],
                 output_values.begin() + segment_offsets[i + 1],
                 fill_values[i]);
  }
};

/**
 * Helper function for random sampling of outgoing edges with a custom bias operator, used in
 * homogeneous and heterogeneous sampling functions.  It can be called with different combinations
 * of edge_property_view for different purposes as described below.
 *
 * 1. Standard biased sampling (sample_with_one_property): biases_op is sample_edge_biases_op_t;
 *    edge_biases_view holds float or double weights. Optional edge_type_view for heterogeneous
 *    type filtering. Used when sampling neighbors without an "unvisited" constraint.
 *
 * 2. Unvisited-neighbor biased sampling (sample_unvisited_with_one_property): biases_op is
 *    sample_unvisited_edge_biases_op_t, which down-weights or excludes already-visited minors
 *    (and optionally uses edge weights). edge_biases_view may be a real weight view or
 *    edge_dummy_property_view_t when no edge weights are used.
 *
 * 3. Temporal sampling (temporal_sample_with_one_property): biases_op is
 *    temporal_sample_edge_biases_op_t; edge_biases_view is either a concatenated (bias, time)
 *    view or time-only view. Selection respects temporal_sampling_comparison (e.g. strictly
 *    increasing time).
 */
template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename biases_view_t,
          typename property_view_t,
          typename edge_type_t,
          typename biases_op_t,
          bool multi_gpu>
std::tuple<std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t>
call_biased_per_v_random_select_transform_outgoing_e(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> const& vertex_frontier,
  biases_view_t edge_biases_view,
  property_view_t edge_property_view,
  biases_op_t biases_op,
  std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};

  using T          = typename decltype(edge_property_view)::value_type;
  using edges_op_t = sample_edges_op_t<vertex_t, T>;
  edges_op_t edges_op{};
  using return_type = typename edges_op_t::return_type;

  if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
    std::forward_as_tuple(offsets, std::tie(majors, minors)) =
      (Ks.size() == 1) ? cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_biases_view,
                           biases_op,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           cugraph::edge_dummy_property_view_t{},
                           edges_op,
                           rng_state,
                           Ks[0],
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false)
                       : cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_biases_view,
                           biases_op,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           cugraph::edge_dummy_property_view_t{},
                           edges_op,
                           *edge_type_view,
                           rng_state,
                           Ks,
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false);
  } else {
    std::forward_as_tuple(offsets, std::tie(majors, minors, sampled_properties)) =
      (Ks.size() == 1) ? cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_biases_view,
                           biases_op,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           rng_state,
                           Ks[0],
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false)
                       : cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_biases_view,
                           biases_op,
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           *edge_type_view,
                           rng_state,
                           Ks,
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false);
  }

  if (offsets) {
    labels =
      rmm::device_uvector<int32_t>(offsets->back_element(handle.get_stream()), handle.get_stream());
    auto num_segments = offsets->size() - size_t{1};
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_segments),
      segmented_fill_t{*active_major_labels,
                       raft::device_span<size_t const>{offsets->data(), offsets->size()},
                       raft::device_span<int32_t>{labels->data(), labels->size()}});
  }

  return std::make_tuple(
    std::move(labels), std::move(majors), std::move(minors), std::move(sampled_properties));
}

/**
 * Helper function for random sampling of outgoing edges without a bias operator, used in
 * homogeneous and heterogeneous sampling functions.
 */
template <typename vertex_t,
          typename edge_t,
          typename property_view_t,
          typename edge_type_t,
          bool multi_gpu>
std::tuple<std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t>
call_unbiased_per_v_random_select_transform_outgoing_e(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> const& vertex_frontier,
  property_view_t edge_property_view,
  std::optional<cugraph::edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement)
{
  using T          = typename decltype(edge_property_view)::value_type;
  using edges_op_t = sample_edges_op_t<vertex_t, T>;
  edges_op_t edges_op{};
  using return_type = typename edges_op_t::return_type;

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

  if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
    std::forward_as_tuple(offsets, std::tie(majors, minors)) =
      (Ks.size() == 1) ? cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           rng_state,
                           Ks[0],
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false)
                       : cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           *edge_type_view,
                           rng_state,
                           Ks,
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false);
  } else {
    std::forward_as_tuple(offsets, std::tie(majors, minors, sampled_properties)) =
      (Ks.size() == 1) ? cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           rng_state,
                           Ks[0],
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false)
                       : cugraph::per_v_random_select_transform_outgoing_e(
                           handle,
                           graph_view,
                           vertex_frontier.bucket(0),
                           edge_src_dummy_property_t{}.view(),
                           edge_dst_dummy_property_t{}.view(),
                           edge_property_view,
                           edges_op,
                           *edge_type_view,
                           rng_state,
                           Ks,
                           with_replacement,
                           std::optional<return_type>{std::nullopt},
                           false);
  }

  if (active_major_labels) {
    labels =
      rmm::device_uvector<int32_t>(offsets->back_element(handle.get_stream()), handle.get_stream());
    auto num_segments = offsets->size() - size_t{1};
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(num_segments),
      segmented_fill_t{*active_major_labels,
                       raft::device_span<size_t const>{offsets->data(), offsets->size()},
                       raft::device_span<int32_t>{labels->data(), labels->size()}});
  }

  return std::make_tuple(
    std::move(labels), std::move(majors), std::move(minors), std::move(sampled_properties));
}

/**
 * Values passed as edge_property_view to sample_with_one_property,
 * sample_unvisited_with_one_property, and temporal_sample_with_one_property (and thus to
 * call_biased_* / call_unbiased_*):
 *
 * - edge_dummy_property_view_t{}: When edge_property_views.size() == 0 (no properties to return),
 * or when edge_property_views.size() > 1 and the graph is not a multigraph (we only need topology;
 *   gather_sampled_properties fetches all properties afterward using multi_index from the single
 *   call that used a dummy view).
 *
 * - The single element of edge_property_views[0] (via variant_type_dispatch): When
 *   edge_property_views.size() == 1. Type is one of edge_property_view_t<edge_t, T const*> for
 *   T in {float, double, int32_t, int64_t, size_t}.
 *
 * - multi_index_property.view(): When edge_property_views.size() > 1 and
 * graph_view.is_multigraph(). Used to sample multi-edge indices so that gather_sampled_properties
 * can then gather all requested edge properties.
 */
template <typename vertex_t, typename edge_t, typename property_view_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false>& vertex_frontier,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement)
{
  using edge_type_t = int32_t;
  using T           = typename decltype(edge_property_view)::value_type;

  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};

  if (edge_bias_view) {
    if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
          *edge_bias_view)) {
      using bias_t = float;

      std::tie(labels, majors, minors, sampled_properties) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
          edge_property_view,
          sample_edge_biases_op_t<vertex_t, bias_t>{},
          edge_type_view ? std::make_optional(
                             std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                               *edge_type_view))
                         : std::nullopt,
          rng_state,
          Ks,
          active_major_labels,
          with_replacement);
    } else if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, double const*>>(
                 *edge_bias_view)) {
      using bias_t = double;

      std::tie(labels, majors, minors, sampled_properties) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
          edge_property_view,
          sample_edge_biases_op_t<vertex_t, bias_t>{},
          edge_type_view ? std::make_optional(
                             std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                               *edge_type_view))
                         : std::nullopt,
          rng_state,
          Ks,
          active_major_labels,
          with_replacement);
    }
  } else {
    std::tie(labels, majors, minors, sampled_properties) =
      call_unbiased_per_v_random_select_transform_outgoing_e(
        handle,
        graph_view,
        vertex_frontier,
        edge_property_view,
        edge_type_view
          ? std::make_optional(
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view))
          : std::nullopt,
        rng_state,
        Ks,
        active_major_labels,
        with_replacement);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(sampled_properties), std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename property_view_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_unvisited_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false>& vertex_frontier,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  raft::host_span<size_t const> Ks,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  bool with_replacement)
{
  using edge_type_t = int32_t;
  using T           = typename decltype(edge_property_view)::value_type;

  rmm::device_uvector<vertex_t> result_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_minors(0, handle.get_stream());
  arithmetic_device_uvector_t result_properties{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  bool sample_and_append{true};
  std::optional<rmm::device_uvector<vertex_t>> resample_active_majors{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> resample_active_major_labels{std::nullopt};

  // FIXME: We could explore increasing the rate of convergency by oversampling to allow
  // for some duplicates to be discarded.  This would allow some vertices to still have the
  // first sampling result be sufficient.  For now we'll leave this as a future optimization.
  while (sample_and_append) {
    std::optional<rmm::device_uvector<int32_t>> sampled_labels{std::nullopt};
    rmm::device_uvector<vertex_t> sampled_majors(0, handle.get_stream());
    rmm::device_uvector<vertex_t> sampled_minors(0, handle.get_stream());
    arithmetic_device_uvector_t sampled_property{std::monostate{}};

    if (edge_bias_view) {
      if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
            *edge_bias_view)) {
        using bias_t = float;

        std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_property) =
          call_biased_per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier,
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_property_view,
            sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
              raft::device_span<vertex_t const>{visited_minors.data(), visited_minors.size()},
              raft::device_span<int32_t const>{visited_minor_labels->data(),
                                               visited_minor_labels->size()}},
            edge_type_view ? std::make_optional(
                               std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                                 *edge_type_view))
                           : std::nullopt,
            rng_state,
            Ks,
            active_major_labels,
            with_replacement);
      } else if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, double const*>>(
                   *edge_bias_view)) {
        using bias_t = double;

        std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_property) =
          call_biased_per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier,
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_property_view,
            sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
              raft::device_span<vertex_t const>{visited_minors.data(), visited_minors.size()},
              raft::device_span<int32_t const>{visited_minor_labels->data(),
                                               visited_minor_labels->size()}},
            edge_type_view ? std::make_optional(
                               std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                                 *edge_type_view))
                           : std::nullopt,
            rng_state,
            Ks,
            active_major_labels,
            with_replacement);
      }
    } else {
      using bias_t = float;

      std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_property) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          cugraph::edge_dummy_property_view_t{},
          edge_property_view,
          sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
            raft::device_span<vertex_t const>{visited_minors.data(), visited_minors.size()},
            raft::device_span<int32_t const>{visited_minor_labels->data(),
                                             visited_minor_labels->size()}},
          edge_type_view ? std::make_optional(
                             std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                               *edge_type_view))
                         : std::nullopt,
          rng_state,
          Ks,
          active_major_labels,
          with_replacement);
    }

    // Check for duplicates in the sampled minor vertices
    std::tie(sampled_majors,
             sampled_minors,
             sampled_property,
             sampled_labels,
             resample_active_majors,
             resample_active_major_labels) = deduplicate_edges_by_minor(handle,
                                                                        graph_view,
                                                                        std::move(sampled_majors),
                                                                        std::move(sampled_minors),
                                                                        std::move(sampled_property),
                                                                        std::move(sampled_labels),
                                                                        true);

    std::tie(visited_minors, visited_minor_labels) =
      detail::update_dst_visited_vertices_and_labels<vertex_t, edge_t, multi_gpu>(
        handle,
        graph_view,
        std::move(visited_minors),
        std::move(visited_minor_labels),
        raft::device_span<vertex_t const>{sampled_minors.data(), sampled_minors.size()},
        sampled_labels ? std::make_optional(raft::device_span<int32_t const>{
                           sampled_labels->data(), sampled_labels->size()})
                       : std::nullopt);

    if constexpr (multi_gpu) {
      sample_and_append =
        (host_scalar_allreduce(handle.get_comms(),
                               (resample_active_majors ? resample_active_majors->size() : 0),
                               raft::comms::op_t::SUM,
                               handle.get_stream()) > 0);
    } else {
      sample_and_append = (resample_active_majors ? resample_active_majors->size() : 0) > 0;
    }

    if (sample_and_append) {
      if constexpr (std::is_same_v<tag_t, void>) {
        vertex_frontier.bucket(0) = cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false>(
          handle,
          raft::device_span<vertex_t const>{resample_active_majors->data(),
                                            resample_active_majors->size()});
      } else {
        vertex_frontier.bucket(0) = cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false>(
          handle,
          raft::device_span<vertex_t const>{resample_active_majors->data(),
                                            resample_active_majors->size()},
          raft::device_span<tag_t const>{resample_active_major_labels->data(),
                                         resample_active_major_labels->size()});
        handle.sync_stream();
        active_major_labels = raft::device_span<int32_t const>{
          resample_active_major_labels->data(), resample_active_major_labels->size()};
      }
    }

    // Append the latest sampled edges to the output
    size_t original_result_majors_size = result_majors.size();
    result_majors.resize(result_majors.size() + sampled_majors.size(), handle.get_stream());
    raft::copy(result_majors.data() + original_result_majors_size,
               sampled_majors.data(),
               sampled_majors.size(),
               handle.get_stream());

    result_minors.resize(result_minors.size() + sampled_minors.size(), handle.get_stream());
    raft::copy(result_minors.data() + original_result_majors_size,
               sampled_minors.data(),
               sampled_minors.size(),
               handle.get_stream());

    if constexpr (!std::is_same_v<T, cuda::std::nullopt_t>) {
      if (std::holds_alternative<std::monostate>(result_properties)) {
        result_properties = std::move(sampled_property);
      } else {
        auto& result_properties_ref = std::get<rmm::device_uvector<T>>(result_properties);
        auto& sampled_property_ref  = std::get<rmm::device_uvector<T>>(sampled_property);
        result_properties_ref.resize(result_properties_ref.size() + sampled_property_ref.size(),
                                     handle.get_stream());
        raft::copy(result_properties_ref.data() + original_result_majors_size,
                   sampled_property_ref.data(),
                   sampled_property_ref.size(),
                   handle.get_stream());
      }
    }

    if (visited_minor_labels) {
      if (result_labels) {
        result_labels->resize(result_labels->size() + sampled_labels->size(), handle.get_stream());
        raft::copy(result_labels->data() + original_result_majors_size,
                   sampled_labels->data(),
                   sampled_labels->size(),
                   handle.get_stream());
      } else {
        result_labels = std::move(sampled_labels);
      }
    }
  }

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(result_properties),
                         std::move(result_labels),
                         std::move(visited_minors),
                         std::move(visited_minor_labels));
}

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

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> sample_labels{std::nullopt};

  if (number_of_edge_properties == 0) {
    std::tie(majors, minors, std::ignore, sample_labels) =
      sample_with_one_property(handle,
                               rng_state,
                               graph_view,
                               cugraph::edge_dummy_property_view_t{},
                               edge_type_view,
                               edge_bias_view,
                               vertex_frontier,
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
                                 vertex_frontier,
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
                                 vertex_frontier,
                                 Ks,
                                 active_major_labels,
                                 with_replacement);
    }
  }

  labels = std::move(sample_labels);

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(tmp_edge_indices), std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename property_view_t,
          typename time_stamp_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, time_stamp_t, multi_gpu, false>& vertex_frontier,
  raft::host_span<size_t const> Ks,
  bool with_replacement,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  using edge_type_t = int32_t;

  using T = typename decltype(edge_property_view)::value_type;

  std::optional<rmm::device_uvector<int32_t>> sample_labels{std::nullopt};
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};

  if (edge_bias_view) {
    if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
          *edge_bias_view)) {
      using bias_t = float;

      std::tie(sample_labels, majors, minors, sampled_properties) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          view_concat(
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_time_view),
          edge_property_view,
          temporal_sample_edge_biases_op_t<vertex_t, bias_t>{temporal_sampling_comparison},
          edge_type_view ? std::make_optional(
                             std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                               *edge_type_view))
                         : std::nullopt,
          rng_state,
          Ks,
          active_major_labels,
          with_replacement);
    } else if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, double const*>>(
                 *edge_bias_view)) {
      using bias_t = double;

      std::tie(sample_labels, majors, minors, sampled_properties) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          view_concat(
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_time_view),
          edge_property_view,
          temporal_sample_edge_biases_op_t<vertex_t, bias_t>{temporal_sampling_comparison},
          edge_type_view ? std::make_optional(
                             std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                               *edge_type_view))
                         : std::nullopt,
          rng_state,
          Ks,
          active_major_labels,
          with_replacement);
    }
  } else {
    using bias_t = float;

    std::tie(sample_labels, majors, minors, sampled_properties) =
      call_biased_per_v_random_select_transform_outgoing_e(
        handle,
        graph_view,
        vertex_frontier,
        edge_time_view,
        edge_property_view,
        temporal_sample_edge_biases_op_t<vertex_t, bias_t>{temporal_sampling_comparison},
        edge_type_view
          ? std::make_optional(
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view))
          : std::nullopt,
        rng_state,
        Ks,
        active_major_labels,
        with_replacement);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(sampled_properties), std::move(sample_labels));
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

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(active_majors.begin(), active_major_times.begin()),
    thrust::make_zip_iterator(active_majors.end(), active_major_times.end()));

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
                                        vertex_frontier,
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
                                        vertex_frontier,
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

    vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    if (number_of_edge_properties == 0) {
      std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           vertex_frontier,
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
                                             vertex_frontier,
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
                                             vertex_frontier,
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

    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    if (number_of_edge_properties == 0) {
      std::tie(majors, minors, std::ignore, labels, visited_minors, visited_minor_labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           vertex_frontier,
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
                                             vertex_frontier,
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
                                             vertex_frontier,
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
