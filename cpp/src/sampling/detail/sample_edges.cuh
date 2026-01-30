/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gather_sampled_properties.cuh"
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
  raft::device_span<vertex_t const> visited_vertices{};
  cuda::std::optional<raft::device_span<int32_t const>> visited_vertex_labels{};

  template <typename edge_property_t>
  bias_t __device__ operator()(vertex_t,
                               vertex_t minor,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               edge_property_t edge_property) const
  {
    bias_t bias{1};
    if constexpr (std::is_same_v<edge_property_t, bias_t>) { bias = edge_property; }

    return thrust::binary_search(
             thrust::seq, visited_vertices.begin(), visited_vertices.end(), minor)
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
             thrust::make_zip_iterator(visited_vertices.begin(), visited_vertex_labels->begin()),
             thrust::make_zip_iterator(visited_vertices.end(), visited_vertex_labels->end()),
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
                           true)
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
           std::optional<rmm::device_uvector<int32_t>>>
sample_unvisited_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false>& vertex_frontier,
  std::optional<rmm::device_uvector<vertex_t>>& visited_vertices,
  std::optional<rmm::device_uvector<int32_t>>& visited_vertex_labels,
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
  rmm::device_uvector<vertex_t> resample_active_majors(0, handle.get_stream());
  rmm::device_uvector<int32_t> resample_active_major_labels(0, handle.get_stream());

  // FIXME: We could explore increasing the rate of convergency by oversampling to allow
  // for some duplicates to be discarded.  This would allow some vertices to still have the
  // first sampling result be sufficient.  For now we'll leave this as a future optimization.
  while (sample_and_append) {
    std::optional<rmm::device_uvector<int32_t>> sampled_labels{std::nullopt};
    rmm::device_uvector<vertex_t> sampled_majors(0, handle.get_stream());
    rmm::device_uvector<vertex_t> sampled_minors(0, handle.get_stream());
    arithmetic_device_uvector_t sampled_properties{std::monostate{}};

    if (edge_bias_view) {
      if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
            *edge_bias_view)) {
        using bias_t = float;

        std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_properties) =
          call_biased_per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier,
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_property_view,
            sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
              raft::device_span<vertex_t const>{visited_vertices->data(), visited_vertices->size()},
              raft::device_span<int32_t const>{visited_vertex_labels->data(),
                                               visited_vertex_labels->size()}},
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

        std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_properties) =
          call_biased_per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier,
            std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
            edge_property_view,
            sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
              raft::device_span<vertex_t const>{visited_vertices->data(), visited_vertices->size()},
              raft::device_span<int32_t const>{visited_vertex_labels->data(),
                                               visited_vertex_labels->size()}},
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

      std::tie(sampled_labels, sampled_majors, sampled_minors, sampled_properties) =
        call_biased_per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier,
          cugraph::edge_dummy_property_view_t{},
          edge_property_view,
          sample_unvisited_edge_biases_op_t<vertex_t, bias_t>{
            raft::device_span<vertex_t const>{visited_vertices->data(), visited_vertices->size()},
            raft::device_span<int32_t const>{visited_vertex_labels->data(),
                                             visited_vertex_labels->size()}},
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
    rmm::device_uvector<vertex_t> local_majors(sampled_majors.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> local_minors(sampled_minors.size(), handle.get_stream());
    rmm::device_uvector<int32_t> local_labels(0, handle.get_stream());

    raft::copy(
      local_majors.data(), sampled_majors.data(), sampled_majors.size(), handle.get_stream());
    raft::copy(
      local_minors.data(), sampled_minors.data(), sampled_minors.size(), handle.get_stream());
    if (visited_vertex_labels) {
      local_labels.resize(sampled_labels->size(), handle.get_stream());
      raft::copy(
        local_labels.data(), sampled_labels->data(), sampled_labels->size(), handle.get_stream());
    }

    if constexpr (multi_gpu) {
      std::vector<cugraph::arithmetic_device_uvector_t> properties_to_shuffle{};
      properties_to_shuffle.push_back(std::move(local_majors));
      if (visited_vertex_labels) properties_to_shuffle.push_back(std::move(local_labels));

      std::tie(local_minors, properties_to_shuffle) =
        cugraph::shuffle_int_vertices(handle,
                                      std::move(local_minors),
                                      std::move(properties_to_shuffle),
                                      graph_view.vertex_partition_range_lasts());

      local_majors = std::move(std::get<rmm::device_uvector<vertex_t>>(properties_to_shuffle[0]));
      if (visited_vertex_labels)
        local_labels = std::move(std::get<rmm::device_uvector<int32_t>>(properties_to_shuffle[1]));
    }

    if (visited_vertex_labels) {
      thrust::sort(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(local_labels.begin(), local_minors.begin(), local_majors.begin()),
        thrust::make_zip_iterator(local_labels.end(), local_minors.end(), local_majors.end()));
    } else {
      thrust::sort(handle.get_thrust_policy(),
                   thrust::make_zip_iterator(local_minors.begin(), local_majors.begin()),
                   thrust::make_zip_iterator(local_minors.end(), local_majors.end()));
    }

    size_t duplicate_count{0};
    rmm::device_uvector<uint32_t> duplicate_flags(0, handle.get_stream());

    if (visited_vertex_labels) {
      auto labels_minor_begin =
        thrust::make_zip_iterator(local_labels.begin(), local_minors.begin());
      std::tie(duplicate_count, duplicate_flags) =
        detail::mark_entries(handle,
                             local_minors.size(),
                             [func = detail::is_first_in_run_t<decltype(labels_minor_begin)>{
                                labels_minor_begin}] __device__(size_t i) { return !func(i); });
    } else {
      std::tie(duplicate_count, duplicate_flags) = detail::mark_entries(
        handle,
        local_minors.size(),
        [func = detail::is_first_in_run_t<vertex_t const*>{local_minors.data()}] __device__(
          size_t i) { return !func(i); });
    }

    local_minors.resize(0, handle.get_stream());
    local_minors.shrink_to_fit(handle.get_stream());

    // We'll have to rerun the sampling on a smaller vertex frontier if there are duplicates
    if constexpr (multi_gpu) {
      sample_and_append =
        (host_scalar_allreduce(
           handle.get_comms(), duplicate_count, raft::comms::op_t::SUM, handle.get_stream()) > 0);
    } else {
      sample_and_append = (duplicate_count > 0);
    }

    if (duplicate_count > 0) {
      local_majors = detail::keep_marked_entries(
        handle,
        std::move(local_majors),
        raft::device_span<uint32_t const>{duplicate_flags.data(), duplicate_flags.size()},
        duplicate_count);

      if (visited_vertex_labels) {
        local_labels = detail::keep_marked_entries(
          handle,
          std::move(local_labels),
          raft::device_span<uint32_t const>{duplicate_flags.data(), duplicate_flags.size()},
          duplicate_count);
      }

      duplicate_flags.resize(0, handle.get_stream());
      duplicate_flags.shrink_to_fit(handle.get_stream());
    } else {
      local_majors.resize(0, handle.get_stream());
      local_majors.shrink_to_fit(handle.get_stream());
      if (visited_vertex_labels) {
        local_labels.resize(0, handle.get_stream());
        local_labels.shrink_to_fit(handle.get_stream());
      }
    }

    if (visited_vertex_labels) {
      auto labels_major_begin =
        thrust::make_zip_iterator(local_labels.begin(), local_majors.begin());
      thrust::sort(
        handle.get_thrust_policy(), labels_major_begin, labels_major_begin + local_majors.size());
      auto new_end = thrust::unique(
        handle.get_thrust_policy(), labels_major_begin, labels_major_begin + local_majors.size());
      local_majors.resize(cuda::std::distance(labels_major_begin, new_end), handle.get_stream());
      local_labels.resize(cuda::std::distance(labels_major_begin, new_end), handle.get_stream());
    } else {
      auto majors_begin = local_majors.begin();
      thrust::sort(handle.get_thrust_policy(), majors_begin, majors_begin + local_majors.size());
      auto new_end = thrust::unique(
        handle.get_thrust_policy(), majors_begin, majors_begin + local_majors.size());
      local_majors.resize(cuda::std::distance(majors_begin, new_end), handle.get_stream());
    }

    if constexpr (multi_gpu) {
      std::vector<cugraph::arithmetic_device_uvector_t> properties_to_shuffle{};
      if (visited_vertex_labels) properties_to_shuffle.push_back(std::move(local_labels));

      std::tie(local_majors, properties_to_shuffle) =
        cugraph::shuffle_int_vertices(handle,
                                      std::move(local_majors),
                                      std::move(properties_to_shuffle),
                                      graph_view.vertex_partition_range_lasts());

      if (visited_vertex_labels) {
        local_labels = std::move(std::get<rmm::device_uvector<int32_t>>(properties_to_shuffle[0]));

        auto labels_major_begin =
          thrust::make_zip_iterator(local_labels.begin(), local_majors.begin());
        thrust::sort(
          handle.get_thrust_policy(), labels_major_begin, labels_major_begin + local_majors.size());
        auto new_end = thrust::unique(
          handle.get_thrust_policy(), labels_major_begin, labels_major_begin + local_majors.size());
        local_majors.resize(cuda::std::distance(labels_major_begin, new_end), handle.get_stream());
        local_labels.resize(cuda::std::distance(labels_major_begin, new_end), handle.get_stream());

      } else {
        auto majors_begin = local_majors.begin();
        thrust::sort(handle.get_thrust_policy(), majors_begin, majors_begin + local_majors.size());
        auto new_end = thrust::unique(
          handle.get_thrust_policy(), majors_begin, majors_begin + local_majors.size());
        local_majors.resize(cuda::std::distance(majors_begin, new_end), handle.get_stream());
      }

      if (local_majors.size() > 0) {
        // Now we need to delete sampled edges from any vertex in local_majors
        size_t keep_count{};
        rmm::device_uvector<uint32_t> keep_flags(0, handle.get_stream());

        if (visited_vertex_labels) {
          auto labels_major_begin =
            thrust::make_zip_iterator(local_labels.begin(), local_majors.begin());
          std::tie(keep_count, keep_flags) = detail::mark_entries(
            handle,
            sampled_majors.size(),
            [majors             = sampled_majors.data(),
             labels             = sampled_labels->data(),
             labels_major_begin = labels_major_begin,
             labels_major_end   = labels_major_begin + local_majors.size()] __device__(size_t i) {
              return !thrust::binary_search(thrust::seq,
                                            labels_major_begin,
                                            labels_major_end,
                                            cuda::std::make_tuple(labels[i], majors[i]));
            });
        } else {
          std::tie(keep_count, keep_flags) = detail::mark_entries(
            handle,
            sampled_majors.size(),
            [majors       = sampled_majors.data(),
             majors_begin = local_majors.begin(),
             majors_end   = local_majors.end()] __device__(size_t i) {
              return !thrust::binary_search(thrust::seq, majors_begin, majors_end, majors[i]);
            });
        }

        // local_majors/local_labels contain the vertices that need to be resampled
        // Replace the current frontier (bucket 0) with the new set
        if constexpr (std::is_same_v<tag_t, void>) {
          resample_active_majors    = std::move(local_majors);
          vertex_frontier.bucket(0) = cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false>(
            handle,
            raft::device_span<vertex_t const>{resample_active_majors.data(),
                                              resample_active_majors.size()});
        } else {
          resample_active_majors       = std::move(local_majors);
          resample_active_major_labels = std::move(local_labels);
          vertex_frontier.bucket(0)    = cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false>(
            handle,
            raft::device_span<vertex_t const>{resample_active_majors.data(),
                                                 resample_active_majors.size()},
            raft::device_span<tag_t const>{resample_active_major_labels.data(),
                                              resample_active_major_labels.size()});
          handle.sync_stream();
          active_major_labels = raft::device_span<int32_t const>{
            resample_active_major_labels.data(), resample_active_major_labels.size()};
        }

        sampled_majors = detail::keep_marked_entries(
          handle,
          std::move(sampled_majors),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
        sampled_minors = detail::keep_marked_entries(
          handle,
          std::move(sampled_minors),
          raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
          keep_count);
        if (visited_vertex_labels) {
          sampled_labels = detail::keep_marked_entries(
            handle,
            std::move(*sampled_labels),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }

        if constexpr (!std::is_same_v<T, cuda::std::nullopt_t>) {
          sampled_properties = detail::keep_marked_entries(
            handle,
            std::move(std::get<rmm::device_uvector<T>>(sampled_properties)),
            raft::device_span<uint32_t const>{keep_flags.data(), keep_flags.size()},
            keep_count);
        }
      }
    } else {
      sample_and_append = false;
    }

    // Now I need to update the visited vertices and labels
    std::tie(visited_vertices, visited_vertex_labels) = update_dst_visited_vertices_and_labels(
      handle,
      graph_view,
      std::move(visited_vertices),
      std::move(visited_vertex_labels),
      raft::device_span<vertex_t const>{sampled_minors.data(), sampled_minors.size()},
      sampled_labels ? std::make_optional(raft::device_span<int32_t const>{sampled_labels->data(),
                                                                           sampled_labels->size()})
                     : std::nullopt);

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
        result_properties = std::move(sampled_properties);
      } else {
        auto& result_properties_ref  = std::get<rmm::device_uvector<T>>(result_properties);
        auto& sampled_properties_ref = std::get<rmm::device_uvector<T>>(sampled_properties);
        result_properties_ref.resize(result_properties_ref.size() + sampled_properties_ref.size(),
                                     handle.get_stream());
        raft::copy(result_properties_ref.data() + original_result_majors_size,
                   sampled_properties_ref.data(),
                   sampled_properties_ref.size(),
                   handle.get_stream());
      }
    }
    if (visited_vertex_labels) {
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
                         std::move(result_labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_edges(raft::handle_t const& handle,
             raft::random::RngState& rng_state,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
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
  std::vector<arithmetic_device_uvector_t> edge_properties{};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> sample_labels{std::nullopt};

  if (edge_property_views.size() == 0) {
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
  } else if (edge_property_views.size() == 1) {
    arithmetic_device_uvector_t tmp{std::monostate{}};
    std::tie(majors, minors, tmp, sample_labels) =
      cugraph::variant_type_dispatch(edge_property_views[0],
                                     [&handle,
                                      &rng_state,
                                      &graph_view,
                                      &edge_type_view,
                                      &edge_bias_view,
                                      &vertex_frontier,
                                      &Ks,
                                      &active_major_labels,
                                      with_replacement](auto edge_property_view) {
                                       return sample_with_one_property(handle,
                                                                       rng_state,
                                                                       graph_view,
                                                                       edge_property_view,
                                                                       edge_type_view,
                                                                       edge_bias_view,
                                                                       vertex_frontier,
                                                                       Ks,
                                                                       active_major_labels,
                                                                       with_replacement);
                                     });

    edge_properties.push_back(std::move(tmp));
  } else {
    arithmetic_device_uvector_t multi_index{std::monostate{}};

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(majors, minors, multi_index, sample_labels) =
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

    std::tie(majors, minors, edge_properties) =
      gather_sampled_properties(handle,
                                graph_view,
                                std::move(majors),
                                std::move(minors),
                                std::move(multi_index),
                                raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                  edge_property_views.data(), edge_property_views.size()});
  }

  labels = std::move(sample_labels);

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(labels));
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
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
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
  CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                  "Temporal sampling requires at least a time as a property");

  using tag_t = time_stamp_t;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(active_majors.begin(), active_major_times.begin()),
    thrust::make_zip_iterator(active_majors.end(), active_major_times.end()));

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> edge_properties{};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};

  if (edge_property_views.size() == 1) {
    arithmetic_device_uvector_t tmp{std::monostate{}};
    std::tie(majors, minors, tmp, labels) = cugraph::variant_type_dispatch(
      edge_property_views[0],
      [&handle,
       &rng_state,
       &graph_view,
       &edge_time_view,
       &edge_type_view,
       &edge_bias_view,
       &vertex_frontier,
       &Ks,
       &active_major_labels,
       with_replacement,
       temporal_sampling_comparison](auto& edge_property_view) {
        return temporal_sample_with_one_property(handle,
                                                 rng_state,
                                                 graph_view,
                                                 edge_property_view,
                                                 edge_time_view,
                                                 edge_type_view,
                                                 edge_bias_view,
                                                 vertex_frontier,
                                                 Ks,
                                                 with_replacement,
                                                 active_major_labels,
                                                 temporal_sampling_comparison);
      });

    edge_properties.push_back(std::move(tmp));
  } else {
    arithmetic_device_uvector_t multi_index{std::monostate{}};

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(majors, minors, multi_index, labels) =
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

    std::tie(majors, minors, edge_properties) = gather_sampled_properties(handle,
                                                                          graph_view,
                                                                          std::move(majors),
                                                                          std::move(minors),
                                                                          std::move(multi_index),
                                                                          edge_property_views);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
sample_edges_with_visited(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  raft::host_span<size_t const> Ks,
  std::optional<rmm::device_uvector<vertex_t>>& visited_vertices,
  std::optional<rmm::device_uvector<int32_t>>& visited_vertex_labels,
  bool with_replacement)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");
  CUGRAPH_EXPECTS(visited_vertices, "Visited vertices must be specified");
  CUGRAPH_EXPECTS(active_major_labels.has_value() == visited_vertex_labels.has_value(),
                  "Active major labels and visited vertex labels must both be specified or both "
                  "be unspecified");

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> edge_properties{};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (active_major_labels) {
    using tag_t = int32_t;  // label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    if (edge_property_views.size() == 0) {
      std::tie(majors, minors, std::ignore, labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           vertex_frontier,
                                           visited_vertices,
                                           visited_vertex_labels,
                                           Ks,
                                           active_major_labels,
                                           with_replacement);
    } else if (edge_property_views.size() == 1) {
      arithmetic_device_uvector_t tmp{std::monostate{}};
      std::tie(majors, minors, tmp, labels) = cugraph::variant_type_dispatch(
        edge_property_views[0],
        [&handle,
         &rng_state,
         &graph_view,
         &edge_type_view,
         &edge_bias_view,
         &vertex_frontier,
         &visited_vertices,
         &visited_vertex_labels,
         &active_major_labels,
         &Ks,
         with_replacement](auto edge_property_view) {
          return sample_unvisited_with_one_property(handle,
                                                    rng_state,
                                                    graph_view,
                                                    edge_property_view,
                                                    edge_type_view,
                                                    edge_bias_view,
                                                    vertex_frontier,
                                                    visited_vertices,
                                                    visited_vertex_labels,
                                                    Ks,
                                                    active_major_labels,
                                                    with_replacement);
        });
      edge_properties.push_back(std::move(tmp));
    } else {
      arithmetic_device_uvector_t multi_index{std::monostate{}};

      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);

        std::tie(majors, minors, multi_index, labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             multi_index_property.view(),
                                             edge_type_view,
                                             edge_bias_view,
                                             vertex_frontier,
                                             visited_vertices,
                                             visited_vertex_labels,
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      } else {
        std::tie(majors, minors, std::ignore, labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             cugraph::edge_dummy_property_view_t{},
                                             edge_type_view,
                                             edge_bias_view,
                                             vertex_frontier,
                                             visited_vertices,
                                             visited_vertex_labels,
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      }
      std::tie(majors, minors, edge_properties) =
        gather_sampled_properties(handle,
                                  graph_view,
                                  std::move(majors),
                                  std::move(minors),
                                  std::move(multi_index),
                                  raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                    edge_property_views.data(), edge_property_views.size()});
    }

  } else {
    using tag_t = void;  // no label

    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    if (edge_property_views.size() == 0) {
      std::tie(majors, minors, std::ignore, labels) =
        sample_unvisited_with_one_property(handle,
                                           rng_state,
                                           graph_view,
                                           cugraph::edge_dummy_property_view_t{},
                                           edge_type_view,
                                           edge_bias_view,
                                           vertex_frontier,
                                           visited_vertices,
                                           visited_vertex_labels,
                                           Ks,
                                           active_major_labels,
                                           with_replacement);
    } else if (edge_property_views.size() == 1) {
      arithmetic_device_uvector_t tmp{std::monostate{}};
      std::tie(majors, minors, tmp, labels) = cugraph::variant_type_dispatch(
        edge_property_views[0],
        [&handle,
         &rng_state,
         &graph_view,
         &edge_type_view,
         &edge_bias_view,
         &vertex_frontier,
         &visited_vertices,
         &visited_vertex_labels,
         &active_major_labels,
         &Ks,
         with_replacement](auto edge_property_view) {
          return sample_unvisited_with_one_property(handle,
                                                    rng_state,
                                                    graph_view,
                                                    edge_property_view,
                                                    edge_type_view,
                                                    edge_bias_view,
                                                    vertex_frontier,
                                                    visited_vertices,
                                                    visited_vertex_labels,
                                                    Ks,
                                                    active_major_labels,
                                                    with_replacement);
        });
      edge_properties.push_back(std::move(tmp));
    } else {
      arithmetic_device_uvector_t multi_index{std::monostate{}};

      if (graph_view.is_multigraph()) {
        cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                    graph_view);
        std::tie(majors, minors, multi_index, labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             multi_index_property.view(),
                                             edge_type_view,
                                             edge_bias_view,
                                             vertex_frontier,
                                             visited_vertices,
                                             visited_vertex_labels,
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      } else {
        std::tie(majors, minors, std::ignore, labels) =
          sample_unvisited_with_one_property(handle,
                                             rng_state,
                                             graph_view,
                                             cugraph::edge_dummy_property_view_t{},
                                             edge_type_view,
                                             edge_bias_view,
                                             vertex_frontier,
                                             visited_vertices,
                                             visited_vertex_labels,
                                             Ks,
                                             active_major_labels,
                                             with_replacement);
      }

      std::tie(majors, minors, edge_properties) =
        gather_sampled_properties(handle,
                                  graph_view,
                                  std::move(majors),
                                  std::move(minors),
                                  std::move(multi_index),
                                  raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                    edge_property_views.data(), edge_property_views.size()});
    }
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(labels));
}

}  // namespace detail
}  // namespace cugraph
