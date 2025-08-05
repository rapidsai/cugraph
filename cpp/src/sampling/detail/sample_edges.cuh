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

#pragma once

#include "gather_sampled_properties.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/transform_gather_e.cuh"
#include "prims/transform_reduce_e.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/sort.h>
#include <thrust/tuple.h>

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
  return_type __device__ operator()(key_t optionally_tagged_src,
                                    vertex_t dst,
                                    cuda::std::nullopt_t,
                                    cuda::std::nullopt_t,
                                    edge_properties_t edge_properties) const
  {
    vertex_t src{};

    if constexpr (std::is_same_v<key_t, vertex_t>)
      src = optionally_tagged_src;
    else
      src = thrust::get<0>(optionally_tagged_src);

    if constexpr (std::is_same_v<edge_properties_t, cuda::std::nullopt_t>) {
      return cuda::std::make_tuple(src, dst);
    } else {
      return cuda::std::make_tuple(src, dst, edge_properties);
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
struct temporal_sample_edge_biases_op_t {
  template <typename edge_time_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, edge_time_t> tagged_src,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               bias_t) const
  {
    // Should not happen at runtime
    return bias_t{0};
  }

  template <typename edge_time_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, edge_time_t> tagged_src,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               edge_time_t edge_time) const
  {
    return (thrust::get<1>(tagged_src) < edge_time) ? bias_t{1} : bias_t{0};
  }

  template <typename edge_time_t>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, edge_time_t> tagged_src,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               cuda::std::tuple<bias_t, edge_time_t> bias_and_time) const
  {
    return (cuda::std::get<1>(tagged_src) < cuda::std::get<1>(bias_and_time))
             ? cuda::std::get<0>(bias_and_time)
             : bias_t{0};
  }

  template <typename edge_time_t,
            typename edge_type_t,
            typename std::enable_if_t<std::is_integral_v<edge_type_t>>* = nullptr>
  bias_t __device__ operator()(cuda::std::tuple<vertex_t, edge_time_t> tagged_src,
                               vertex_t,
                               cuda::std::nullopt_t,
                               cuda::std::nullopt_t,
                               cuda::std::tuple<edge_time_t, edge_type_t> time_and_type) const
  {
    return (thrust::get<1>(tagged_src) < thrust::get<0>(time_and_type)) ? bias_t{1} : bias_t{0};
  }

  template <typename edge_time_t, typename edge_type_t>
  bias_t __device__
  operator()(cuda::std::tuple<vertex_t, edge_time_t> tagged_src,
             vertex_t,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             cuda::std::tuple<bias_t, edge_time_t, edge_type_t> bias_time_and_type) const
  {
    return (thrust::get<1>(tagged_src) < thrust::get<1>(bias_time_and_type))
             ? thrust::get<0>(bias_time_and_type)
             : bias_t{0};
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

template <typename vertex_t, typename edge_t, typename property_view_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<size_t>>>
sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false>& vertex_frontier,
  raft::host_span<size_t const> Ks,
  bool with_replacement)
{
  using edge_type_t = int32_t;

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_bias_view) {
    if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
          *edge_bias_view)) {
      using bias_t = float;

      using T = typename decltype(edge_property_view)::value_type;

      if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false);
      }
    } else if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, double const*>>(
                 *edge_bias_view)) {
      using bias_t = double;

      using T = typename decltype(edge_property_view)::value_type;

      if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false);
      }
    }
  } else {
    using T = typename decltype(edge_property_view)::value_type;

    if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
      std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
        (Ks.size() == 1)
          ? cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              rng_state,
              Ks[0],
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
              false)
          : cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view),
              rng_state,
              Ks,
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
              false);
    } else {
      std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
        (Ks.size() == 1)
          ? cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              rng_state,
              Ks[0],
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
              false)
          : cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view),
              rng_state,
              Ks,
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
              false);
    }
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(sampled_properties), std::move(sample_offsets));
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
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_property_views.size() == 0) {
    std::tie(majors, minors, std::ignore, sample_offsets) =
      sample_with_one_property(handle,
                               rng_state,
                               graph_view,
                               cugraph::edge_dummy_property_view_t{},
                               edge_type_view,
                               edge_bias_view,
                               vertex_frontier,
                               Ks,
                               with_replacement);
  } else if (edge_property_views.size() == 1) {
    arithmetic_device_uvector_t tmp{std::monostate{}};
    std::tie(majors, minors, tmp, sample_offsets) =
      cugraph::variant_type_dispatch(edge_property_views[0],
                                     [&handle,
                                      &rng_state,
                                      &graph_view,
                                      &edge_type_view,
                                      &edge_bias_view,
                                      &vertex_frontier,
                                      &Ks,
                                      with_replacement](auto edge_property_view) {
                                       return sample_with_one_property(handle,
                                                                       rng_state,
                                                                       graph_view,
                                                                       edge_property_view,
                                                                       edge_type_view,
                                                                       edge_bias_view,
                                                                       vertex_frontier,
                                                                       Ks,
                                                                       with_replacement);
                                     });

    edge_properties.push_back(std::move(tmp));
  } else {
    arithmetic_device_uvector_t multi_index{std::monostate{}};

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);
      std::tie(majors, minors, multi_index, sample_offsets) =
        sample_with_one_property(handle,
                                 rng_state,
                                 graph_view,
                                 multi_index_property.view(),
                                 edge_type_view,
                                 edge_bias_view,
                                 vertex_frontier,
                                 Ks,
                                 with_replacement);

    } else {
      std::tie(majors, minors, std::ignore, sample_offsets) =
        sample_with_one_property(handle,
                                 rng_state,
                                 graph_view,
                                 cugraph::edge_dummy_property_view_t{},
                                 edge_type_view,
                                 edge_bias_view,
                                 vertex_frontier,
                                 Ks,
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

  if (active_major_labels) {
    labels = rmm::device_uvector<int32_t>((*sample_offsets).back_element(handle.get_stream()),
                                          handle.get_stream());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(active_majors.size()),
                     segmented_fill_t{*active_major_labels,
                                      raft::device_span<size_t const>(sample_offsets->data(),
                                                                      sample_offsets->size()),
                                      raft::device_span<int32_t>(labels->data(), labels->size())});
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename property_view_t,
          typename edge_time_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<size_t>>>
temporal_sample_with_one_property(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  property_view_t edge_property_view,
  edge_property_view_t<edge_t, edge_time_t const*> edge_time_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
  std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
  cugraph::vertex_frontier_t<vertex_t, edge_time_t, multi_gpu, false>& vertex_frontier,
  raft::host_span<size_t const> Ks,
  bool with_replacement)
{
  using edge_type_t = int32_t;

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  arithmetic_device_uvector_t sampled_properties{std::monostate{}};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_bias_view) {
    if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, float const*>>(
          *edge_bias_view)) {
      using bias_t = float;

      using T = typename decltype(edge_property_view)::value_type;

      if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false);
      }
    } else if (std::holds_alternative<cugraph::edge_property_view_t<edge_t, double const*>>(
                 *edge_bias_view)) {
      using bias_t = double;

      using T = typename decltype(edge_property_view)::value_type;

      if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
                false);

      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
          (Ks.size() == 1)
            ? cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                rng_state,
                Ks[0],
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false)
            : cugraph::per_v_random_select_transform_outgoing_e(
                handle,
                graph_view,
                vertex_frontier.bucket(0),
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                view_concat(
                  std::get<cugraph::edge_property_view_t<edge_t, bias_t const*>>(*edge_bias_view),
                  edge_time_view),
                temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_property_view,
                sample_edges_op_t<vertex_t, T>{},
                std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(
                  *edge_type_view),
                rng_state,
                Ks,
                with_replacement,
                std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
                false);
      }
    }
  } else {
    using bias_t = float;

    using T = typename decltype(edge_property_view)::value_type;

    if constexpr (std::is_same_v<T, cuda::std::nullopt_t>) {
      std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
        (Ks.size() == 1)
          ? cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_time_view,
              temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              rng_state,
              Ks[0],
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
              false)
          : cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_time_view,
              temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view),
              rng_state,
              Ks,
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t>>{std::nullopt},
              false);
    } else {
      std::forward_as_tuple(sample_offsets, std::tie(majors, minors, sampled_properties)) =
        (Ks.size() == 1)
          ? cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_time_view,
              temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              rng_state,
              Ks[0],
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
              false)
          : cugraph::per_v_random_select_transform_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_time_view,
              temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              edge_property_view,
              sample_edges_op_t<vertex_t, T>{},
              std::get<cugraph::edge_property_view_t<edge_t, edge_type_t const*>>(*edge_type_view),
              rng_state,
              Ks,
              with_replacement,
              std::optional<cuda::std::tuple<vertex_t, vertex_t, T>>{std::nullopt},
              false);
    }
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(sampled_properties), std::move(sample_offsets));
}

template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_sample_edges(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
                      edge_property_view_t<edge_t, edge_time_t const*> edge_time_view,
                      std::optional<edge_arithmetic_property_view_t<edge_t>> edge_type_view,
                      std::optional<edge_arithmetic_property_view_t<edge_t>> edge_bias_view,
                      raft::device_span<vertex_t const> active_majors,
                      raft::device_span<edge_time_t const> active_major_times,
                      std::optional<raft::device_span<int32_t const>> active_major_labels,
                      raft::host_span<size_t const> Ks,
                      bool with_replacement)
{
  CUGRAPH_EXPECTS(Ks.size() >= 1, "Must specify non-zero value for Ks");
  CUGRAPH_EXPECTS((Ks.size() == 1) || edge_type_view,
                  "If Ks has more than 1 element must specify types");
  CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                  "Temporal sampling requires at least a time as a property");

  using tag_t = edge_time_t;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(active_majors.begin(), active_major_times.begin()),
    thrust::make_zip_iterator(active_majors.end(), active_major_times.end()));

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> edge_properties{};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_property_views.size() == 1) {
    arithmetic_device_uvector_t tmp{std::monostate{}};
    std::tie(majors, minors, tmp, sample_offsets) =
      cugraph::variant_type_dispatch(edge_property_views[0],
                                     [&handle,
                                      &rng_state,
                                      &graph_view,
                                      &edge_time_view,
                                      &edge_type_view,
                                      &edge_bias_view,
                                      &vertex_frontier,
                                      &Ks,
                                      with_replacement](auto& edge_property_view) {
                                       return temporal_sample_with_one_property(handle,
                                                                                rng_state,
                                                                                graph_view,
                                                                                edge_property_view,
                                                                                edge_time_view,
                                                                                edge_type_view,
                                                                                edge_bias_view,
                                                                                vertex_frontier,
                                                                                Ks,
                                                                                with_replacement);
                                     });

    edge_properties.push_back(std::move(tmp));
  } else {
    arithmetic_device_uvector_t multi_index{std::monostate{}};

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(majors, minors, multi_index, sample_offsets) =
        temporal_sample_with_one_property(handle,
                                          rng_state,
                                          graph_view,
                                          multi_index_property.view(),
                                          edge_time_view,
                                          edge_type_view,
                                          edge_bias_view,
                                          vertex_frontier,
                                          Ks,
                                          with_replacement);

    } else {
      std::tie(majors, minors, std::ignore, sample_offsets) =
        temporal_sample_with_one_property(handle,
                                          rng_state,
                                          graph_view,
                                          cugraph::edge_dummy_property_view_t{},
                                          edge_time_view,
                                          edge_type_view,
                                          edge_bias_view,
                                          vertex_frontier,
                                          Ks,
                                          with_replacement);
    }

    std::tie(majors, minors, edge_properties) = gather_sampled_properties(handle,
                                                                          graph_view,
                                                                          std::move(majors),
                                                                          std::move(minors),
                                                                          std::move(multi_index),
                                                                          edge_property_views);
  }

  if (active_major_labels) {
    labels = rmm::device_uvector<int32_t>((*sample_offsets).back_element(handle.get_stream()),
                                          handle.get_stream());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(active_majors.size()),
                     segmented_fill_t{*active_major_labels,
                                      raft::device_span<size_t const>(sample_offsets->data(),
                                                                      sample_offsets->size()),
                                      raft::device_span<int32_t>(labels->data(), labels->size())});
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_properties), std::move(labels));
}

}  // namespace detail
}  // namespace cugraph
