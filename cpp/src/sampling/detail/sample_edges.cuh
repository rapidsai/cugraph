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

#include <prims/per_v_random_select_transform_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>  // ??
#include <prims/vertex_frontier.cuh>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct sample_edges_op_t {
  template <typename EdgeProperties>
  auto __host__ __device__ operator()(vertex_t src,
                                      vertex_t dst,
                                      thrust::nullopt_t,
                                      thrust::nullopt_t,
                                      EdgeProperties edge_properties) const
  {
    // FIXME: A solution using thrust_tuple_cat would be more flexible here
    if constexpr (std::is_same_v<EdgeProperties, thrust::nullopt_t>) {
      return thrust::make_tuple(src, dst);
    } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
      return thrust::make_tuple(src, dst, edge_properties);
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 2)) {
      return thrust::make_tuple(
        src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties));
    } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                         (thrust::tuple_size<EdgeProperties>::value == 3)) {
      return thrust::make_tuple(src,
                                dst,
                                thrust::get<0>(edge_properties),
                                thrust::get<1>(edge_properties),
                                thrust::get<2>(edge_properties));
    }
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
          typename weight_t,
          typename edge_type_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
             std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
             raft::random::RngState& rng_state,
             raft::device_span<vertex_t const> active_majors,
             std::optional<raft::device_span<label_t const>> active_major_labels,
             size_t fanout,
             bool with_replacement)
{
  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_weight_view) {
    if (edge_id_view) {
      if (edge_type_view) {
        std::forward_as_tuple(sample_offsets,
                              std::tie(majors, minors, weights, edge_ids, edge_types)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            view_concat(*edge_weight_view, *edge_id_view, *edge_type_view),
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, weight_t, edge_t, edge_type_t>>{
              std::nullopt},
            true);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, weights, edge_ids)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            view_concat(*edge_weight_view, *edge_id_view),
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, weight_t, edge_t>>{std::nullopt},
            true);
      }
    } else {
      if (edge_type_view) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, weights, edge_types)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            view_concat(*edge_weight_view, *edge_type_view),
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, weight_t, edge_type_t>>{std::nullopt},
            true);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, weights)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            *edge_weight_view,
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, weight_t>>{std::nullopt},
            true);
      }
    }
  } else {
    if (edge_id_view) {
      if (edge_type_view) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, edge_ids, edge_types)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            view_concat(*edge_id_view, *edge_type_view),
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, edge_t, edge_type_t>>{std::nullopt},
            true);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, edge_ids)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            *edge_id_view,
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, edge_t>>{std::nullopt},
            true);
      }
    } else {
      if (edge_type_view) {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors, edge_types)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            *edge_type_view,
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t, edge_type_t>>{std::nullopt},
            true);
      } else {
        std::forward_as_tuple(sample_offsets, std::tie(majors, minors)) =
          cugraph::per_v_random_select_transform_outgoing_e(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_src_dummy_property_t{}.view(),
            edge_dst_dummy_property_t{}.view(),
            edge_dummy_property_t{}.view(),
            sample_edges_op_t<vertex_t>{},
            rng_state,
            fanout,
            with_replacement,
            std::optional<thrust::tuple<vertex_t, vertex_t>>{std::nullopt},
            true);
      }
    }
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

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(labels));
}

}  // namespace detail
}  // namespace cugraph
