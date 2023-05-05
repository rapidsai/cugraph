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

#include <prims/extract_transform_v_frontier_outgoing_e.cuh>
#include <prims/per_v_random_select_transform_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>  // ??
#include <prims/vertex_frontier.cuh>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <thrust/optional.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

struct return_edges_with_properties_e_op {
  template <typename key_t, typename vertex_t, typename EdgeProperties>
  auto __host__ __device__ operator()(key_t optionally_tagged_src,
                                      vertex_t dst,
                                      thrust::nullopt_t,
                                      thrust::nullopt_t,
                                      EdgeProperties edge_properties)
  {
    // FIXME: A solution using thrust_tuple_cat would be more flexible here
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      vertex_t src{optionally_tagged_src};

      if constexpr (std::is_same_v<EdgeProperties, thrust::nullopt_t>) {
        return thrust::make_optional(thrust::make_tuple(src, dst));
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_optional(thrust::make_tuple(src, dst, edge_properties));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_optional(thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties)));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return thrust::make_optional(thrust::make_tuple(src,
                                                        dst,
                                                        thrust::get<0>(edge_properties),
                                                        thrust::get<1>(edge_properties),
                                                        thrust::get<2>(edge_properties)));
      }
    } else if constexpr (std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>) {
      vertex_t src{thrust::get<0>(optionally_tagged_src)};
      int32_t label{thrust::get<1>(optionally_tagged_src)};

      src = thrust::get<0>(optionally_tagged_src);
      if constexpr (std::is_same_v<EdgeProperties, thrust::nullopt_t>) {
        return thrust::make_optional(thrust::make_tuple(src, dst, label));
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_optional(thrust::make_tuple(src, dst, edge_properties, label));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_optional(thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties), label));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return thrust::make_optional(thrust::make_tuple(src,
                                                        dst,
                                                        thrust::get<0>(edge_properties),
                                                        thrust::get<1>(edge_properties),
                                                        thrust::get<2>(edge_properties),
                                                        label));
      }
    }
  }
};

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
          typename tag_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<label_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> const& vertex_frontier,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

  if (edge_weight_view) {
    if (edge_id_view) {
      if (edge_type_view) {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, weights, edge_ids, edge_types, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_id_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        } else {
          std::tie(majors, minors, weights, edge_ids, edge_types) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_id_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        }
      } else {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, weights, edge_ids, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_id_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        } else {
          std::tie(majors, minors, weights, edge_ids) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_id_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        }
      }
    } else {
      if (edge_type_view) {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, weights, edge_types, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        } else {
          std::tie(majors, minors, weights, edge_types) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_weight_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        }
      } else {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, weights, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_weight_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        } else {
          std::tie(majors, minors, weights) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_weight_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        }
      }
    }
  } else {
    if (edge_id_view) {
      if (edge_type_view) {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, edge_ids, edge_types, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_id_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        } else {
          std::tie(majors, minors, edge_ids, edge_types) =
            cugraph::extract_transform_v_frontier_outgoing_e(
              handle,
              graph_view,
              vertex_frontier.bucket(0),
              edge_src_dummy_property_t{}.view(),
              edge_dst_dummy_property_t{}.view(),
              view_concat(*edge_id_view, *edge_type_view),
              return_edges_with_properties_e_op{},
              do_expensive_check);
        }
      } else {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, edge_ids, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_id_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        } else {
          std::tie(majors, minors, edge_ids) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_id_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        }
      }
    } else {
      if (edge_type_view) {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, edge_types, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_type_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        } else {
          std::tie(majors, minors, edge_types) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             *edge_type_view,
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        }
      } else {
        if constexpr (std::is_same_v<tag_t, int32_t>) {
          std::tie(majors, minors, labels) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             edge_dummy_property_t{}.view(),
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        } else {
          std::tie(majors, minors) =
            cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                             graph_view,
                                                             vertex_frontier.bucket(0),
                                                             edge_src_dummy_property_t{}.view(),
                                                             edge_dst_dummy_property_t{}.view(),
                                                             edge_dummy_property_t{}.view(),
                                                             return_edges_with_properties_e_op{},
                                                             do_expensive_check);
        }
      }
    }
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(labels));
}

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
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  bool do_expensive_check)
{
  if (active_major_labels) {
    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    vertex_label_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    return gather_one_hop_edgelist<vertex_t,
                                   edge_t,
                                   weight_t,
                                   edge_type_t,
                                   label_t,
                                   label_t,
                                   multi_gpu>(handle,
                                              graph_view,
                                              edge_weight_view,
                                              edge_id_view,
                                              edge_type_view,
                                              vertex_label_frontier,
                                              do_expensive_check);
  } else {
    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    return gather_one_hop_edgelist<vertex_t,
                                   edge_t,
                                   weight_t,
                                   edge_type_t,
                                   label_t,
                                   void,
                                   multi_gpu>(handle,
                                              graph_view,
                                              edge_weight_view,
                                              edge_id_view,
                                              edge_type_view,
                                              vertex_frontier,
                                              do_expensive_check);
  }
}

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

template <typename vertex_t,
          typename weight_t,
          typename edge_t,
          typename edge_type_t,
          typename label_t>
void sort_sampled_tuples(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& majors,
                         rmm::device_uvector<vertex_t>& minors,
                         std::optional<rmm::device_uvector<weight_t>>& weights,
                         std::optional<rmm::device_uvector<edge_t>>& edge_ids,
                         std::optional<rmm::device_uvector<edge_type_t>>& edge_types,
                         std::optional<rmm::device_uvector<int32_t>>& hops,
                         std::optional<rmm::device_uvector<label_t>>& labels)
{
  if (weights) {
    if (edge_ids) {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        weights->begin(),
                                                        edge_ids->begin(),
                                                        edge_types->begin(),
                                                        hops->begin()));
        } else {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        weights->begin(),
                                                        edge_ids->begin(),
                                                        edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_ids->begin(), hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_ids->begin()));
        }
      }
    } else {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        weights->begin(),
                                                        edge_types->begin(),
                                                        hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), weights->begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(
                                majors.begin(), minors.begin(), weights->begin(), hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), weights->begin()));
        }
      }
    }
  } else {
    if (edge_ids) {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(),
                                                        minors.begin(),
                                                        edge_ids->begin(),
                                                        edge_types->begin(),
                                                        hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), edge_ids->begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(
                                majors.begin(), minors.begin(), edge_ids->begin(), hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_ids->begin()));
        }
      }
    } else {
      if (edge_types) {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(
              majors.begin(), minors.begin(), edge_types->begin(), hops->begin()));
        } else {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), edge_types->begin()));
        }
      } else {
        if (hops) {
          thrust::sort_by_key(
            handle.get_thrust_policy(),
            labels->begin(),
            labels->end(),
            thrust::make_zip_iterator(majors.begin(), minors.begin(), hops->begin()));
        } else {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              labels->begin(),
                              labels->end(),
                              thrust::make_zip_iterator(majors.begin(), minors.begin()));
        }
      }
    }
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<label_t>>,
           std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  std::optional<rmm::device_uvector<weight_t>>&& weights,
  std::optional<rmm::device_uvector<edge_t>>&& edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<label_t>>&& labels,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank)
{
  std::optional<rmm::device_uvector<size_t>> offsets{std::nullopt};

  if (labels) {
    sort_sampled_tuples(handle, majors, minors, weights, edge_ids, edge_types, hops, labels);

    if (label_to_output_comm_rank) {
      CUGRAPH_EXPECTS(labels, "labels must be specified in order to shuffle sampling results");

      auto& comm           = handle.get_comms();
      auto const comm_size = comm.get_size();

      auto total_global_mem = handle.get_device_properties().totalGlobalMem;
      auto element_size     = sizeof(vertex_t) * 2 + (weights ? sizeof(weight_t) : size_t{0}) +
                          (edge_ids ? sizeof(edge_t) : size_t{0}) +
                          (edge_types ? sizeof(edge_type_t) : size_t{0}) +
                          (hops ? sizeof(int32_t) : size_t{0}) + sizeof(label_t);

      auto constexpr mem_frugal_ratio =
        0.1;  // if the expected temporary buffer size exceeds the mem_frugal_ratio of the
      // total_global_mem, switch to the memory frugal approach (thrust::sort is used to
      // group-by by default, and thrust::sort requires temporary buffer comparable to the input
      // data size)
      auto mem_frugal_threshold = static_cast<size_t>(
        static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);

      auto d_tx_value_counts = cugraph::groupby_and_count(
        labels->begin(),
        labels->end(),
        [output_label = std::get<0>(*label_to_output_comm_rank),
         output_rank  = std::get<1>(*label_to_output_comm_rank)] __device__(auto val) {
          auto pos =
            thrust::lower_bound(thrust::seq, output_label.begin(), output_label.end(), val);
          return output_rank[thrust::distance(output_label.begin(), pos)];
        },
        comm_size,
        mem_frugal_threshold,
        handle.get_stream());

      std::vector<size_t> h_tx_value_counts(d_tx_value_counts.size());
      raft::update_host(h_tx_value_counts.data(),
                        d_tx_value_counts.data(),
                        d_tx_value_counts.size(),
                        handle.get_stream());
      handle.sync_stream();

      if (weights) {
        if (edge_ids) {
          if (edge_types) {
            if (hops) {
              std::forward_as_tuple(
                std::tie(majors, minors, weights, edge_ids, edge_types, hops, labels),
                std::ignore) = shuffle_values(comm,
                                              thrust::make_zip_iterator(majors.begin(),
                                                                        minors.begin(),
                                                                        weights->begin(),
                                                                        edge_ids->begin(),
                                                                        edge_types->begin(),
                                                                        hops->begin(),
                                                                        labels->begin()),
                                              h_tx_value_counts,
                                              handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, weights, edge_ids, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_ids->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          }
        } else {
          if (edge_types) {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, weights, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, weights, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, weights, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         weights->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, weights, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(
                                 majors.begin(), minors.begin(), weights->begin(), labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          }
        }
      } else {
        if (edge_ids) {
          if (edge_types) {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, edge_ids, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, edge_ids, edge_types, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         edge_types->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            }
          } else {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, edge_ids, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_ids->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, edge_ids, labels), std::ignore) =
                shuffle_values(
                  comm,
                  thrust::make_zip_iterator(
                    majors.begin(), minors.begin(), edge_ids->begin(), labels->begin()),
                  h_tx_value_counts,
                  handle.get_stream());
            }
          }
        } else {
          if (edge_types) {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, edge_types, hops, labels),
                                    std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(majors.begin(),
                                                         minors.begin(),
                                                         edge_types->begin(),
                                                         hops->begin(),
                                                         labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, edge_types, labels), std::ignore) =
                shuffle_values(
                  comm,
                  thrust::make_zip_iterator(
                    majors.begin(), minors.begin(), edge_types->begin(), labels->begin()),
                  h_tx_value_counts,
                  handle.get_stream());
            }
          } else {
            if (hops) {
              std::forward_as_tuple(std::tie(majors, minors, hops, labels), std::ignore) =
                shuffle_values(comm,
                               thrust::make_zip_iterator(
                                 majors.begin(), minors.begin(), hops->begin(), labels->begin()),
                               h_tx_value_counts,
                               handle.get_stream());
            } else {
              std::forward_as_tuple(std::tie(majors, minors, labels), std::ignore) = shuffle_values(
                comm,
                thrust::make_zip_iterator(majors.begin(), minors.begin(), labels->begin()),
                h_tx_value_counts,
                handle.get_stream());
            }
          }
        }
      }

      sort_sampled_tuples(handle, majors, minors, weights, edge_ids, edge_types, hops, labels);
    }

    size_t num_unique_labels =
      thrust::count_if(handle.get_thrust_policy(),
                       thrust::make_counting_iterator<size_t>(0),
                       thrust::make_counting_iterator<size_t>(labels->size()),
                       is_first_in_run_t<label_t const*>{labels->data()});

    rmm::device_uvector<label_t> unique_labels(num_unique_labels, handle.get_stream());
    offsets = rmm::device_uvector<size_t>(num_unique_labels + 1, handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          labels->begin(),
                          labels->end(),
                          thrust::make_constant_iterator(size_t{1}),
                          unique_labels.begin(),
                          offsets->begin());

    thrust::exclusive_scan(
      handle.get_thrust_policy(), offsets->begin(), offsets->end(), offsets->begin());
    labels = std::move(unique_labels);
  }

  return std::make_tuple(std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types),
                         std::move(hops),
                         std::move(labels),
                         std::move(offsets));
}

}  // namespace detail
}  // namespace cugraph
