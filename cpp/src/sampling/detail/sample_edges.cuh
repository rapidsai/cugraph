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

#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template <typename vertex_t>
struct sample_edges_op_t {
  template <typename edge_property_t>
  auto __host__ __device__ operator()(vertex_t src,
                                      vertex_t dst,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      edge_property_t edge_properties) const
  {
    std::conditional_t<std::is_same_v<edge_property_t, cuda::std::nullopt_t>,
                       thrust::tuple<>,
                       std::conditional_t<std::is_arithmetic_v<edge_property_t>,
                                          thrust::tuple<edge_property_t>,
                                          edge_property_t>>
      edge_property_tup{};
    if constexpr (!std::is_same_v<edge_property_t, cuda::std::nullopt_t>) {
      if constexpr (std::is_arithmetic_v<edge_property_t>) {
        thrust::get<0>(edge_property_tup) = edge_properties;
      } else {
        edge_property_tup = edge_properties;
      }
    }
    return thrust_tuple_cat(thrust::make_tuple(src, dst), edge_property_tup);
  }
};

template <typename vertex_t, typename bias_t>
struct sample_edge_biases_op_t {
  auto __host__ __device__
  operator()(vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, bias_t bias) const
  {
    return bias;
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

template <bool has_weight,
          bool has_edge_id,
          bool has_edge_type,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
          bool multi_gpu>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
sample_edges_with_edge_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, void, multi_gpu, false> const& key_list,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::random::RngState& rng_state,
  raft::device_span<vertex_t const> active_majors,
  raft::host_span<size_t const> Ks,
  bool with_replacement)
{
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};

  assert(Ks.size() >= 1);
  assert((Ks.size() == 1) || edge_type_view);
  using edge_value_t = std::conditional_t<
    has_weight,
    std::conditional_t<
      has_edge_id,
      std::conditional_t<has_edge_type,
                         thrust::tuple<weight_t, edge_t, edge_type_t>,
                         thrust::tuple<weight_t, edge_t>>,
      std::conditional_t<has_edge_type, thrust::tuple<weight_t, edge_type_t>, weight_t>>,
    std::conditional_t<
      has_edge_id,
      std::conditional_t<has_edge_type, thrust::tuple<edge_t, edge_type_t>, edge_t>,
      std::conditional_t<has_edge_type, edge_type_t, cuda::std::nullopt_t>>>;
  using sample_e_op_result_t = std::conditional_t<
    has_weight,
    std::conditional_t<
      has_edge_id,
      std::conditional_t<has_edge_type,
                         thrust::tuple<vertex_t, vertex_t, weight_t, edge_t, edge_type_t>,
                         thrust::tuple<vertex_t, vertex_t, weight_t, edge_t>>,
      std::conditional_t<has_edge_type,
                         thrust::tuple<vertex_t, vertex_t, weight_t, edge_type_t>,
                         thrust::tuple<vertex_t, vertex_t, weight_t>>>,
    std::conditional_t<has_edge_id,
                       std::conditional_t<has_edge_type,
                                          thrust::tuple<vertex_t, vertex_t, edge_t, edge_type_t>,
                                          thrust::tuple<vertex_t, vertex_t, edge_t>>,
                       std::conditional_t<has_edge_type,
                                          thrust::tuple<vertex_t, vertex_t, edge_type_t>,
                                          thrust::tuple<vertex_t, vertex_t>>>>;

  using edge_value_view_t = edge_property_view_type_t<edge_t, edge_value_t>;

  edge_value_view_t edge_value_view{};
  if constexpr (has_weight) {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_weight_view, *edge_id_view, *edge_type_view);
      } else {
        edge_value_view = view_concat(*edge_weight_view, *edge_id_view);
      }
    } else {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_weight_view, *edge_type_view);
      } else {
        edge_value_view = *edge_weight_view;
      }
    }
  } else {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_value_view = view_concat(*edge_id_view, *edge_type_view);
      } else {
        edge_value_view = *edge_id_view;
      }
    } else {
      if constexpr (has_edge_type) { edge_value_view = *edge_type_view; }
    }
  }

  std::optional<sample_e_op_result_t> invalid_value{std::nullopt};

  auto [offsets, output_buffer] =
    edge_bias_view
      ? (Ks.size() == 1 ? cugraph::per_v_random_select_transform_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            *edge_bias_view,
                            sample_edge_biases_op_t<vertex_t, bias_t>{},
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            sample_edges_op_t<vertex_t>{},
                            rng_state,
                            Ks[0],
                            with_replacement,
                            invalid_value)
                        : cugraph::per_v_random_select_transform_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            *edge_bias_view,
                            sample_edge_biases_op_t<vertex_t, bias_t>{},
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            sample_edges_op_t<vertex_t>{},
                            *edge_type_view,
                            rng_state,
                            Ks,
                            with_replacement,
                            invalid_value))
      : (Ks.size() == 1
           ? cugraph::per_v_random_select_transform_outgoing_e(handle,
                                                               graph_view,
                                                               key_list,
                                                               edge_src_dummy_property_t{}.view(),
                                                               edge_dst_dummy_property_t{}.view(),
                                                               edge_value_view,
                                                               sample_edges_op_t<vertex_t>{},
                                                               rng_state,
                                                               Ks[0],
                                                               with_replacement,
                                                               invalid_value)
           : cugraph::per_v_random_select_transform_outgoing_e(handle,
                                                               graph_view,
                                                               key_list,
                                                               edge_src_dummy_property_t{}.view(),
                                                               edge_dst_dummy_property_t{}.view(),
                                                               edge_value_view,
                                                               sample_edges_op_t<vertex_t>{},
                                                               *edge_type_view,
                                                               rng_state,
                                                               Ks,
                                                               with_replacement,
                                                               invalid_value));
  sample_offsets = std::move(offsets);
  majors         = std::move(std::get<0>(output_buffer));
  minors         = std::move(std::get<1>(output_buffer));
  if constexpr (has_weight) {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        weights    = std::move(std::get<2>(output_buffer));
        edge_ids   = std::move(std::get<3>(output_buffer));
        edge_types = std::move(std::get<4>(output_buffer));
      } else {
        weights  = std::move(std::get<2>(output_buffer));
        edge_ids = std::move(std::get<3>(output_buffer));
      }
    } else {
      if constexpr (has_edge_type) {
        weights    = std::move(std::get<2>(output_buffer));
        edge_types = std::move(std::get<3>(output_buffer));
      } else {
        weights = std::move(std::get<2>(output_buffer));
      }
    }
  } else {
    if constexpr (has_edge_id) {
      if constexpr (has_edge_type) {
        edge_ids   = std::move(std::get<2>(output_buffer));
        edge_types = std::move(std::get<3>(output_buffer));
      } else {
        edge_ids = std::move(std::get<2>(output_buffer));
      }
    } else {
      if constexpr (has_edge_type) { edge_types = std::move(std::get<2>(output_buffer)); }
    }
  }

  return std::make_tuple(std::move(sample_offsets),
                         std::move(majors),
                         std::move(minors),
                         std::move(weights),
                         std::move(edge_ids),
                         std::move(edge_types));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename bias_t,
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
             std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
             raft::random::RngState& rng_state,
             raft::device_span<vertex_t const> active_majors,
             std::optional<raft::device_span<label_t const>> active_major_labels,
             raft::host_span<size_t const> Ks,
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
  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};

  if (edge_weight_view) {
    bool constexpr has_weight = true;
    if (edge_id_view) {
      bool constexpr has_edge_id = true;
      if (edge_type_view) {
        bool constexpr has_edge_type = true;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      } else {
        bool constexpr has_edge_type = false;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      }
    } else {
      bool constexpr has_edge_id = false;
      if (edge_type_view) {
        bool constexpr has_edge_type = true;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      } else {
        bool constexpr has_edge_type = false;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      }
    }
  } else {
    bool constexpr has_weight = false;
    if (edge_id_view) {
      bool constexpr has_edge_id = true;
      if (edge_type_view) {
        bool constexpr has_edge_type = true;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      } else {
        bool constexpr has_edge_type = false;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      }
    } else {
      bool constexpr has_edge_id = false;
      if (edge_type_view) {
        bool constexpr has_edge_type = true;
        auto edge_value_view         = *edge_type_view;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      } else {
        bool constexpr has_edge_type = false;
        std::tie(sample_offsets, majors, minors, weights, edge_ids, edge_types) =
          sample_edges_with_edge_values<has_weight, has_edge_id, has_edge_type>(
            handle,
            graph_view,
            vertex_frontier.bucket(0),
            edge_weight_view,
            edge_id_view,
            edge_type_view,
            edge_bias_view,
            rng_state,
            active_majors,
            Ks,
            with_replacement);
      }
    }
  }

  std::optional<rmm::device_uvector<int32_t>> labels{std::nullopt};
  if (active_major_labels) {
    labels = rmm::device_uvector<int32_t>(sample_offsets->back_element(handle.get_stream()),
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
