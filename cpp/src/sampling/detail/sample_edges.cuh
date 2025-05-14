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
#include "prims/transform_reduce_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"
#include "utilities/tuple_with_optionals_dispatching.hpp"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
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

template <typename vertex_t>
struct sample_edges_op_t {
  template <typename key_t, typename edge_property_t>
  auto __device__ operator()(key_t optionally_tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_property_t edge_properties) const
  {
    vertex_t src{};

    if constexpr (std::is_same_v<key_t, vertex_t>)
      src = optionally_tagged_src;
    else
      src = thrust::get<0>(optionally_tagged_src);

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

template <typename vertex_t, typename bias_t>
struct temporal_sample_edge_biases_op_t {
#if 1
  // THIS FUNCTION SHOULD NOT BE NEEDED... ADDING IT BECAUSE LEAVING IT OUT MASKS THE MORE
  // CHALLENGING ERROR
  template <typename edge_time_t>
  auto __host__ __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                                      vertex_t,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      bias_t bias) const
  {
    return bias;
  }
#endif

  template <typename edge_time_t>
  auto __host__ __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                                      vertex_t,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      edge_time_t edge_time) const
  {
    return (thrust::get<1>(tagged_src) < edge_time) ? bias_t{1} : bias_t{0};
  }

  template <typename edge_time_t>
  auto __host__ __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                                      vertex_t,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      thrust::tuple<bias_t, edge_time_t> bias_and_time) const
  {
    return (thrust::get<1>(tagged_src) < thrust::get<1>(bias_and_time))
             ? thrust::get<0>(bias_and_time)
             : bias_t{0};
  }

  template <typename edge_time_t,
            typename edge_type_t,
            typename std::enable_if_t<std::is_integral_v<edge_type_t>>* = nullptr>
  auto __host__ __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                                      vertex_t,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      thrust::tuple<edge_time_t, edge_type_t> time_and_type) const
  {
    return (thrust::get<1>(tagged_src) < thrust::get<0>(time_and_type)) ? bias_t{1} : bias_t{0};
  }

  template <typename edge_time_t, typename edge_type_t>
  auto __host__ __device__
  operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
             vertex_t,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             thrust::tuple<bias_t, edge_time_t, edge_type_t> bias_time_and_type) const
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

template <size_t input_tuple_pos,
          size_t output_tuple_pos,
          bool Flag,
          bool... Flags,
          typename InputTupleType,
          typename OutputTupleType>
void move_results(InputTupleType& input_tuple, OutputTupleType& output_tuple)
{
  if constexpr (Flag) {
    std::get<output_tuple_pos>(output_tuple) = std::move(std::get<input_tuple_pos>(input_tuple));
  }

  if constexpr (sizeof...(Flags) > 0) {
    if constexpr (Flag) {
      move_results<input_tuple_pos + 1, output_tuple_pos + 1, Flags...>(input_tuple, output_tuple);
    } else {
      move_results<input_tuple_pos, output_tuple_pos + 1, Flags...>(input_tuple, output_tuple);
    }
  }
}

template <typename EdgePropertyView>
typename EdgePropertyView::value_type default_value(EdgePropertyView const&)
{
  return 0;
}

template <typename EdgeProperties, std::size_t... I>
auto construct_invalid_value(EdgeProperties const& properties,
                             std::integer_sequence<std::size_t, I...>)
{
  return thrust::make_tuple(default_value(std::get<I>(properties))...);
}

template <typename vertex_t, typename... Ts>
auto construct_invalid_value(std::tuple<Ts...> const& properties)
{
  auto tmp_result = thrust_tuple_cat(
    thrust::make_tuple(vertex_t{}, vertex_t{}),
    construct_invalid_value(properties, std::make_index_sequence<sizeof...(Ts)>{}));
  return std::optional<decltype(tmp_result)>{std::nullopt};
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename tag_t,
          typename bias_view_t,
          typename bias_functor_t,
          bool multi_gpu>
struct sample_edges_functor_t {
  raft::handle_t const& handle;
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view;
  key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list;
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view;
  std::optional<bias_view_t> edge_bias_view;
  bias_functor_t edge_bias_functor;
  raft::random::RngState& rng_state;
  raft::host_span<size_t const> Ks;
  bool with_replacement;

  template <typename TupleType>
  auto concatenate_views(TupleType edge_properties)
  {
    if constexpr (std::tuple_size_v<TupleType> == 0) {
      return edge_property_view_type_t<graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                       cuda::std::nullopt_t>{};
    } else if constexpr (std::tuple_size_v<TupleType> == 1) {
      return std::get<0>(edge_properties);
    } else {
      return view_concat(edge_properties);
    }
  }

  template <bool... Flags, typename TupleType>
  auto operator()(TupleType edge_properties)
  {
    auto edge_value_view = concatenate_views(edge_properties);
    auto invalid_value   = construct_invalid_value<vertex_t>(edge_properties);

    auto [offsets, output_buffer] =
      edge_bias_view
        ? (Ks.size() == 1
             ? cugraph::per_v_random_select_transform_outgoing_e(handle,
                                                                 graph_view,
                                                                 key_list,
                                                                 edge_src_dummy_property_t{}.view(),
                                                                 edge_dst_dummy_property_t{}.view(),
                                                                 *edge_bias_view,
                                                                 edge_bias_functor,
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
                                                                 *edge_bias_view,
                                                                 edge_bias_functor,
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

    auto return_result =
      std::make_tuple(std::move(offsets),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      std::optional<rmm::device_uvector<weight_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_type_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt});

    move_results<0, 1, true, true, Flags...>(output_buffer, return_result);

    return return_result;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename bias_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<label_t>>>
sample_edges(raft::handle_t const& handle,
             graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
             std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
             std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
             std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
             std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_start_time_view,
             std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
             std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
             raft::random::RngState& rng_state,
             raft::device_span<vertex_t const> active_majors,
             std::optional<raft::device_span<label_t const>> active_major_labels,
             raft::host_span<size_t const> Ks,
             bool with_replacement)
{
  assert(Ks.size() >= 1);
  assert((Ks.size() == 1) || edge_type_view);

  using tag_t = void;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

  sample_edges_functor_t<vertex_t,
                         edge_t,
                         weight_t,
                         edge_type_t,
                         edge_time_t,
                         tag_t,
                         edge_property_view_t<edge_t, bias_t const*, bias_t>,
                         decltype(sample_edge_biases_op_t<vertex_t, bias_t>{}),
                         multi_gpu>
    sample_functor{handle,
                   graph_view,
                   vertex_frontier.bucket(0),
                   edge_type_view,
                   edge_bias_view,
                   sample_edge_biases_op_t<vertex_t, bias_t>{},
                   rng_state,
                   Ks,
                   with_replacement};

  auto [sample_offsets,
        majors,
        minors,
        weights,
        edge_ids,
        edge_types,
        edge_start_times,
        edge_end_times] = tuple_with_optionals_dispatch(sample_functor,
                                                        edge_weight_view,
                                                        edge_id_view,
                                                        edge_type_view,
                                                        edge_start_time_view,
                                                        edge_end_time_view);

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
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(labels));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename bias_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<edge_time_t>>,
           std::optional<rmm::device_uvector<label_t>>>
temporal_sample_edges(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::random::RngState& rng_state,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<edge_time_t const> active_major_times,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  raft::host_span<size_t const> Ks,
  bool with_replacement)
{
  assert(Ks.size() >= 1);
  assert((Ks.size() == 1) || edge_type_view);
  assert(edge_start_time_view);

  using tag_t = edge_time_t;

  cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_frontier(handle, 1);

  vertex_frontier.bucket(0).insert(
    thrust::make_zip_iterator(active_majors.begin(), active_major_times.begin()),
    thrust::make_zip_iterator(active_majors.end(), active_major_times.end()));

  std::optional<rmm::device_uvector<size_t>> sample_offsets{std::nullopt};
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  std::optional<rmm::device_uvector<edge_t>> edge_ids{std::nullopt};
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> edge_start_times{std::nullopt};
  std::optional<rmm::device_uvector<edge_time_t>> edge_end_times{std::nullopt};

  if (edge_bias_view) {
    sample_edges_functor_t<vertex_t,
                           edge_t,
                           weight_t,
                           edge_type_t,
                           edge_time_t,
                           tag_t,
                           decltype(view_concat(*edge_bias_view, *edge_start_time_view)),
                           decltype(temporal_sample_edge_biases_op_t<vertex_t, bias_t>{}),
                           multi_gpu>
      sample_functor{handle,
                     graph_view,
                     vertex_frontier.bucket(0),
                     edge_type_view,
                     std::make_optional(view_concat(*edge_bias_view, *edge_start_time_view)),
                     temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                     rng_state,
                     Ks,
                     with_replacement};

    std::tie(sample_offsets,
             majors,
             minors,
             weights,
             edge_ids,
             edge_types,
             edge_start_times,
             edge_end_times) = tuple_with_optionals_dispatch(sample_functor,
                                                             edge_weight_view,
                                                             edge_id_view,
                                                             edge_type_view,
                                                             edge_start_time_view,
                                                             edge_end_time_view);
  } else {
    sample_edges_functor_t<vertex_t,
                           edge_t,
                           weight_t,
                           edge_type_t,
                           edge_time_t,
                           tag_t,
                           edge_property_view_t<edge_t, edge_time_t const*, edge_time_t>,
                           decltype(temporal_sample_edge_biases_op_t<vertex_t, bias_t>{}),
                           multi_gpu>
      sample_functor{handle,
                     graph_view,
                     vertex_frontier.bucket(0),
                     edge_type_view,
                     edge_start_time_view,
                     temporal_sample_edge_biases_op_t<vertex_t, bias_t>{},
                     rng_state,
                     Ks,
                     with_replacement};

    std::tie(sample_offsets,
             majors,
             minors,
             weights,
             edge_ids,
             edge_types,
             edge_start_times,
             edge_end_times) = tuple_with_optionals_dispatch(sample_functor,
                                                             edge_weight_view,
                                                             edge_id_view,
                                                             edge_type_view,
                                                             edge_start_time_view,
                                                             edge_end_time_view);
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
                         std::move(edge_start_times),
                         std::move(edge_end_times),
                         std::move(labels));
}

}  // namespace detail
}  // namespace cugraph
