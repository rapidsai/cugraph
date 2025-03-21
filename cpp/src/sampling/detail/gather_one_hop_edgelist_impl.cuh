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

#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"
#include "structure/detail/structure_utils.cuh"
#include "utilities/tuple_with_optionals_dispatching.hpp"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_tuple_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/optional>
#include <thrust/tuple.h>

#include <optional>

namespace cugraph {
namespace detail {

struct return_edges_with_properties_e_op {
  template <typename key_t, typename vertex_t, typename EdgeProperties>
  auto __host__ __device__ operator()(key_t optionally_tagged_src,
                                      vertex_t dst,
                                      cuda::std::nullopt_t,
                                      cuda::std::nullopt_t,
                                      EdgeProperties edge_properties) const
  {
    static_assert(std::is_same_v<key_t, vertex_t> ||
                  std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>);

    // FIXME: A solution using thrust_tuple_cat would be more flexible here
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      vertex_t src{optionally_tagged_src};

      if constexpr (std::is_same_v<EdgeProperties, cuda::std::nullopt_t>) {
        return thrust::make_tuple(src, dst);
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_tuple(src, dst, edge_properties);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties)));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 4)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties),
                                                           thrust::get<3>(edge_properties)));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 5)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties),
                                                           thrust::get<3>(edge_properties),
                                                           thrust::get<4>(edge_properties)));
      }
    } else {
      vertex_t src{thrust::get<0>(optionally_tagged_src)};
      int32_t label{thrust::get<1>(optionally_tagged_src)};

      src = thrust::get<0>(optionally_tagged_src);
      if constexpr (std::is_same_v<EdgeProperties, cuda::std::nullopt_t>) {
        return thrust::make_tuple(src, dst, label);
      } else if constexpr (std::is_arithmetic<EdgeProperties>::value) {
        return thrust::make_tuple(src, dst, edge_properties, label);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 2)) {
        return thrust::make_tuple(
          src, dst, thrust::get<0>(edge_properties), thrust::get<1>(edge_properties), label);
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 3)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties),
                                                           label));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 4)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties),
                                                           thrust::get<3>(edge_properties),
                                                           label));
      } else if constexpr (cugraph::is_thrust_tuple_of_arithmetic<EdgeProperties>::value &&
                           (thrust::tuple_size<EdgeProperties>::value == 5)) {
        return cuda::std::make_optional(thrust::make_tuple(src,
                                                           dst,
                                                           thrust::get<0>(edge_properties),
                                                           thrust::get<1>(edge_properties),
                                                           thrust::get<2>(edge_properties),
                                                           thrust::get<3>(edge_properties),
                                                           thrust::get<4>(edge_properties),
                                                           label));
      }
    }
  }
};

// FIXME:  Duplicated...
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

template <int I>
struct gather_type_check_t {
  raft::device_span<uint8_t const> gather_flags{};

  template <typename TupleType>
  bool __device__ operator()(TupleType tup)
  {
    auto edge_type = thrust::get<I>(tup);
    return gather_flags[edge_type] == static_cast<uint8_t>(false);
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          typename tag_t,
          typename label_t,
          bool multi_gpu>
struct gather_one_hop_edgelist_functor_t {
  raft::handle_t const& handle;
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view;
  key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list;
  std::optional<raft::host_span<uint8_t const>> gather_flags;
  bool do_expensive_check{false};

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
    auto return_result =
      std::make_tuple(rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      rmm::device_uvector<vertex_t>(0, handle.get_stream()),
                      std::optional<rmm::device_uvector<edge_type_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<weight_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<edge_time_t>>{std::nullopt},
                      std::optional<rmm::device_uvector<label_t>>{std::nullopt});

    if constexpr (std::is_same_v<std::tuple_element<0, TupleType>,
                                 edge_property_view_t<edge_t, edge_type_t const*>>) {
      auto edge_value_view = concatenate_views(edge_properties);

      auto output_buffer =
        cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                         graph_view,
                                                         key_list,
                                                         edge_src_dummy_property_t{}.view(),
                                                         edge_dst_dummy_property_t{}.view(),
                                                         edge_value_view,
                                                         return_edges_with_properties_e_op{},
                                                         do_expensive_check);

      if (gather_flags) {
        // Require the type to be specified first in the edge properties, otherwise
        // We would need complex logic here.  That makes it the third item in the
        // output_buffer... since we force type view to be set if gather_flags is
        constexpr int I = 2;

        rmm::device_uvector<uint8_t> d_gather_flags(gather_flags->size(), handle.get_stream());
        raft::update_device(
          d_gather_flags.data(), gather_flags->data(), gather_flags->size(), handle.get_stream());

        auto tuple_first = get_dataframe_buffer_begin(output_buffer);
        auto new_size    = static_cast<size_t>(thrust::distance(
          tuple_first,
          thrust::remove_if(handle.get_thrust_policy(),
                            tuple_first,
                            tuple_first + size_dataframe_buffer(output_buffer),
                            gather_type_check_t<I>{raft::device_span<uint8_t const>{
                              d_gather_flags.data(), d_gather_flags.size()}})));
        resize_dataframe_buffer(output_buffer, new_size, handle.get_stream());
      }

      move_results<0, 0, true, true, Flags..., std::is_same_v<tag_t, label_t>>(output_buffer,
                                                                               return_result);
    }

    return return_result;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
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
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, edge_time_t const*>> edge_end_time_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  std::optional<raft::host_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  assert(!gather_flags || edge_type_view);

  if (active_major_labels) {
    using tag_t = label_t;

    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    vertex_label_frontier.bucket(0).insert(
      thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
      thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    gather_one_hop_edgelist_functor_t<vertex_t,
                                      edge_t,
                                      weight_t,
                                      edge_type_t,
                                      edge_time_t,
                                      tag_t,
                                      label_t,
                                      multi_gpu>
      gather_functor{
        handle, graph_view, vertex_label_frontier.bucket(0), gather_flags, do_expensive_check};

    auto [majors, minors, edge_types, weights, edge_ids, edge_start_times, edge_end_times, labels] =
      tuple_with_optionals_dispatch(gather_functor,
                                    edge_type_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    edge_start_time_view,
                                    edge_end_time_view);

    return std::make_tuple(std::move(majors),
                           std::move(minors),
                           std::move(weights),
                           std::move(edge_ids),
                           std::move(edge_types),
                           std::move(edge_start_times),
                           std::move(edge_end_times),
                           std::move(labels));
  } else {
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_frontier(handle, 1);
    vertex_frontier.bucket(0).insert(active_majors.begin(), active_majors.end());

    gather_one_hop_edgelist_functor_t<vertex_t,
                                      edge_t,
                                      weight_t,
                                      edge_type_t,
                                      edge_time_t,
                                      tag_t,
                                      label_t,
                                      multi_gpu>
      gather_functor{
        handle, graph_view, vertex_frontier.bucket(0), gather_flags, do_expensive_check};

    auto [majors, minors, edge_types, weights, edge_ids, edge_start_times, edge_end_times, labels] =
      tuple_with_optionals_dispatch(gather_functor,
                                    edge_type_view,
                                    edge_weight_view,
                                    edge_id_view,
                                    edge_start_time_view,
                                    edge_end_time_view);

    return std::make_tuple(std::move(majors),
                           std::move(minors),
                           std::move(weights),
                           std::move(edge_ids),
                           std::move(edge_types),
                           std::move(edge_start_times),
                           std::move(edge_end_times),
                           std::move(labels));
  }
}

}  // namespace detail
}  // namespace cugraph
