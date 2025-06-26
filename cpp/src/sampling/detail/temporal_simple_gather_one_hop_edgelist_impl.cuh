/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "format_gather_edges_return.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_time_t, typename edge_properties_t>
struct return_temporal_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, cuda::std::nullopt_t> {
  typename format_gather_edges_return_t<vertex_t, edge_properties_t, cuda::std::nullopt_t>::
    return_type __device__
    operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    return format_result(thrust::get<0>(tagged_src), dst, edge_properties, cuda::std::nullopt);
  }
};

struct simple_time_filtered_edges_with_properties_pred_op {
  cuda::std::optional<raft::device_span<uint8_t const>> optional_gather_flags_{cuda::std::nullopt};

  template <typename vertex_t, typename edge_time_t, typename edge_properties_t>
  bool __device__ operator()(thrust::tuple<vertex_t, edge_time_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    static_assert(std::is_arithmetic<edge_properties_t>::value ||
                  cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value);

    vertex_t src{thrust::get<0>(tagged_src)};
    edge_time_t src_time{thrust::get<1>(tagged_src)};
    edge_time_t edge_time{};

    if constexpr (std::is_arithmetic_v<edge_properties_t>) {
      edge_time = edge_properties;
    } else {
      edge_time = thrust::get<0>(edge_properties);
    }

    if (src_time < edge_time) {
      if (optional_gather_flags_) {
        if constexpr (cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value) {
          if constexpr (thrust::tuple_size<edge_properties_t>::value > 1) {
            if constexpr (std::is_integral_v<thrust::tuple_element<1, edge_properties_t>>) {
              return ((*optional_gather_flags_)[thrust::get<1>(edge_properties)] ==
                      static_cast<uint8_t>(true));
            }
          }
        }
      } else {
        return true;
      }
    }

    return false;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>,
           rmm::device_uvector<T4>,
           rmm::device_uvector<T5>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, T1, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*, T4 const*, T5 const*>>,
    thrust::tuple<T1, T2, T3, T4, T5>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_temporal_edges_with_properties_e_op<vertex_t, T1, thrust::tuple<T1, T2, T3, T4, T5>>{},
    simple_time_filtered_edges_with_properties_pred_op{gather_flags},
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>,
           rmm::device_uvector<T4>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, T1, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*, T4 const*>>,
    thrust::tuple<T1, T2, T3, T4>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_temporal_edges_with_properties_e_op<vertex_t, T1, thrust::tuple<T1, T2, T3, T4>>{},
    simple_time_filtered_edges_with_properties_pred_op{gather_flags},
    do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename T1, typename T2, typename T3, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, T1, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*>>,
    thrust::tuple<T1, T2, T3>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_temporal_edges_with_properties_e_op<vertex_t, T1, thrust::tuple<T1, T2, T3>>{},
    simple_time_filtered_edges_with_properties_pred_op{gather_flags},
    do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename T1, typename T2, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, T1, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t,
                                thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*>>,
                                thrust::tuple<T1, T2>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_temporal_edges_with_properties_e_op<vertex_t, T1, thrust::tuple<T1, T2>>{},
    simple_time_filtered_edges_with_properties_pred_op{gather_flags},
    do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename T1, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<T1>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, T1, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t, T1 const*, T1> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_temporal_edges_with_properties_e_op<vertex_t, T1, thrust::tuple<T1>>{},
    simple_time_filtered_edges_with_properties_pred_op{gather_flags},
    do_expensive_check);
}

}  // namespace detail
}  // namespace cugraph
