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
#include "prims/kv_store.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_time_t, typename edge_properties_t, typename label_t>
struct return_indirect_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, label_t> {
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>>
    kv_store_view_;

  return_indirect_edges_with_properties_e_op(
    kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
      size_t const*,
      thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view)
    : format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>(),
      kv_store_view_(kv_store_view)
  {
  }

  typename format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>::return_type
    __device__
    operator()(thrust::tuple<vertex_t, size_t> tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    auto tuple = kv_store_view_.find(thrust::get<1>(tagged_src));

    label_t src_label{thrust::get<1>(tuple)};

    return format_result(thrust::get<0>(tagged_src), dst, edge_properties, src_label);
  }
};

template <typename vertex_t, typename edge_time_t, typename edge_properties_t, typename label_t>
struct label_time_filtered_edges_with_properties_pred_op {
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>>
    kv_store_view_;

  cuda::std::optional<raft::device_span<uint8_t const>> optional_gather_flags_{std::nullopt};

  bool __device__ operator()(thrust::tuple<vertex_t, size_t> tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    if constexpr (cugraph::is_thrust_tuple_of_arithmetic<edge_properties_t>::value) {
      auto edge_time = thrust::get<0>(edge_properties);

      if constexpr (std::is_integral_v<decltype(edge_time)>) {
        vertex_t src{thrust::get<0>(tagged_src)};
        size_t label_time_id{thrust::get<1>(tagged_src)};

        auto tuple = kv_store_view_.find(label_time_id);

        edge_time_t src_time{thrust::get<0>(tuple)};
        label_t src_label{thrust::get<1>(tuple)};

        if (src_time < edge_time) {
          if (optional_gather_flags_) {
            auto edge_type = thrust::get<1>(edge_properties);
            if constexpr (std::is_integral_v<decltype(edge_type)>) {
              return ((*optional_gather_flags_)[thrust::get<1>(edge_properties)] ==
                      static_cast<uint8_t>(true));
            }
          } else {
            return true;
          }
        }
      }
    }
    return false;
  }
};

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>,
           rmm::device_uvector<T4>,
           rmm::device_uvector<label_t>>
temporal_label_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<
      thrust::tuple<edge_time_t const*, T1 const*, T2 const*, T3 const*, T4 const*>>,
    thrust::tuple<edge_time_t, T1, T2, T3, T4>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_indirect_edges_with_properties_e_op<vertex_t,
                                               edge_time_t,
                                               thrust::tuple<edge_time_t, T1, T2, T3, T4>,
                                               label_t>(kv_store_view),
    label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                      edge_time_t,
                                                      thrust::tuple<edge_time_t, T1, T2, T3, T4>,
                                                      label_t>{kv_store_view, gather_flags},
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          typename T3,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>,
           rmm::device_uvector<label_t>>
temporal_label_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*, T2 const*, T3 const*>>,
    thrust::tuple<edge_time_t, T1, T2, T3>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_indirect_edges_with_properties_e_op<vertex_t,
                                               edge_time_t,
                                               thrust::tuple<edge_time_t, T1, T2, T3>,
                                               label_t>(kv_store_view),
    label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                      edge_time_t,
                                                      thrust::tuple<edge_time_t, T1, T2, T3>,
                                                      label_t>{kv_store_view, gather_flags},
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<label_t>>
temporal_label_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*, T2 const*>>,
    thrust::tuple<edge_time_t, T1, T2>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_indirect_edges_with_properties_e_op<vertex_t,
                                               edge_time_t,
                                               thrust::tuple<edge_time_t, T1, T2>,
                                               label_t>(kv_store_view),
    label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                      edge_time_t,
                                                      thrust::tuple<edge_time_t, T1, T2>,
                                                      label_t>{kv_store_view, gather_flags},
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<label_t>>
temporal_label_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view,
  cugraph::edge_property_view_t<edge_t,
                                thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*>>,
                                thrust::tuple<edge_time_t, T1>> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_indirect_edges_with_properties_e_op<vertex_t,
                                               edge_time_t,
                                               thrust::tuple<edge_time_t, T1>,
                                               label_t>(kv_store_view),
    label_time_filtered_edges_with_properties_pred_op<vertex_t,
                                                      edge_time_t,
                                                      thrust::tuple<edge_time_t, T1>,
                                                      label_t>{kv_store_view, gather_flags},
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename label_t,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<label_t>>
temporal_label_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  key_bucket_t<vertex_t, size_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  kv_binary_search_store_device_view_t<kv_binary_search_store_view_t<
    size_t const*,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, label_t const*>>>> kv_store_view,
  cugraph::edge_property_view_t<edge_t, edge_time_t const*, edge_time_t> edge_value_view,
  bool do_expensive_check)
{
  return cugraph::extract_transform_if_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_indirect_edges_with_properties_e_op<vertex_t, edge_time_t, edge_time_t, label_t>(
      kv_store_view),
    label_time_filtered_edges_with_properties_pred_op<vertex_t, edge_time_t, edge_time_t, label_t>{
      kv_store_view, gather_flags},
    do_expensive_check);
}

}  // namespace detail
}  // namespace cugraph
