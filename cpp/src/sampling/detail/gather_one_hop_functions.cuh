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

#include "prims/kv_store.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          typename T5,
          bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>,
                              rmm::device_uvector<T4>,
                              rmm::device_uvector<T5>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>,
                              rmm::device_uvector<T4>,
                              rmm::device_uvector<T5>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*, T4 const*, T5 const*>>,
    thrust::tuple<T1, T2, T3, T4, T5>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>,
                              rmm::device_uvector<T4>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>,
                              rmm::device_uvector<T4>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*, T4 const*>>,
    thrust::tuple<T1, T2, T3, T4>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename T1,
          typename T2,
          typename T3,
          bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<T3>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*, T3 const*>>,
    thrust::tuple<T1, T2, T3>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename tag_t,
          typename T1,
          typename T2,
          bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<T1>,
                              rmm::device_uvector<T2>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t,
                                thrust::zip_iterator<thrust::tuple<T1 const*, T2 const*>>,
                                thrust::tuple<T1, T2>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t, typename edge_t, typename tag_t, typename T1, bool multi_gpu>
std::conditional_t<
  std::is_same_v<tag_t, void>,
  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<T1>>,
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             rmm::device_uvector<T1>,
             rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t, T1 const*, T1> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t, typename edge_t, typename tag_t, bool multi_gpu>
std::conditional_t<std::is_same_v<tag_t, void>,
                   std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>,
                   std::tuple<rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<vertex_t>,
                              rmm::device_uvector<tag_t>>>
simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_dummy_property_view_t edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          typename T3,
          typename T4,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>,
           rmm::device_uvector<T4>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<
      thrust::tuple<edge_time_t const*, T1 const*, T2 const*, T3 const*, T4 const*>>,
    thrust::tuple<edge_time_t, T1, T2, T3, T4>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          typename T3,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>,
           rmm::device_uvector<T3>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*, T2 const*, T3 const*>>,
    thrust::tuple<edge_time_t, T1, T2, T3>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t,
          typename edge_t,
          typename edge_time_t,
          typename T1,
          typename T2,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>,
           rmm::device_uvector<T2>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*, T2 const*>>,
    thrust::tuple<edge_time_t, T1, T2>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t, typename edge_t, typename edge_time_t, typename T1, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>,
           rmm::device_uvector<T1>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t,
                                thrust::zip_iterator<thrust::tuple<edge_time_t const*, T1 const*>>,
                                thrust::tuple<edge_time_t, T1>> edge_value_view,
  bool do_expensive_check);

template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_time_t>>
temporal_simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, edge_time_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t, edge_time_t const*, edge_time_t> edge_value_view,
  bool do_expensive_check);

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
  bool do_expensive_check);

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
  bool do_expensive_check);

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
  bool do_expensive_check);

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
  bool do_expensive_check);

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
  bool do_expensive_check);

}  // namespace detail
}  // namespace cugraph
