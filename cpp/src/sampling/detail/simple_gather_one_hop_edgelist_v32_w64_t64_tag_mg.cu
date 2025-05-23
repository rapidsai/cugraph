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

#include "cugraph/edge_property.hpp"
#include "simple_gather_one_hop_edgelist_impl.cuh"

using vertex_t    = int32_t;
using edge_t      = int32_t;
using weight_t    = double;
using edge_type_t = int32_t;
using edge_time_t = int64_t;
using tag_t       = int32_t;

constexpr bool multi_gpu = true;

// 11111: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_type_t const*,
                                       weight_t const*,
                                       edge_t const*,
                                       edge_time_t const*,
                                       edge_time_t const*>>,
    thrust::tuple<edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 11110: edge_type_t, weight_t, edge_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<
      thrust::tuple<edge_type_t const*, weight_t const*, edge_t const*, edge_time_t const*>>,
    thrust::tuple<edge_type_t, weight_t, edge_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 11101: edge_type_t, weight_t, edge_t, edge_time_t - duplicate of 11110 in this file

// 11011: edge_type_t, weight_t, edge_time_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<
      thrust::tuple<edge_type_t const*, weight_t const*, edge_time_t const*, edge_time_t const*>>,
    thrust::tuple<edge_type_t, weight_t, edge_time_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 10111: edge_type_t, edge_t, edge_time_t, edge_time_t - duplicated in w32 file

// 01111: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<
      thrust::tuple<weight_t const*, edge_t const*, edge_time_t const*, edge_time_t const*>>,
    thrust::tuple<weight_t, edge_t, edge_time_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 11100: edge_type_t, weight_t, edge_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_type_t const*, weight_t const*, edge_t const*>>,
    thrust::tuple<edge_type_t, weight_t, edge_t>> edge_value_view,
  bool do_expensive_check);

// 11010: edge_type_t, weight_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_type_t const*, weight_t const*, edge_time_t const*>>,
    thrust::tuple<edge_type_t, weight_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 11001: edge_type_t, weight_t, edge_time_t - duplicate of 11010 in this file
// 10110: edge_type_t, edge_t, edge_time_t - duplicated in w32 file
// 10101: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t - duplicate of 10110 in this file
// 10011: edge_type_t, edge_time_t, edge_time_t - duplicated in w32 file

// 01110: weight_t, edge_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<weight_t const*, edge_t const*, edge_time_t const*>>,
    thrust::tuple<weight_t, edge_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 01101: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t - duplicate of 01110 in this file

// 01011: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<weight_t const*, edge_time_t const*, edge_time_t const*>>,
    thrust::tuple<weight_t, edge_time_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 00111: edge_t, edge_time_t, edge_time_t - duplicated in w32 file

// 11000: edge_type_t, weight_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<edge_type_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<edge_type_t const*, weight_t const*>>,
    thrust::tuple<edge_type_t, weight_t>> edge_value_view,
  bool do_expensive_check);

// 10100: edge_type_t, edge_t - duplicated in w32 file
// 10010: edge_type_t, edge_time_t - duplicated in w32 file
// 10001: edge_type_t, weight_t, edge_t, edge_time_t, edge_time_t - duplicate of 10010 in this file

// 01100: weight_t, edge_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t,
                                thrust::zip_iterator<thrust::tuple<weight_t const*, edge_t const*>>,
                                thrust::tuple<weight_t, edge_t>> edge_value_view,
  bool do_expensive_check);

// 01010: weight_t, edge_time_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<edge_time_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<
    edge_t,
    thrust::zip_iterator<thrust::tuple<weight_t const*, edge_time_t const*>>,
    thrust::tuple<weight_t, edge_time_t>> edge_value_view,
  bool do_expensive_check);

// 01001: weight_t, edge_time_t - duplicate of 01010 in this file
// 00110: edge_t, edge_time_t - duplicated in w32 file
// 00101: edge_t, edge_time_t  - duplicate of 00110 in this file
// 00011: edge_time_t, edge_time_t - duplicated in w32 file
// 10000: edge_type_t - duplicated in w32 file

// 01000: weight_t

template std::tuple<rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<vertex_t>,
                    rmm::device_uvector<weight_t>,
                    rmm::device_uvector<tag_t>>
cugraph::detail::simple_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
  cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
  cugraph::edge_property_view_t<edge_t, weight_t const*, weight_t> edge_value_view,
  bool do_expensive_check);

// 00100: edge_t - duplicated in w32 file
// 00010: edge_time_t - duplicated in w32 file
// 00001: edge_time_t - duplicate of 00010 in this file

// 00000:

template std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<tag_t>>
  cugraph::detail::simple_gather_one_hop_edgelist(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
    cugraph::key_bucket_t<vertex_t, tag_t, multi_gpu, false> const& key_list,
    cuda::std::optional<raft::device_span<uint8_t const>> gather_flags,
    cugraph::edge_dummy_property_view_t edge_value_view,
    bool do_expensive_check);
