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

#include <type_traits>

namespace cugraph {
namespace detail {

template <typename key_t, typename vertex_t, typename edge_properties_t, typename label_t>
struct return_edges_with_properties_e_op
  : public format_gather_edges_return_t<vertex_t, edge_properties_t, label_t> {
  typename format_gather_edges_return_t<vertex_t, edge_properties_t, label_t>::return_type
    __device__
    operator()(key_t optionally_tagged_src,
               vertex_t dst,
               cuda::std::nullopt_t,
               cuda::std::nullopt_t,
               edge_properties_t edge_properties) const
  {
    static_assert(std::is_same_v<key_t, vertex_t> ||
                  std::is_same_v<key_t, thrust::tuple<vertex_t, int32_t>>);

    if constexpr (std::is_same_v<key_t, vertex_t>) {
      return format_result(optionally_tagged_src, dst, edge_properties, cuda::std::nullopt);
    } else {
      return format_result(thrust::get<0>(optionally_tagged_src),
                           dst,
                           edge_properties,
                           thrust::get<1>(optionally_tagged_src));
    }
  }
};

struct type_filtered_edges_with_properties_pred_op {
  raft::device_span<uint8_t const> gather_flags_{nullptr, size_t{0}};

  template <typename key_t, typename vertex_t, typename edge_properties_t>
  bool __device__ operator()(key_t tagged_src,
                             vertex_t dst,
                             cuda::std::nullopt_t,
                             cuda::std::nullopt_t,
                             edge_properties_t edge_properties) const
  {
    using edge_type_t = int32_t;

    if constexpr (std::is_arithmetic_v<edge_properties_t>) {
      return (gather_flags_[edge_properties] == static_cast<uint8_t>(true));
    } else {
      static_assert(thrust::tuple_size<edge_properties_t>::value > 1);
      static_assert(
        std::is_same_v<edge_type_t, typename thrust::tuple_element<0, edge_properties_t>::type>);

      return (gather_flags_[thrust::get<0>(edge_properties)] == static_cast<uint8_t>(true));
    }
  }
};

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
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4, T5>,
                                                              label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4, T5>,
                                                              label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");

    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t,
                                        vertex_t,
                                        thrust::tuple<T1, T2, T3, T4, T5>,
                                        label_t>{},
      do_expensive_check);
  }
}

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
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4>,
                                                              label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3, T4>,
                                                              label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");
    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t, vertex_t, thrust::tuple<T1, T2, T3, T4>, label_t>{},
      do_expensive_check);
  }
}

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
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3>,
                                                              label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2, T3>,
                                                              label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");
    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t, vertex_t, thrust::tuple<T1, T2, T3>, label_t>{},
      do_expensive_check);
  }
}

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
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2>,
                                                              label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t,
                                                              vertex_t,
                                                              thrust::tuple<T1, T2>,
                                                              label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");
    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t, vertex_t, thrust::tuple<T1, T2>, label_t>{},
      do_expensive_check);
  }
}

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
  bool do_expensive_check)
{
  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  if constexpr (std::is_same_v<T1, tag_t>) {
    return gather_flags ? cugraph::extract_transform_if_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t, vertex_t, T1, label_t>{},
                            type_filtered_edges_with_properties_pred_op{*gather_flags},
                            do_expensive_check)
                        : cugraph::extract_transform_v_frontier_outgoing_e(
                            handle,
                            graph_view,
                            key_list,
                            edge_src_dummy_property_t{}.view(),
                            edge_dst_dummy_property_t{}.view(),
                            edge_value_view,
                            return_edges_with_properties_e_op<key_t, vertex_t, T1, label_t>{},
                            do_expensive_check);
  } else {
    CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");
    return cugraph::extract_transform_v_frontier_outgoing_e(
      handle,
      graph_view,
      key_list,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_value_view,
      return_edges_with_properties_e_op<key_t, vertex_t, T1, label_t>{},
      do_expensive_check);
  }
}

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
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!gather_flags, "gather_flags can only be specified with edge type");

  using key_t =
    std::conditional_t<std::is_same_v<tag_t, void>, vertex_t, thrust::tuple<vertex_t, tag_t>>;
  using label_t = std::conditional_t<std::is_same_v<tag_t, void>, cuda::std::nullopt_t, tag_t>;

  return cugraph::extract_transform_v_frontier_outgoing_e(
    handle,
    graph_view,
    key_list,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_value_view,
    return_edges_with_properties_e_op<key_t, vertex_t, cuda::std::nullopt_t, label_t>{},
    do_expensive_check);
}

}  // namespace detail
}  // namespace cugraph
