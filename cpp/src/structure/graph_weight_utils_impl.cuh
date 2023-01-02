/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_e.cuh>

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/extrema.h>

#include <type_traits>

namespace cugraph {

namespace {

template <bool major,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<weight_t> compute_weight_sums(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  rmm::device_uvector<weight_t> weight_sums(graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());
  if (major == store_transposed) {
    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_weight_view,
      [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w; },
      weight_t{0.0},
      reduce_op::plus<weight_t>{},
      weight_sums.data());
  } else {
    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_weight_view,
      [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w; },
      weight_t{0.0},
      reduce_op::plus<weight_t>{},
      weight_sums.data());
  }

  return weight_sums;
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<weight_t> compute_in_weight_sums(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  if (store_transposed) {
    return compute_weight_sums<true>(handle, graph_view, edge_weight_view);
  } else {
    return compute_weight_sums<false>(handle, graph_view, edge_weight_view);
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
rmm::device_uvector<weight_t> compute_out_weight_sums(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  if (store_transposed) {
    return compute_weight_sums<false>(handle, graph_view, edge_weight_view);
  } else {
    return compute_weight_sums<true>(handle, graph_view, edge_weight_view);
  }
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
weight_t compute_max_in_weight_sum(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  auto in_weight_sums = compute_in_weight_sums(handle, graph_view, edge_weight_view);
  auto it =
    thrust::max_element(handle.get_thrust_policy(), in_weight_sums.begin(), in_weight_sums.end());
  weight_t ret{0.0};
  if (it != in_weight_sums.end()) { raft::update_host(&ret, it, 1, handle.get_stream()); }
  if constexpr (multi_gpu) {
    ret =
      host_scalar_allreduce(handle.get_comms(), ret, raft::comms::op_t::MAX, handle.get_stream());
  } else {
    handle.sync_stream();
  }
  return ret;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
weight_t compute_max_out_weight_sum(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  auto out_weight_sums = compute_out_weight_sums(handle, graph_view, edge_weight_view);
  auto it =
    thrust::max_element(handle.get_thrust_policy(), out_weight_sums.begin(), out_weight_sums.end());
  weight_t ret{0.0};
  if (it != out_weight_sums.end()) { raft::update_host(&ret, it, 1, handle.get_stream()); }
  if constexpr (multi_gpu) {
    ret =
      host_scalar_allreduce(handle.get_comms(), ret, raft::comms::op_t::MAX, handle.get_stream());
  } else {
    handle.sync_stream();
  }
  return ret;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
weight_t compute_total_edge_weight(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  return transform_reduce_e(
    handle,
    graph_view,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_weight_view,
    [] __device__(auto, auto, auto, auto, weight_t w) { return w; },
    weight_t{0});
}

}  // namespace cugraph
