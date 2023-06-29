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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#pragma once

#include <structure/detail/structure_utils.cuh>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_utilities.hpp>
#include <utilities/thrust_wrapper.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/device_span.hpp>

#include <thrust/sort.h>

#include <numeric>
#include <variant>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  rmm::device_uvector<vertex_t> d_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_wgt{std::nullopt};

  std::tie(d_src, d_dst, d_wgt, std::ignore) = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view,
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  std::vector<vertex_t> h_src(d_src.size());
  std::vector<vertex_t> h_dst(d_dst.size());
  std::optional<std::vector<weight_t>> h_wgt(std::nullopt);

  raft::update_host(h_src.data(), d_src.data(), d_src.size(), handle.get_stream());
  raft::update_host(h_dst.data(), d_dst.data(), d_dst.size(), handle.get_stream());

  if (d_wgt) {
    h_wgt = std::make_optional<std::vector<weight_t>>(d_wgt->size());
    raft::update_host(h_wgt->data(), d_wgt->data(), d_wgt->size(), handle.get_stream());
  }

  return std::make_tuple(std::move(h_src), std::move(h_dst), std::move(h_wgt));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
graph_to_device_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  rmm::device_uvector<vertex_t> d_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_wgt{std::nullopt};

  std::tie(d_src, d_dst, d_wgt, std::ignore) = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view,
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  return std::make_tuple(std::move(d_src), std::move(d_dst), std::move(d_wgt));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  rmm::device_uvector<vertex_t> d_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_wgt{std::nullopt};

  std::tie(d_src, d_dst, d_wgt, std::ignore) = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view,
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  rmm::device_uvector<edge_t> d_offsets(0, handle.get_stream());

  if (d_wgt) {
    std::tie(d_offsets, d_dst, *d_wgt, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          d_wgt->begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());

    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end(),
                                d_wgt->begin());
  } else {
    std::tie(d_offsets, d_dst, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());
    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end());
  }

  return std::make_tuple(
    to_host(handle, raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size())),
    to_host(handle, raft::device_span<vertex_t const>(d_dst.data(), d_dst.size())),
    d_wgt ? to_host(handle, raft::device_span<weight_t const>(d_wgt->data(), d_wgt->size()))
          : std::optional<std::vector<weight_t>>(std::nullopt));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, false>,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> number_map,
  bool renumber)
{
  rmm::device_uvector<vertex_t> d_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_dst(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> d_wgt{std::nullopt};

  std::tie(d_src, d_dst, d_wgt, std::ignore) = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view,
    std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
    number_map);

  d_src = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
  d_dst = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
  if (d_wgt)
    *d_wgt = cugraph::test::device_gatherv(
      handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});

  rmm::device_uvector<vertex_t> vertices(0, handle.get_stream());
  if (number_map) { vertices = cugraph::test::device_gatherv(handle, *number_map); }

  graph_t<vertex_t, edge_t, store_transposed, false> sg_graph(handle);
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>
    sg_edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> sg_number_map;
  if (handle.get_comms().get_rank() == 0) {
    if (!number_map) {
      vertices.resize(graph_view.number_of_vertices(), handle.get_stream());
      cugraph::detail::sequence_fill(
        handle.get_stream(), vertices.data(), vertices.size(), vertex_t{0});
    }

    std::tie(sg_graph, sg_edge_weights, std::ignore, std::ignore, sg_number_map) =
      cugraph::create_graph_from_edgelist<vertex_t,
                                          edge_t,
                                          weight_t,
                                          edge_t,
                                          int32_t,
                                          store_transposed,
                                          false>(
        handle,
        std::make_optional(std::move(vertices)),
        std::move(d_src),
        std::move(d_dst),
        std::move(d_wgt),
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()},
        renumber);
  } else {
    d_src.resize(0, handle.get_stream());
    d_src.shrink_to_fit(handle.get_stream());
    d_dst.resize(0, handle.get_stream());
    d_dst.shrink_to_fit(handle.get_stream());
    if (d_wgt) {
      (*d_wgt).resize(0, handle.get_stream());
      (*d_wgt).shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sg_graph), std::move(sg_edge_weights), std::move(sg_number_map));
}

template <typename vertex_t, typename value_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>, rmm::device_uvector<value_t>>
mg_vertex_property_values_to_sg_vertex_property_values(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>> mg_renumber_map,
  std::tuple<vertex_t, vertex_t> mg_local_vertex_partition_range,
  std::optional<raft::device_span<vertex_t const>> sg_renumber_map,
  std::optional<raft::device_span<vertex_t const>> mg_vertices,
  raft::device_span<value_t const> mg_values)
{
  rmm::device_uvector<vertex_t> mg_aggregate_vertices(0, handle.get_stream());
  if (mg_renumber_map) {
    if (mg_vertices) {
      rmm::device_uvector<vertex_t> local_vertices((*mg_vertices).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   (*mg_vertices).begin(),
                   (*mg_vertices).end(),
                   local_vertices.begin());
      cugraph::unrenumber_local_int_vertices(handle,
                                             local_vertices.data(),
                                             local_vertices.size(),
                                             (*mg_renumber_map).data(),
                                             std::get<0>(mg_local_vertex_partition_range),
                                             std::get<1>(mg_local_vertex_partition_range));
      mg_aggregate_vertices = cugraph::test::device_gatherv(
        handle, raft::device_span<vertex_t const>(local_vertices.data(), local_vertices.size()));
    } else {
      mg_aggregate_vertices = cugraph::test::device_gatherv(handle, *mg_renumber_map);
    }
  } else {
    if (mg_vertices) {
      mg_aggregate_vertices = cugraph::test::device_gatherv(handle, *mg_vertices);
    } else {
      rmm::device_uvector<vertex_t> local_vertices(
        std::get<1>(mg_local_vertex_partition_range) - std::get<0>(mg_local_vertex_partition_range),
        handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(),
                       local_vertices.begin(),
                       local_vertices.end(),
                       std::get<0>(mg_local_vertex_partition_range));
      mg_aggregate_vertices = cugraph::test::device_gatherv(
        handle, raft::device_span<vertex_t const>(local_vertices.data(), local_vertices.size()));
    }
  }
  auto mg_aggregate_values = cugraph::test::device_gatherv(handle, mg_values);

  if (handle.get_comms().get_rank() == 0) {
    auto sg_vertices = std::move(mg_aggregate_vertices);
    auto sg_values   = std::move(mg_aggregate_values);
    if (sg_renumber_map) {
      cugraph::renumber_ext_vertices<vertex_t, false>(
        handle,
        sg_vertices.data(),
        sg_vertices.size(),
        (*sg_renumber_map).data(),
        vertex_t{0},
        static_cast<vertex_t>((*sg_renumber_map).size()));
    }

    std::tie(sg_vertices, sg_values) = cugraph::test::sort_by_key(handle, sg_vertices, sg_values);

    if (mg_vertices) {
      return std::make_tuple(std::move(sg_vertices), std::move(sg_values));
    } else {
      return std::make_tuple(std::nullopt, std::move(sg_values));
    }
  } else {
    return std::make_tuple(std::nullopt, rmm::device_uvector<value_t>(0, handle.get_stream()));
  }
}

}  // namespace test
}  // namespace cugraph
