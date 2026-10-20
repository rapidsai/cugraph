/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace {

// FIXME:  This function should support edge id/type
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
                            std::optional<edge_property_t<edge_t, weight_t>>,
                            std::optional<vertex_property_t<vertex_t, vertex_t>>>>
transpose_graph_storage_impl(raft::handle_t const& handle,
                             graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
                             std::optional<edge_property_t<edge_t, weight_t>>&& edge_weights,
                             std::optional<vertex_property_t<vertex_t, vertex_t>>&& renumber_map,
                             bool do_expensive_check)
{
  auto graph_view = graph.view();

  CUGRAPH_EXPECTS(renumber_map.has_value(),
                  "Invalid input arguments: renumber_map.has_value() should be true if multi-GPU.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  auto is_multigraph = graph.is_multigraph();
  auto is_symmetric  = graph.is_symmetric();

  // FIXME: if is_symmetric is true we can do this more efficiently,
  //        since the graph contents should be exactly the same

  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> edgelist_weights{std::nullopt};

  std::vector<arithmetic_device_uvector_t> edge_property_values{};
  std::tie(edgelist_srcs, edgelist_dsts, edge_property_values) = decompress_to_edgelist(
    handle,
    graph_view,
    edge_weights ? std::vector<edge_arithmetic_property_view_t<edge_t>>{edge_weights->view()}
                 : std::vector<edge_arithmetic_property_view_t<edge_t>>{},
    std::make_optional(renumber_map->view()));
  if (edge_weights) {
    edgelist_weights = std::move(std::get<rmm::device_uvector<weight_t>>(edge_property_values[0]));
  }
  graph = graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);

  std::vector<cugraph::arithmetic_device_uvector_t> edgelist_edge_properties{};
  if (edgelist_weights) edgelist_edge_properties.push_back(std::move(*edgelist_weights));

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_edge_properties) =
    shuffle_ext_edges(handle,
                      std::move(edgelist_srcs),
                      std::move(edgelist_dsts),
                      std::move(edgelist_edge_properties),
                      !store_transposed);

  graph_t<vertex_t, edge_t, !store_transposed, multi_gpu> storage_transposed_graph(handle);
  std::vector<edge_arithmetic_property_t<edge_t>> storage_transposed_edge_properties{};
  std::optional<vertex_property_t<vertex_t, vertex_t>> new_renumber_map{std::nullopt};
  std::tie(storage_transposed_graph, storage_transposed_edge_properties, new_renumber_map) =
    create_graph<vertex_t, edge_t, !store_transposed, multi_gpu>(
      handle,
      std::make_optional(renumber_map->release_buffer()),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_edge_properties),
      graph_properties_t{is_symmetric, is_multigraph},
      true);

  return std::make_tuple(
    std::move(storage_transposed_graph),
    edgelist_weights
      ? std::make_optional<edge_property_t<edge_t, weight_t>>(std::move(
          std::get<edge_property_t<edge_t, weight_t>>(storage_transposed_edge_properties[0])))
      : std::nullopt,
    std::move(new_renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<!multi_gpu,
                 std::tuple<graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
                            std::optional<edge_property_t<edge_t, weight_t>>,
                            std::optional<vertex_property_t<vertex_t, vertex_t>>>>
transpose_graph_storage_impl(raft::handle_t const& handle,
                             graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
                             std::optional<edge_property_t<edge_t, weight_t>>&& edge_weights,
                             std::optional<vertex_property_t<vertex_t, vertex_t>>&& renumber_map,
                             bool do_expensive_check)
{
  auto graph_view = graph.view();

  if (do_expensive_check) { /* currently, nothing to do */
  }

  auto number_of_vertices = graph.number_of_vertices();
  auto is_multigraph      = graph.is_multigraph();
  bool renumber           = renumber_map.has_value();
  auto is_symmetric       = graph.is_symmetric();

  // FIXME: if is_symmetric is true we can do this more efficiently,
  //        since the graph contents should be exactly the same

  rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> edgelist_weights{std::nullopt};

  std::vector<arithmetic_device_uvector_t> edge_property_values{};
  std::tie(edgelist_srcs, edgelist_dsts, edge_property_values) = decompress_to_edgelist(
    handle,
    graph_view,
    edge_weights ? std::vector<edge_arithmetic_property_view_t<edge_t>>{edge_weights->view()}
                 : std::vector<edge_arithmetic_property_view_t<edge_t>>{},
    renumber_map ? std::make_optional(renumber_map->view()) : std::nullopt);
  if (edge_weights) {
    edgelist_weights = std::move(std::get<rmm::device_uvector<weight_t>>(edge_property_values[0]));
  }
  graph         = graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);
  auto vertices = renumber ? std::make_optional(renumber_map->release_buffer())
                           : std::make_optional<rmm::device_uvector<vertex_t>>(number_of_vertices,
                                                                               handle.get_stream());
  if (!renumber) {
    cugraph::sequence(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), vertex_t{0});
  }

  graph_t<vertex_t, edge_t, !store_transposed, multi_gpu> storage_transposed_graph(handle);
  std::vector<edge_arithmetic_property_t<edge_t>> storage_transposed_edge_properties{};
  std::vector<arithmetic_device_uvector_t> edgelist_edge_properties{};
  if (edgelist_weights) edgelist_edge_properties.push_back(std::move(*edgelist_weights));
  std::optional<vertex_property_t<vertex_t, vertex_t>> new_renumber_map{std::nullopt};
  std::tie(storage_transposed_graph, storage_transposed_edge_properties, new_renumber_map) =
    create_graph<vertex_t, edge_t, !store_transposed, multi_gpu>(
      handle,
      std::move(vertices),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_edge_properties),
      graph_properties_t{is_symmetric, is_multigraph},
      renumber);

  return std::make_tuple(
    std::move(storage_transposed_graph),
    edgelist_weights
      ? std::make_optional<edge_property_t<edge_t, weight_t>>(std::move(
          std::get<edge_property_t<edge_t, weight_t>>(storage_transposed_edge_properties[0])))
      : std::nullopt,
    std::move(new_renumber_map));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
           std::optional<edge_property_t<edge_t, weight_t>>,
           std::optional<vertex_property_t<vertex_t, vertex_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
                        std::optional<edge_property_t<edge_t, weight_t>>&& edge_weights,
                        std::optional<vertex_property_t<vertex_t, vertex_t>>&& renumber_map,
                        bool do_expensive_check)
{
  return transpose_graph_storage_impl(
    handle, std::move(graph), std::move(edge_weights), std::move(renumber_map), do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>,
           std::optional<edge_property_t<edge_t, weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
transpose_graph_storage(raft::handle_t const& handle,
                        graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
                        std::optional<edge_property_t<edge_t, weight_t>>&& edge_weights,
                        std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
                        bool do_expensive_check)
{
  auto [new_graph, new_edge_weights, new_renumber_map] =
    transpose_graph_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(graph),
      std::move(edge_weights),
      renumber_map ? std::make_optional(vertex_property_t<vertex_t, vertex_t>(
                       std::move(*renumber_map), static_cast<vertex_t>(renumber_map->size())))
                   : std::nullopt,
      do_expensive_check);
  return std::make_tuple(
    std::move(new_graph),
    std::move(new_edge_weights),
    new_renumber_map ? std::make_optional(new_renumber_map->release_buffer()) : std::nullopt);
}

}  // namespace cugraph
