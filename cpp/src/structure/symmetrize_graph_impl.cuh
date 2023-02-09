/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/sequence.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<
    graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
symmetrize_graph_impl(
  raft::handle_t const& handle,
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&& edge_weights,
  std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check)
{
  auto graph_view = graph.view();

  CUGRAPH_EXPECTS(renumber_map.has_value(),
                  "Invalid input arguments: renumber_map.has_value() should be true if multi-GPU.");
  CUGRAPH_EXPECTS(
    (*renumber_map).size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid input arguments: (*renumber_map).size() should match with the local "
    "vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  if (graph.is_symmetric()) {
    return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
  }

  auto is_multigraph = graph.is_multigraph();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] = decompress_to_edgelist(
    handle,
    graph_view,
    edge_weights
      ? std::optional<edge_property_view_t<edge_t, weight_t const*>>{(*edge_weights).view()}
      : std::nullopt,
    std::make_optional<raft::device_span<vertex_t const>>((*renumber_map).data(),
                                                          (*renumber_map).size()));
  graph = graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    symmetrized_edge_weights{};
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, symmetrized_edge_weights, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, true},
      true);

  return std::make_tuple(
    std::move(symmetrized_graph), std::move(symmetrized_edge_weights), std::move(new_renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<
    graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
    std::optional<
      edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
    std::optional<rmm::device_uvector<vertex_t>>>>
symmetrize_graph_impl(
  raft::handle_t const& handle,
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&& edge_weights,
  std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check)
{
  auto graph_view = graph.view();

  CUGRAPH_EXPECTS(
    !renumber_map.has_value() ||
      (*renumber_map).size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid input arguments: if renumber_map.has_value() == true, (*renumber_map).size() should "
    "match with the local vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  if (graph.is_symmetric()) {
    return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
  }

  auto number_of_vertices = graph.number_of_vertices();
  auto is_multigraph      = graph.is_multigraph();
  bool renumber           = renumber_map.has_value();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] = decompress_to_edgelist(
    handle,
    graph_view,
    edge_weights
      ? std::optional<edge_property_view_t<edge_t, weight_t const*>>{(*edge_weights).view()}
      : std::nullopt,
    renumber_map ? std::make_optional<raft::device_span<vertex_t const>>((*renumber_map).data(),
                                                                         (*renumber_map).size())
                 : std::nullopt);
  graph = graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  auto vertices = renumber ? std::move(renumber_map)
                           : std::make_optional<rmm::device_uvector<vertex_t>>(number_of_vertices,
                                                                               handle.get_stream());
  if (!renumber) {
    thrust::sequence(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), vertex_t{0});
  }

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    symmetrized_edge_weights{};
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, symmetrized_edge_weights, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertices),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, true},
      renumber);

  return std::make_tuple(
    std::move(symmetrized_graph), std::move(symmetrized_edge_weights), std::move(new_renumber_map));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
  std::optional<
    edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
symmetrize_graph(
  raft::handle_t const& handle,
  graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& graph,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                weight_t>>&& edge_weights,
  std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
  bool reciprocal,
  bool do_expensive_check)
{
  return symmetrize_graph_impl(handle,
                               std::move(graph),
                               std::move(edge_weights),
                               std::move(renumber_map),
                               reciprocal,
                               do_expensive_check);
}

}  // namespace cugraph
