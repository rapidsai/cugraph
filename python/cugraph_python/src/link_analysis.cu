/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * PageRank over a transposed (incoming-edge) graph.
 *
 * The caller passes the edge-weight property. A missing property uses float32 and weight 1.
 * The score dtype matches the property that was passed.
 */
#include "cugraph_python.hpp"

#include <raft/util/cudart_utils.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cugraph_python {
namespace {

template <typename weight_t, typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<device_array_t, device_array_t, std::size_t, bool> pagerank_with_weights(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  double alpha,
  double epsilon,
  std::size_t max_iterations)
{
  static_assert(store_transposed, "pagerank reads a transposed graph");

  auto& handle           = graph.context->handle;
  auto stream            = handle.get_stream();
  auto view              = graph.native.view();
  auto const& number_map = *graph.number_map;

  // vertex_property_t overload. No personalization and no initial guess.
  auto [pagerank_values, metadata] =
    cugraph::pagerank<vertex_t, edge_t, weight_t, weight_t, multi_gpu>(
      handle,
      view,
      std::move(edge_weight_view),
      std::nullopt,
      std::nullopt,
      std::optional<cugraph::vertex_property_t<vertex_t, weight_t>>{},
      static_cast<weight_t>(alpha),
      static_cast<weight_t>(epsilon),
      max_iterations,
      /*do_expensive_check=*/false);

  auto vertices = std::make_shared<rmm::device_uvector<vertex_t>>(number_map.size(), stream);
  raft::copy(vertices->data(), number_map.data(), number_map.size(), stream);

  auto scores = std::make_shared<rmm::device_uvector<weight_t>>(pagerank_values.release_buffer());

  handle.sync_stream();
  return {device_array_t{std::move(vertices)}, device_array_t{std::move(scores)},
          metadata.number_of_iterations_, metadata.converged_};
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<device_array_t, device_array_t, std::size_t, bool> pagerank_impl(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  std::optional<edge_property_t> const& weights,
  double alpha,
  double epsilon,
  std::size_t max_iterations)
{
  if constexpr (!store_transposed) {
    throw std::invalid_argument("pagerank: requires a graph created with store_transposed=true");
  } else if (!graph.number_map) {
    throw std::logic_error("pagerank: graph has no number map (renumbering disabled)");
  } else if (!weights) {
    return pagerank_with_weights<float>(
      graph,
      std::optional<cugraph::edge_property_view_t<edge_t, float const*>>{},
      alpha,
      epsilon,
      max_iterations);
  } else {
    auto const* handle = std::get_if<edge_property_handle_t<edge_t>>(&weights->impl);
    if (handle == nullptr) {
      throw std::invalid_argument("pagerank: edge property type does not match the graph");
    }
    if (handle->state->graph_id != graph.graph_id) {
      throw std::invalid_argument("pagerank: edge property was created for a different graph");
    }
    return std::visit(
      [&](auto const& property) -> std::tuple<device_array_t, device_array_t, std::size_t, bool> {
        using property_t = std::decay_t<decltype(property)>;
        if constexpr (std::is_same_v<property_t, std::monostate>) {
          throw std::invalid_argument("pagerank: edge property is empty");
        } else {
          using weight_t = typename property_t::value_type;
          if constexpr (std::is_same_v<weight_t, float> || std::is_same_v<weight_t, double>) {
            return pagerank_with_weights<weight_t>(
              graph, std::make_optional(property.view()), alpha, epsilon, max_iterations);
          } else {
            throw std::invalid_argument("pagerank: edge weights must be float32 or float64");
          }
        }
      },
      handle->state->values[handle->index]);
  }
}

}  // namespace

std::tuple<device_array_t, device_array_t, std::size_t, bool> pagerank(
  graph_t const& graph,
  std::optional<edge_property_t> weights,
  double alpha,
  double epsilon,
  std::size_t max_iterations)
{
  return std::visit(
    [&](auto const& instance) -> std::tuple<device_array_t, device_array_t, std::size_t, bool> {
      return pagerank_impl(instance, weights, alpha, epsilon, max_iterations);
    },
    graph.impl);
}

}  // namespace cugraph_python
