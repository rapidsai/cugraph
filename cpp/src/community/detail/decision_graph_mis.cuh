/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common_methods.hpp"
#include "decision_graph_mis.hpp"
#include "maximal_independent_moves.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/vertex_property.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> vertices_in_mis_from_decision_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  raft::host_span<vertex_t const> vertex_partition_range_lasts,
  rmm::device_uvector<vertex_t>&& d_srcs,
  rmm::device_uvector<vertex_t>&& d_dsts)
{
  // NOTE: the maximum number of edges is the number of vertices in the graph,
  // so we can use the vertex type for the edge type
  using edge_t = vertex_t;

  constexpr bool decision_store_transposed = false;

  cugraph::graph_t<vertex_t, edge_t, decision_store_transposed, multi_gpu> decision_graph(handle);

  if constexpr (multi_gpu) {
    std::tie(d_srcs, d_dsts, std::ignore) =
      cugraph::shuffle_ext_edges(handle,
                                 std::move(d_srcs),
                                 std::move(d_dsts),
                                 std::vector<arithmetic_device_uvector_t>{},
                                 false);
  }

  std::optional<vertex_property_t<vertex_t, vertex_t>> renumber_map{std::nullopt};
  std::tie(decision_graph, std::ignore, renumber_map) =
    create_graph<vertex_t, edge_t, decision_store_transposed, multi_gpu>(
      handle,
      std::nullopt,
      std::move(d_srcs),
      std::move(d_dsts),
      std::vector<arithmetic_device_uvector_t>{},
      cugraph::graph_properties_t{false, false},
      true /* renumber */);

  auto decision_graph_view = decision_graph.view();

  auto vertices_in_mis =
    maximal_independent_moves<vertex_t, edge_t, multi_gpu>(handle, decision_graph_view, rng_state);

  vertices_in_mis = unrenumber_local_int_vertices(
    handle, decision_graph_view, std::move(vertices_in_mis), renumber_map->view());

  renumber_map = std::nullopt;

  if constexpr (multi_gpu) {
    std::tie(vertices_in_mis, std::ignore) =
      cugraph::shuffle_int_vertices(handle,
                                    std::move(vertices_in_mis),
                                    std::vector<cugraph::arithmetic_device_uvector_t>{},
                                    vertex_partition_range_lasts);
  }

  return vertices_in_mis;
}

}  // namespace detail
}  // namespace cugraph
