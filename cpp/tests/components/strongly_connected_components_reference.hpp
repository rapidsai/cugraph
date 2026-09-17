/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/graph.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <stack>
#include <vector>

namespace {

// Tarjan's strongly connected components algorithm.
// (https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
template <typename vertex_t, typename edge_t>
std::vector<vertex_t> strongly_connected_components_reference(std::span<edge_t const> offsets,
                                                              std::span<vertex_t const> indices)
{
  using index_t                   = size_t;
  constexpr index_t invalid_index = std::numeric_limits<index_t>::max();

  auto num_vertices = static_cast<vertex_t>(offsets.size() - 1);

  std::vector<index_t> index(num_vertices, invalid_index);
  std::vector<index_t> lowlink(num_vertices, invalid_index);
  std::vector<bool> on_stack(num_vertices, false);
  std::stack<vertex_t> S{};
  index_t current_index{0};
  vertex_t next_component_id{0};

  std::vector<vertex_t> components(num_vertices, cugraph::invalid_component_id<vertex_t>::value);

  auto strongconnect = [&](vertex_t v, auto&& strongconnect_ref) -> void {
    index[v]   = current_index;
    lowlink[v] = current_index;
    ++current_index;
    S.push(v);
    on_stack[v] = true;

    // Consider successors of v (outgoing edges)
    edge_t nbr_begin = offsets[v];
    edge_t nbr_end   = offsets[v + 1];
    for (edge_t e = nbr_begin; e != nbr_end; ++e) {
      vertex_t w = indices[e];
      if (index[w] == invalid_index) {
        strongconnect_ref(w, strongconnect_ref);
        lowlink[v] = std::min(lowlink[v], lowlink[w]);
      } else if (on_stack[w]) {
        lowlink[v] = std::min(lowlink[v], index[w]);
      }
    }

    // If v is a root node, pop the stack and assign component id
    if (lowlink[v] == index[v]) {
      vertex_t w;
      do {
        w = S.top();
        S.pop();
        on_stack[w]   = false;
        components[w] = next_component_id;
      } while (w != v);
      ++next_component_id;
    }
  };

  for (vertex_t v = 0; v < num_vertices; ++v) {
    if (index[v] == invalid_index) { strongconnect(v, strongconnect); }
  }

  return components;
}

}  // namespace
