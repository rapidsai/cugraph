/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {
namespace test {

// Build an edge mask that drops every edge whose source or destination lies in a non-trivial
// strongly connected component, plus every self-loop. Intended for DAG algorithm tests (e.g.
// topological_sort) so a cyclic test dataset can be masked down to a DAG before the algorithm
// is invoked.

template <typename vertex_t, typename edge_t, bool multi_gpu>
cugraph::edge_property_t<edge_t, bool> build_acyclic_edge_mask(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view);

}  // namespace test
}  // namespace cugraph
