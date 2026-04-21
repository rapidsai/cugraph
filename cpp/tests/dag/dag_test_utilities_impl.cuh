/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "dag/dag_test_utilities.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/make_initialized_edge_property.cuh>
#include <cugraph/prims/transform_e.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>

#include <raft/core/handle.hpp>

#include <cuda/functional>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, bool multi_gpu>
cugraph::edge_property_t<edge_t, bool> build_acyclic_edge_mask(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view)
{
  // SCC partitions vertices into strongly connected components. Two vertices share a label iff
  // they are in the same SCC. A non-self-loop edge u->v is contained in a cycle iff u and v are
  // in the same SCC (and that SCC has >1 vertex, which is implied by u != v && label(u) ==
  // label(v)). So: keep edge <=> (u != v) && (label(u) != label(v)).
  auto labels = cugraph::strongly_connected_components(handle, graph_view, false);

  cugraph::edge_src_property_t<vertex_t, vertex_t> src_labels(handle, graph_view);
  cugraph::edge_dst_property_t<vertex_t, vertex_t> dst_labels(handle, graph_view);
  cugraph::update_edge_src_property(
    handle, graph_view, labels.begin(), src_labels.mutable_view());
  cugraph::update_edge_dst_property(
    handle, graph_view, labels.begin(), dst_labels.mutable_view());

  auto edge_mask = make_initialized_edge_property(handle, graph_view, false);

  cugraph::transform_e(
    handle,
    graph_view,
    src_labels.view(),
    dst_labels.view(),
    cugraph::edge_dummy_property_t{}.view(),
    cuda::proclaim_return_type<bool>(
      [] __device__(auto src, auto dst, auto src_label, auto dst_label, auto) {
        return (src != dst) && (src_label != dst_label);
      }),
    edge_mask.mutable_view());

  return edge_mask;
}

}  // namespace test
}  // namespace cugraph
