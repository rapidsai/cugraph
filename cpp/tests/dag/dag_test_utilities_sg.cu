/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dag/dag_test_utilities_impl.cuh"

namespace cugraph {
namespace test {

template cugraph::edge_property_t<int32_t, bool> build_acyclic_edge_mask<int32_t, int32_t, false>(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int32_t, int32_t, false, false> const& graph_view);

template cugraph::edge_property_t<int64_t, bool> build_acyclic_edge_mask<int64_t, int64_t, false>(
  raft::handle_t const& handle,
  cugraph::graph_view_t<int64_t, int64_t, false, false> const& graph_view);

}  // namespace test
}  // namespace cugraph
