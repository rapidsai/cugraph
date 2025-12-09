/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
void check_correctness(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  rmm::device_uvector<edge_t> const& core_numbers,
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>> const& subgraph,
  size_t k);

}  // namespace test
}  // namespace cugraph
