/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
egonet_reference(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t const> ego_sources,
  int radius);

template <typename vertex_t, typename weight_t>
void egonet_validate(raft::handle_t const& handle,
                     rmm::device_uvector<vertex_t>& d_cugraph_egonet_src,
                     rmm::device_uvector<vertex_t>& d_cugraph_egonet_dst,
                     std::optional<rmm::device_uvector<weight_t>>& d_cugraph_egonet_wgt,
                     rmm::device_uvector<size_t>& d_cugraph_egonet_offsets,
                     rmm::device_uvector<vertex_t>& d_reference_egonet_src,
                     rmm::device_uvector<vertex_t>& d_reference_egonet_dst,
                     std::optional<rmm::device_uvector<weight_t>>& d_reference_egonet_wgt,
                     rmm::device_uvector<size_t>& d_reference_egonet_offsets);

}  // namespace test
}  // namespace cugraph
