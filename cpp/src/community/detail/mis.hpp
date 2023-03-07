#pragma once
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<vertex_t> compute_mis(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view);
}
}  // namespace cugraph