/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "prims/edge_bucket.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>

namespace cugraph {
namespace detail {

/**
 * @brief Gather properties
 *
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>>
gather_sampled_properties(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  rmm::device_uvector<vertex_t>&& majors,
  rmm::device_uvector<vertex_t>&& minors,
  arithmetic_device_uvector_t&& multi_index,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views);

}  // namespace detail
}  // namespace cugraph
