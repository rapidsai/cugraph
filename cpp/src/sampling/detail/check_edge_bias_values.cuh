/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/count_if_e.cuh>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/vertex_property.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/tuple>

#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename bias_t, bool multi_gpu>
std::tuple<size_t, size_t> check_edge_bias_values(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, bias_t const*> edge_bias_view)
{
  auto num_negative_edge_weights =
    count_if_e(handle,
               graph_view,
               edge_src_dummy_property_t{}.view(),
               edge_dst_dummy_property_t{}.view(),
               edge_bias_view,
               [] __device__(vertex_t, vertex_t, auto, auto, bias_t b) { return b < 0.0; });

  auto bias_sums = compute_out_weight_sums(handle, graph_view, edge_bias_view, false);
  auto num_overflows =
    count_if_v(handle, graph_view, bias_sums.view(), [] __device__(vertex_t, auto sum) {
      return sum > std::numeric_limits<bias_t>::max();
    });

  return std::make_tuple(static_cast<size_t>(num_negative_edge_weights),
                         static_cast<size_t>(num_overflows));
}

}  // namespace detail
}  // namespace cugraph
