/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sampling/detail/check_edge_bias_values.cuh"

namespace cugraph {
namespace detail {

template std::tuple<size_t, size_t> check_edge_bias_values(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, float const*> edge_bias_view);

template std::tuple<size_t, size_t> check_edge_bias_values(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  edge_property_view_t<int64_t, double const*> edge_bias_view);

}  // namespace detail
}  // namespace cugraph
