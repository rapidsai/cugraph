/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>> simple_cycles_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  vertex_t length_bound,
  bool do_expensive_check)
{
  (void)graph_view;
  (void)seed_vertices;
  (void)length_bound;
  (void)do_expensive_check;

  rmm::device_uvector<vertex_t> cycle_vertices(0, handle.get_stream());
  rmm::device_uvector<size_t> cycle_offsets(size_t{1}, handle.get_stream());
  cycle_offsets.set_element_to_zero_async(size_t{0}, handle.get_stream());
  return std::make_tuple(std::move(cycle_vertices), std::move(cycle_offsets));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>> simple_cycles(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  vertex_t length_bound,
  bool do_expensive_check)
{
  return detail::simple_cycles_impl(
    handle, graph_view, seed_vertices, length_bound, do_expensive_check);
}

}  // namespace cugraph
