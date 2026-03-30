/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/fill_edge_src_dst_property.cuh"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {

/**
 * @brief Create an edge source property and fill the entire buffer with the given value.
 *
 * Create an edge source property object for @p graph_view and initialize the edge source
 * property values to @p input.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge source property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge source property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return edge_src_property_t<typename GraphViewType::vertex_type, T> with all entries set to @p
 * input.
 */
template <typename GraphViewType, typename T>
edge_src_property_t<typename GraphViewType::vertex_type, T> make_initialized_edge_src_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  T input,
  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  edge_src_property_t<vertex_t, T> edge_src_prop(handle, graph_view);
  fill_edge_src_property(
    handle, graph_view, edge_src_prop.mutable_view(), input, do_expensive_check);

  return edge_src_prop;
}

/**
 * @brief Create an edge destination property and fill the entire buffer with the given value.
 *
 * Create an edge destination property object for @p graph_view and initialize the edge destination
 * property values to @p input.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge destination property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge destination property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return edge_dst_property_t<typename GraphViewType::vertex_type, T> with all entries set to @p
 * input.
 */
template <typename GraphViewType, typename T>
edge_dst_property_t<typename GraphViewType::vertex_type, T> make_initialized_edge_dst_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  T input,
  bool do_expensive_check = false)
{
  using vertex_t = typename GraphViewType::vertex_type;

  if (do_expensive_check) {
    // currently, nothing to do
  }

  edge_dst_property_t<vertex_t, T> edge_dst_prop(handle, graph_view);
  fill_edge_dst_property(
    handle, graph_view, edge_dst_prop.mutable_view(), input, do_expensive_check);

  return edge_dst_prop;
}

}  // namespace cugraph
