/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/fill_edge_property.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>

namespace cugraph {

/**
 * @brief Create an edge property and fill the entire buffer with the given value.
 *
 * Create an edge property object for @p graph_view and initialize the edge property values to @p
 * input. Note that the entire edge buffer is initialized to the provided @p
 * input (ignoring the edge mask if @p graph_view has an attached edge mask); this contrasts
 * with the fill_edge_property function which only fills the edge values for un-masked edges if @p
 * graph_view has an attached edge mask.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam T Type of the edge property values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param input Edge property values will be set to @p input.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return edge_property_t<typename GraphViewType::edge_type, T> with all entries set to @p input.
 */
template <typename GraphViewType, typename T>
edge_property_t<typename GraphViewType::edge_type, T> make_initialized_edge_property(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  T input,
  bool do_expensive_check = false)
{
  using edge_t = typename GraphViewType::edge_type;

  edge_property_t<edge_t, T> edge_prop(handle, graph_view);
  if (graph_view.has_edge_mask()) {
    auto graph_view_copy = graph_view;
    graph_view_copy.clear_edge_mask();
    fill_edge_property(
      handle, graph_view_copy, edge_prop.mutable_view(), input, do_expensive_check);
  } else {
    fill_edge_property(handle, graph_view, edge_prop.mutable_view(), input, do_expensive_check);
  }

  return edge_prop;
}

}  // namespace cugraph
