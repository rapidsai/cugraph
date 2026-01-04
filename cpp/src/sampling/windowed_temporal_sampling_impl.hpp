/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @file windowed_temporal_sampling_impl.hpp
 * @brief Windowed temporal sampling combining B/C (window filtering) with D (inline temporal)
 *
 * This file provides a wrapper around temporal_neighbor_sample_impl that adds
 * window-based edge filtering:
 *
 * - B: Binary search for window bounds (O(log E))
 * - C: Incremental mask update for sliding windows (O(ΔE))  
 * - D: Inline temporal filtering during sampling (O(frontier_edges))
 *
 * References: CUDA Programming Guide - Cooperative Groups, Thrust algorithms
 */

#include "temporal_sampling_impl.hpp"
#include "detail/window_edge_mask.cuh"

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>

#include <optional>

namespace cugraph {
namespace detail {

/**
 * @brief State for incremental window updates (Optimization C)
 *
 * Maintains sorted edge indices and current window bounds for efficient
 * incremental mask updates when sliding the window.
 */
template <typename edge_t, typename time_stamp_t>
struct window_state_t {
  rmm::device_uvector<edge_t> sorted_edge_indices;
  rmm::device_uvector<time_stamp_t> sorted_edge_times;
  size_t current_start_idx{0};
  size_t current_end_idx{0};
  bool initialized{false};
  
  window_state_t(rmm::cuda_stream_view stream)
    : sorted_edge_indices(0, stream),
      sorted_edge_times(0, stream) {}
};

/**
 * @brief Initialize window state by sorting edges by time
 *
 * This is a one-time O(E log E) operation that enables O(log E) window
 * bound computation and O(ΔE) incremental updates.
 *
 * @param handle RAFT handle
 * @param edge_times Edge timestamps
 * @param num_edges Number of edges
 * @param state Output window state
 */
template <typename edge_t, typename time_stamp_t>
void initialize_window_state(
  raft::handle_t const& handle,
  time_stamp_t const* edge_times,
  edge_t num_edges,
  window_state_t<edge_t, time_stamp_t>& state)
{
  auto stream = handle.get_stream();
  
  // Allocate and initialize sorted indices
  state.sorted_edge_indices.resize(num_edges, stream);
  state.sorted_edge_times.resize(num_edges, stream);
  
  thrust::sequence(thrust::device.on(stream),
                   state.sorted_edge_indices.data(),
                   state.sorted_edge_indices.data() + num_edges);
  
  thrust::copy(thrust::device.on(stream),
               edge_times,
               edge_times + num_edges,
               state.sorted_edge_times.data());
  
  // Sort indices by time
  thrust::sort_by_key(thrust::device.on(stream),
                      state.sorted_edge_times.data(),
                      state.sorted_edge_times.data() + num_edges,
                      state.sorted_edge_indices.data());
  
  state.initialized = true;
}

/**
 * @brief Set window mask using binary search (Optimization B)
 *
 * Finds window bounds in O(log E) and sets mask in O(E_window).
 *
 * @param handle RAFT handle
 * @param state Window state with sorted edges
 * @param window_start Start of time window (inclusive)
 * @param window_end End of time window (exclusive)
 * @param edge_mask Edge mask to update
 * @param num_edges Total number of edges
 */
template <typename edge_t, typename time_stamp_t>
void set_window_mask(
  raft::handle_t const& handle,
  window_state_t<edge_t, time_stamp_t>& state,
  time_stamp_t window_start,
  time_stamp_t window_end,
  uint32_t* edge_mask,
  edge_t num_edges)
{
  CUGRAPH_EXPECTS(state.initialized, "Window state not initialized");
  
  // Binary search for window bounds
  auto [start_idx, end_idx] = compute_window_bounds_binary_search<time_stamp_t>(
    handle,
    state.sorted_edge_times.data(),
    state.sorted_edge_times.size(),
    window_start,
    window_end);
  
  // Set mask for edges in window
  set_mask_from_sorted_range<edge_t>(
    handle,
    edge_mask,
    num_edges,
    state.sorted_edge_indices.data(),
    start_idx,
    end_idx);
  
  // Update state
  state.current_start_idx = start_idx;
  state.current_end_idx = end_idx;
}

/**
 * @brief Update window mask incrementally (Optimization C)
 *
 * For sliding windows, only processes edges entering/leaving the window.
 * Complexity: O(ΔE) where ΔE is the number of edges in the delta.
 *
 * @param handle RAFT handle
 * @param state Window state with sorted edges
 * @param window_start New window start (inclusive)
 * @param window_end New window end (exclusive)
 * @param edge_mask Edge mask to update
 */
template <typename edge_t, typename time_stamp_t>
void update_window_mask_incremental(
  raft::handle_t const& handle,
  window_state_t<edge_t, time_stamp_t>& state,
  time_stamp_t window_start,
  time_stamp_t window_end,
  uint32_t* edge_mask)
{
  CUGRAPH_EXPECTS(state.initialized, "Window state not initialized");
  
  // Compute new bounds
  auto [new_start_idx, new_end_idx] = compute_window_bounds_binary_search<time_stamp_t>(
    handle,
    state.sorted_edge_times.data(),
    state.sorted_edge_times.size(),
    window_start,
    window_end);
  
  // Update mask incrementally
  update_mask_incremental<edge_t>(
    handle,
    edge_mask,
    state.sorted_edge_indices.data(),
    state.current_start_idx, new_start_idx,  // edges leaving (old start to new start)
    state.current_end_idx, new_end_idx);     // edges entering (old end to new end)
  
  // Update state
  state.current_start_idx = new_start_idx;
  state.current_end_idx = new_end_idx;
}

/**
 * @brief Windowed temporal neighbor sampling with B+C+D optimizations
 *
 * This function combines:
 * - B: Binary search for window bounds
 * - C: Incremental mask updates for sliding windows
 * - D: Inline temporal filtering during sampling
 *
 * @tparam All template parameters same as temporal_neighbor_sample_impl
 *
 * @param handle RAFT handle
 * @param rng_state Random state
 * @param graph_view Graph view
 * @param edge_weight_view Optional edge weights
 * @param edge_id_view Optional edge IDs
 * @param edge_type_view Optional edge types
 * @param edge_start_time_view Edge start times (required)
 * @param edge_end_time_view Optional edge end times
 * @param edge_bias_view Optional edge biases
 * @param starting_vertices Starting vertices for sampling
 * @param starting_vertex_times Vertex query times (for D optimization)
 * @param starting_vertex_labels Optional vertex labels
 * @param label_to_output_comm_rank Optional output rank mapping
 * @param fan_out Fan-out per hop
 * @param num_edge_types Number of edge types (for heterogeneous graphs)
 * @param sampling_flags Sampling configuration flags
 * @param window_start Start of time window (for B/C optimization)
 * @param window_end End of time window (for B/C optimization)
 * @param window_state Optional state for incremental updates
 * @param do_expensive_check Whether to perform expensive validation
 *
 * @return Sampled edges (sources, destinations, and optional properties)
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t,
          typename bias_t,
          typename label_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<time_stamp_t>>,
           std::optional<rmm::device_uvector<int32_t>>,
           std::optional<rmm::device_uvector<size_t>>>
windowed_temporal_neighbor_sample_impl(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_start_time_view,
  std::optional<edge_property_view_t<edge_t, time_stamp_t const*>> edge_end_time_view,
  std::optional<edge_property_view_t<edge_t, bias_t const*>> edge_bias_view,
  raft::device_span<vertex_t const> starting_vertices,
  std::optional<raft::device_span<time_stamp_t const>> starting_vertex_times,
  std::optional<raft::device_span<label_t const>> starting_vertex_labels,
  std::optional<raft::device_span<int32_t const>> label_to_output_comm_rank,
  raft::host_span<int32_t const> fan_out,
  std::optional<edge_type_t> num_edge_types,
  sampling_flags_t sampling_flags,
  std::optional<time_stamp_t> window_start,
  std::optional<time_stamp_t> window_end,
  std::optional<std::reference_wrapper<window_state_t<edge_t, time_stamp_t>>> window_state,
  bool do_expensive_check)
{
  // If window parameters provided, create a windowed graph view
  std::optional<cugraph::edge_property_t<edge_t, bool>> window_edge_mask{std::nullopt};
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> windowed_graph_view{graph_view};
  
  if (window_start && window_end) {
    // Create edge mask for window
    window_edge_mask = cugraph::edge_property_t<edge_t, bool>(handle, graph_view);
    
    auto num_edges = graph_view.compute_number_of_edges(handle);
    
    if (window_state) {
      // Use existing window state for incremental update (Optimization C)
      auto& state = window_state->get();
      
      if (!state.initialized) {
        // First call - initialize state and set full window mask
        // Note: This requires access to edge times as a contiguous array
        // For now, fall back to non-incremental path
        // TODO: Extract edge times to device array for initialization
        CUGRAPH_FAIL("Incremental window updates require pre-initialized window state");
      }
      
      // Update mask incrementally
      update_window_mask_incremental<edge_t, time_stamp_t>(
        handle,
        state,
        *window_start,
        *window_end,
        window_edge_mask->mutable_view().value_firsts()[0]);
        
    } else {
      // No window state - use the simpler set_window_edge_mask (Optimization B)
      // This scans all edges in O(E) time
      set_window_edge_mask<vertex_t, edge_t, time_stamp_t, multi_gpu>(
        handle,
        graph_view,
        edge_start_time_view,
        *window_start,
        *window_end,
        window_edge_mask->mutable_view());
    }
    
    // Attach window mask to graph view
    windowed_graph_view.attach_edge_mask(window_edge_mask->view());
  }
  
  // Call the existing temporal sampling with D optimization
  // Note: We pass the windowed_graph_view which may have window mask attached
  // The D optimization will do additional per-vertex temporal filtering
  return temporal_neighbor_sample_impl<vertex_t, edge_t, weight_t, edge_type_t,
                                        time_stamp_t, bias_t, label_t,
                                        store_transposed, multi_gpu>(
    handle,
    rng_state,
    windowed_graph_view,
    edge_weight_view,
    edge_id_view,
    edge_type_view,
    edge_start_time_view,
    edge_end_time_view,
    edge_bias_view,
    starting_vertices,
    starting_vertex_times,
    starting_vertex_labels,
    label_to_output_comm_rank,
    fan_out,
    num_edge_types,
    sampling_flags,
    do_expensive_check);
}

}  // namespace detail
}  // namespace cugraph
