/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

#include "detail/window_edge_mask.cuh"
#include "temporal_sampling_impl.hpp"
#include "window_state_fwd.hpp"

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>

#include <cstdlib>
#include <optional>

namespace cugraph {
namespace detail {

// window_state_t is defined in window_state_fwd.hpp

/**
 * @brief Initialize window state by sorting edges by time
 *
 * This is a one-time O(E log E) operation that enables O(log E) window
 * bound computation and O(ΔE) incremental updates.
 *
 * If assume_temporally_sorted_edges is true, the edges are assumed to already
 * be sorted by time (e.g., if edge_start_time_array was sorted at graph
 * creation). This reduces initialization from O(E log E) to O(E).
 *
 * @param handle RAFT handle
 * @param edge_times Edge timestamps
 * @param num_edges Number of edges
 * @param state Output window state
 * @param assume_temporally_sorted_edges If true, skip sorting (edges already sorted by time)
 */
template <typename edge_t, typename time_stamp_t>
void initialize_window_state(raft::handle_t const& handle,
                             time_stamp_t const* edge_times,
                             edge_t num_edges,
                             window_state_t<edge_t, time_stamp_t>& state,
                             bool assume_temporally_sorted_edges = false)
{
  auto stream = handle.get_stream();

  // Allocate and initialize sorted indices
  state.sorted_edge_indices.resize(num_edges, stream);
  state.sorted_edge_times.resize(num_edges, stream);

  thrust::sequence(thrust::device.on(stream),
                   state.sorted_edge_indices.data(),
                   state.sorted_edge_indices.data() + num_edges);

  thrust::copy(
    thrust::device.on(stream), edge_times, edge_times + num_edges, state.sorted_edge_times.data());

  if (!assume_temporally_sorted_edges) {
    // Sort indices by time - O(E log E)
    thrust::sort_by_key(thrust::device.on(stream),
                        state.sorted_edge_times.data(),
                        state.sorted_edge_times.data() + num_edges,
                        state.sorted_edge_indices.data());
  }
  // If assume_temporally_sorted_edges, edges are already in time order,
  // so sorted_edge_indices is just [0, 1, 2, ...] which maps directly
  // to edges in time order.

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
void set_window_mask(raft::handle_t const& handle,
                     window_state_t<edge_t, time_stamp_t>& state,
                     time_stamp_t window_start,
                     time_stamp_t window_end,
                     uint32_t* edge_mask,
                     edge_t num_edges)
{
  CUGRAPH_EXPECTS(state.initialized, "Window state not initialized");

  // Binary search for window bounds
  auto [start_idx, end_idx] =
    compute_window_bounds_binary_search<time_stamp_t>(handle,
                                                      state.sorted_edge_times.data(),
                                                      state.sorted_edge_times.size(),
                                                      window_start,
                                                      window_end);

  // Set mask for edges in window
  set_mask_from_sorted_range<edge_t>(
    handle, edge_mask, num_edges, state.sorted_edge_indices.data(), start_idx, end_idx);

  // Update state
  state.current_start_idx = start_idx;
  state.current_end_idx   = end_idx;
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
void update_window_mask_incremental(raft::handle_t const& handle,
                                    window_state_t<edge_t, time_stamp_t>& state,
                                    time_stamp_t window_start,
                                    time_stamp_t window_end,
                                    uint32_t* edge_mask)
{
  CUGRAPH_EXPECTS(state.initialized, "Window state not initialized");

  // Compute new bounds
  auto [new_start_idx, new_end_idx] =
    compute_window_bounds_binary_search<time_stamp_t>(handle,
                                                      state.sorted_edge_times.data(),
                                                      state.sorted_edge_times.size(),
                                                      window_start,
                                                      window_end);

  // Robustness: incremental update assumes the mask currently represents the previous window.
  // Also assumes forward motion for O(ΔE) updates. If the window shrinks or moves backward,
  // fall back to setting the mask from scratch.
  if ((new_start_idx < state.current_start_idx) || (new_end_idx < state.current_end_idx)) {
    set_mask_from_sorted_range<edge_t>(handle,
                                       edge_mask,
                                       static_cast<edge_t>(state.sorted_edge_times.size()),
                                       state.sorted_edge_indices.data(),
                                       new_start_idx,
                                       new_end_idx);
    state.current_start_idx = new_start_idx;
    state.current_end_idx   = new_end_idx;
    return;
  }

  // Update mask incrementally
  update_mask_incremental<edge_t>(handle,
                                  edge_mask,
                                  state.sorted_edge_indices.data(),
                                  state.current_start_idx,
                                  new_start_idx,  // edges leaving (old start to new start)
                                  state.current_end_idx,
                                  new_end_idx);  // edges entering (old end to new end)

  // Update state
  state.current_start_idx = new_start_idx;
  state.current_end_idx   = new_end_idx;
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
 * @param assume_temporally_sorted_edges If true, edges are assumed pre-sorted by time.
 *        This enables O(log E) binary search without needing window_state.
 *        Set to true when edge_start_time_array was sorted at graph creation.
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
  bool do_expensive_check,
  bool assume_temporally_sorted_edges = false)
{
  // Debug/benchmark knob: force the O(E) transform_e scan path even when B/C are available.
  // This is intended to enable apples-to-apples A/B comparisons (windowed baseline vs B+C+D)
  // from Python without adding new API parameters. Off by default.
  //
  // Environment variable: CUGRAPH_WINDOWED_TEMPORAL_FORCE_OE=1
  static bool const force_oe_scan = []() {
    auto const* v = std::getenv("CUGRAPH_WINDOWED_TEMPORAL_FORCE_OE");
    return (v != nullptr) && (v[0] == '1');
  }();

  // Default behavior: avoid attaching a global edge mask for sampling (fan_out > 0) because it
  // forces the expensive masked-sampling pipeline (partition/unique-keys, etc.).
  //
  // If you need the legacy edge-mask behavior (e.g., gather path fan_out < 0, or for A/B),
  // set CUGRAPH_WINDOWED_TEMPORAL_USE_EDGE_MASK=1.
  static bool const force_edge_mask = []() {
    auto const* v = std::getenv("CUGRAPH_WINDOWED_TEMPORAL_USE_EDGE_MASK");
    return (v != nullptr) && (v[0] == '1');
  }();

  bool has_gather_fanout{false};
  for (size_t i = 0; i < fan_out.size(); ++i) {
    if (fan_out[i] < 0) {
      has_gather_fanout = true;
      break;
    }
  }
  bool use_edge_mask = force_edge_mask || has_gather_fanout;

  // If window parameters provided, create a windowed graph view
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> windowed_graph_view{graph_view};

  if (use_edge_mask && window_start && window_end) {
    auto num_edges = graph_view.compute_number_of_edges(handle);

    if (force_oe_scan) {
      std::optional<cugraph::edge_property_t<edge_t, bool>> window_edge_mask{std::nullopt};
      window_edge_mask = cugraph::edge_property_t<edge_t, bool>(handle, graph_view);
      set_window_edge_mask<vertex_t, edge_t, time_stamp_t, multi_gpu>(
        handle,
        graph_view,
        edge_start_time_view,
        *window_start,
        *window_end,
        window_edge_mask->mutable_view());
      windowed_graph_view.attach_edge_mask(window_edge_mask->view());
    } else

      if (window_state) {
      // Use existing window state for incremental update (Optimization C)
      auto& state = window_state->get();

      // Ensure persisted packed mask storage exists (Optimization C requires mask persistence)
      state.ensure_edge_mask_size(num_edges, handle.get_stream());

      if (!state.initialized) {
        // First call with window_state - initialize it
        // Get edge times from the edge property view
        auto edge_times_ptr = edge_start_time_view.value_firsts()[0];

        // Optional validation: if the caller claims the graph's internal edge ordering is
        // temporally sorted, validate that claim (O(E)) only when do_expensive_check is enabled.
        if (assume_temporally_sorted_edges && do_expensive_check) {
          auto stream    = handle.get_stream();
          bool is_sorted = thrust::is_sorted(
            thrust::device.on(stream), edge_times_ptr, edge_times_ptr + num_edges);
          CUGRAPH_EXPECTS(
            is_sorted,
            "assume_temporally_sorted_edges=true but edge_start_time is not sorted in the graph's "
            "internal edge ordering (graph construction may reorder edges). Disable the flag or "
            "let cuGraph sort times once by using assume_temporally_sorted_edges=false.");
        }

        // Initialize window state (O(E log E) one-time cost, or O(E) if edges are already sorted).
        //
        // IMPORTANT: We cannot assume edges are temporally sorted in the graph's internal edge
        // ordering. Graph construction often reorders edges (e.g., by major vertex) to build
        // CSR/CSC, which can destroy time-sortedness even if the input COO was time-sorted.
        //
        // Use the caller-provided flag to decide whether to skip sorting.
        initialize_window_state<edge_t, time_stamp_t>(
          handle, edge_times_ptr, num_edges, state, assume_temporally_sorted_edges);

        // First windowed call: set mask from scratch.
        //
        // NOTE: We must NOT use the incremental updater here because the current mask
        // does not represent any prior window yet (state.current_* defaults to 0).
        // Using update_window_mask_incremental from an "empty" state can incorrectly
        // re-add edges below new_start (e.g., edge at time=100 when window_start=200).
        set_window_mask<edge_t, time_stamp_t>(
          handle, state, *window_start, *window_end, state.edge_mask_words.data(), num_edges);
      } else {
        // Subsequent calls - use incremental update (O(ΔE))
        update_window_mask_incremental<edge_t, time_stamp_t>(
          handle, state, *window_start, *window_end, state.edge_mask_words.data());
      }

      // Attach persisted packed edge mask to graph view (single-GPU => single partition)
      auto mask_view = cugraph::edge_property_view_t<edge_t, uint32_t const*, bool>(
        std::vector<uint32_t const*>{state.edge_mask_words.data()}, std::vector<edge_t>{num_edges});
      windowed_graph_view.attach_edge_mask(mask_view);

    } else if (assume_temporally_sorted_edges) {
      // Without persistent window_state, we need a per-call mask buffer.
      // This path is not optimized for O(ΔE) but still avoids O(E) scanning.
      std::optional<cugraph::edge_property_t<edge_t, bool>> window_edge_mask{std::nullopt};
      window_edge_mask = cugraph::edge_property_t<edge_t, bool>(handle, graph_view);

      // Edges are pre-sorted by time - use O(log E) binary search
      // Note: Without persistent window_state, we get O(log E) + O(E_window)
      // which is better than O(E) transform_e but not as good as O(ΔE) incremental
      //
      // For full B+C+D optimization (O(ΔE)), pass a persistent window_state.
      auto stream         = handle.get_stream();
      auto edge_times_ptr = edge_start_time_view.value_firsts()[0];

      // Safety check: "assume_temporally_sorted_edges" must refer to the *graph's internal*
      // edge ordering. Graph construction often reorders edges (e.g., by major vertex) to build
      // CSR/CSC, which can destroy time-sortedness even if the input COO was time-sorted.
      //
      // If internal ordering is not sorted, binary search bounds would be incorrect, so we
      // fall back to the safe O(E) mask build.
      if (do_expensive_check) {
        bool is_sorted =
          thrust::is_sorted(thrust::device.on(stream), edge_times_ptr, edge_times_ptr + num_edges);
        CUGRAPH_EXPECTS(
          is_sorted,
          "assume_temporally_sorted_edges=true but edge_start_time is not sorted in the graph's "
          "internal edge ordering (graph construction may reorder edges). Disable the flag or use "
          "the window_state path.");
      }

      // Binary search for window bounds - O(log E)
      auto [start_idx, end_idx] = compute_window_bounds_binary_search<time_stamp_t>(
        handle, edge_times_ptr, num_edges, *window_start, *window_end);

      // For pre-sorted edges, edge index == sorted position
      // Set mask directly without needing sorted_indices array - O(E_window)
      auto* edge_mask       = window_edge_mask->mutable_view().value_firsts()[0];
      size_t num_mask_words = (num_edges + 31) / 32;

      // Clear entire mask - O(E/32)
      thrust::fill(
        thrust::device.on(stream), edge_mask, edge_mask + num_mask_words, static_cast<uint32_t>(0));

      // Set bits for edges in window [start_idx, end_idx) - O(E_window)
      size_t num_window_edges = end_idx - start_idx;
      if (num_window_edges > 0) {
        thrust::for_each(thrust::device.on(stream),
                         thrust::make_counting_iterator<size_t>(start_idx),
                         thrust::make_counting_iterator<size_t>(end_idx),
                         [edge_mask] __device__(size_t edge_idx) {
                           uint32_t word_idx = edge_idx / 32;
                           uint32_t bit_idx  = edge_idx % 32;
                           atomicOr(&edge_mask[word_idx], 1u << bit_idx);
                         });
      }

      // Attach window mask to graph view
      windowed_graph_view.attach_edge_mask(window_edge_mask->view());

    } else {
      std::optional<cugraph::edge_property_t<edge_t, bool>> window_edge_mask{std::nullopt};
      window_edge_mask = cugraph::edge_property_t<edge_t, bool>(handle, graph_view);

      // No window state and edges not sorted - use O(E) transform_e scan
      // This is the slowest path, used as fallback
      set_window_edge_mask<vertex_t, edge_t, time_stamp_t, multi_gpu>(
        handle,
        graph_view,
        edge_start_time_view,
        *window_start,
        *window_end,
        window_edge_mask->mutable_view());

      // Attach window mask to graph view
      windowed_graph_view.attach_edge_mask(window_edge_mask->view());
    }
  }

  // Call the existing temporal sampling with D optimization
  // Note: We pass the windowed_graph_view which may have window mask attached
  // The D optimization will do additional per-vertex temporal filtering
  return temporal_neighbor_sample_impl<vertex_t,
                                       edge_t,
                                       weight_t,
                                       edge_type_t,
                                       time_stamp_t,
                                       bias_t,
                                       label_t,
                                       store_transposed,
                                       multi_gpu>(handle,
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
                                                  window_start,
                                                  window_end,
                                                  do_expensive_check);
}

}  // namespace detail
}  // namespace cugraph
