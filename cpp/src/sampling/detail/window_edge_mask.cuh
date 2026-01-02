/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "prims/transform_e.cuh"

#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cugraph {
namespace detail {

/**
 * @brief Set edge mask based on a time window [window_start, window_end).
 *
 * This function creates an edge mask where only edges with timestamps
 * in the specified time window are included. This is useful for window-based
 * temporal sampling where we want to restrict sampling to a specific time period.
 *
 * Complexity: O(E) parallel comparisons
 *
 * @tparam vertex_t Vertex type
 * @tparam edge_t Edge type
 * @tparam time_stamp_t Timestamp type
 * @tparam multi_gpu Multi-GPU flag
 *
 * @param handle RAFT handle
 * @param graph_view Graph view
 * @param edge_time_view Edge property view containing edge timestamps
 * @param window_start Start of time window (inclusive)
 * @param window_end End of time window (exclusive)
 * @param edge_mask_view Output edge mask view
 */
template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
void set_window_edge_mask(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  time_stamp_t window_start,
  time_stamp_t window_end,
  edge_property_view_t<edge_t, uint32_t*, bool> edge_mask_view)
{
  // Use transform_e to set mask bits based on time window
  // This is O(E) but with very low constants - just a comparison per edge
  cugraph::transform_e(
    handle,
    graph_view,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_time_view,
    [window_start, window_end] __device__(
      auto src, auto dst, auto, auto, auto edge_time) {
      // Include edge if timestamp is in [window_start, window_end)
      return (edge_time >= window_start) && (edge_time < window_end);
    },
    edge_mask_view,
    false);
}

/**
 * @brief Compute window bounds for sorted edge times using binary search.
 *
 * If edges are pre-sorted by time, this function can find the window bounds
 * in O(log E) time. The caller can then use these bounds to efficiently
 * process only edges in the window.
 *
 * Note: This assumes edge_times is sorted. If not sorted, use set_window_edge_mask instead.
 *
 * @tparam time_stamp_t Timestamp type
 *
 * @param handle RAFT handle
 * @param sorted_edge_times Device array of sorted edge timestamps
 * @param num_edges Number of edges
 * @param window_start Start of time window (inclusive)
 * @param window_end End of time window (exclusive)
 * @return Pair of (start_idx, end_idx) for edges in the window
 */
template <typename time_stamp_t>
std::pair<size_t, size_t> compute_window_bounds_binary_search(
  raft::handle_t const& handle,
  time_stamp_t const* sorted_edge_times,
  size_t num_edges,
  time_stamp_t window_start,
  time_stamp_t window_end)
{
  // Use thrust binary search for O(log E) complexity
  auto stream = handle.get_stream();
  
  auto start_iter = thrust::lower_bound(
    thrust::device.on(stream),
    sorted_edge_times,
    sorted_edge_times + num_edges,
    window_start);
  
  auto end_iter = thrust::lower_bound(
    thrust::device.on(stream),
    sorted_edge_times,
    sorted_edge_times + num_edges,
    window_end);
  
  size_t start_idx = thrust::distance(sorted_edge_times, start_iter);
  size_t end_idx = thrust::distance(sorted_edge_times, end_iter);
  
  return std::make_pair(start_idx, end_idx);
}

/**
 * @brief Set edge mask using sorted edge index range.
 *
 * For pre-sorted edges, this sets the mask for edges in [start_idx, end_idx).
 * This is O(E_window) which can be much faster than O(E) if window is small.
 *
 * @tparam edge_t Edge type
 *
 * @param handle RAFT handle
 * @param edge_mask Output edge mask array (packed booleans)
 * @param num_edges Total number of edges
 * @param sorted_edge_indices Device array mapping sorted position to original edge index
 * @param start_idx Start index in sorted order
 * @param end_idx End index in sorted order
 */
template <typename edge_t>
void set_mask_from_sorted_range(
  raft::handle_t const& handle,
  uint32_t* edge_mask,
  edge_t num_edges,
  edge_t const* sorted_edge_indices,
  size_t start_idx,
  size_t end_idx)
{
  auto stream = handle.get_stream();
  
  // First clear the entire mask
  size_t num_mask_words = (num_edges + 31) / 32;
  thrust::fill(thrust::device.on(stream),
               edge_mask,
               edge_mask + num_mask_words,
               static_cast<uint32_t>(0));
  
  // Then set bits for edges in the window
  // Use atomic OR since edges may map to the same mask word
  size_t num_window_edges = end_idx - start_idx;
  if (num_window_edges > 0) {
    thrust::for_each(
      thrust::device.on(stream),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_window_edges),
      [edge_mask, sorted_edge_indices, start_idx] __device__(size_t i) {
        edge_t edge_idx = sorted_edge_indices[start_idx + i];
        uint32_t word_idx = edge_idx / 32;
        uint32_t bit_idx = edge_idx % 32;
        atomicOr(&edge_mask[word_idx], 1u << bit_idx);
      });
  }
}

/**
 * @brief Incrementally update edge mask for sliding window.
 *
 * When sliding a time window, only process edges leaving and entering the window.
 * This is O(ΔE) where ΔE is the number of edges in the delta.
 *
 * For a 1-day step on 300M edges over 730 days: ΔE ≈ 410K (0.14% of total)
 *
 * @tparam edge_t Edge type
 *
 * @param handle RAFT handle
 * @param edge_mask Edge mask array (packed booleans)
 * @param sorted_edge_indices Device array mapping sorted position to original edge index
 * @param leaving_start Start index of edges leaving the window
 * @param leaving_end End index of edges leaving the window
 * @param entering_start Start index of edges entering the window
 * @param entering_end End index of edges entering the window
 */
template <typename edge_t>
void update_mask_incremental(
  raft::handle_t const& handle,
  uint32_t* edge_mask,
  edge_t const* sorted_edge_indices,
  size_t leaving_start,
  size_t leaving_end,
  size_t entering_start,
  size_t entering_end)
{
  auto stream = handle.get_stream();
  
  // Clear bits for edges leaving the window
  size_t num_leaving = leaving_end - leaving_start;
  if (num_leaving > 0) {
    thrust::for_each(
      thrust::device.on(stream),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_leaving),
      [edge_mask, sorted_edge_indices, leaving_start] __device__(size_t i) {
        edge_t edge_idx = sorted_edge_indices[leaving_start + i];
        uint32_t word_idx = edge_idx / 32;
        uint32_t bit_idx = edge_idx % 32;
        atomicAnd(&edge_mask[word_idx], ~(1u << bit_idx));
      });
  }
  
  // Set bits for edges entering the window
  size_t num_entering = entering_end - entering_start;
  if (num_entering > 0) {
    thrust::for_each(
      thrust::device.on(stream),
      thrust::make_counting_iterator<size_t>(0),
      thrust::make_counting_iterator<size_t>(num_entering),
      [edge_mask, sorted_edge_indices, entering_start] __device__(size_t i) {
        edge_t edge_idx = sorted_edge_indices[entering_start + i];
        uint32_t word_idx = edge_idx / 32;
        uint32_t bit_idx = edge_idx % 32;
        atomicOr(&edge_mask[word_idx], 1u << bit_idx);
      });
  }
}

}  // namespace detail
}  // namespace cugraph
