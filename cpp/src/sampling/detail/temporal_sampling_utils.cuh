/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/sampling_functions.hpp>

#include <raft/core/device_span.hpp>

#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>

#include <limits>

namespace cugraph {
namespace detail {

__host__ __device__ inline bool is_temporal_decreasing(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return temporal_sampling_comparison == temporal_sampling_comparison_t::STRICTLY_DECREASING ||
         temporal_sampling_comparison == temporal_sampling_comparison_t::MONOTONICALLY_DECREASING;
}

__host__ __device__ inline bool is_fixed_window(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return temporal_sampling_comparison == temporal_sampling_comparison_t::FIXED_WINDOW;
}

// Increasing/decreasing walks tighten the frontier window-start to the most recently sampled
// edge's time at every hop.  FIXED_WINDOW instead keeps applying each seed's original window-start
// at every hop, so it must never be replaced by a sampled edge time.
__host__ __device__ inline bool propagates_sampled_edge_times_as_window_start(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return !is_fixed_window(temporal_sampling_comparison);
}

// Sentinel for an absent / unbounded frontier time.
// Increasing walks treat window_start as a lower bound (window_start <= edge_time) => lowest().
// Decreasing walks treat window_start as an upper bound (window_start >= edge_time) => max().
template <typename time_stamp_t>
__host__ __device__ inline time_stamp_t unbounded_temporal_window_start(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return is_temporal_decreasing(temporal_sampling_comparison)
           ? std::numeric_limits<time_stamp_t>::max()
           : std::numeric_limits<time_stamp_t>::lowest();
}

// Sentinel for an absent / unbounded second time bound.
// Increasing walks treat window_end as an upper bound (edge_time <= window_end) => max().
// Decreasing walks treat window_end as a lower bound (edge_time >= window_end) => lowest().
template <typename time_stamp_t>
__host__ __device__ inline time_stamp_t unbounded_temporal_window_end(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return is_temporal_decreasing(temporal_sampling_comparison)
           ? std::numeric_limits<time_stamp_t>::lowest()
           : std::numeric_limits<time_stamp_t>::max();
}

template <typename time_stamp_t>
__host__ __device__ inline bool passes_temporal_filter(
  temporal_sampling_comparison_t temporal_sampling_comparison,
  time_stamp_t key_time,
  time_stamp_t window_end,
  time_stamp_t edge_time)
{
  switch (temporal_sampling_comparison) {
    case temporal_sampling_comparison_t::STRICTLY_INCREASING:
      return (key_time < edge_time) && (edge_time <= window_end);
    case temporal_sampling_comparison_t::MONOTONICALLY_INCREASING:
      return (key_time <= edge_time) && (edge_time <= window_end);
    case temporal_sampling_comparison_t::STRICTLY_DECREASING:
      return (key_time > edge_time) && (edge_time >= window_end);
    case temporal_sampling_comparison_t::MONOTONICALLY_DECREASING:
      return (key_time >= edge_time) && (edge_time >= window_end);
    case temporal_sampling_comparison_t::FIXED_WINDOW:
      return (key_time <= edge_time) && (edge_time <= window_end);
  }
  return false;
}

// Binary-search a sorted major key table.  Returns false when major is absent.
template <typename vertex_t>
__device__ inline bool try_find_temporal_key_index(raft::device_span<vertex_t const> majors,
                                                   vertex_t major,
                                                   size_t& idx)
{
  auto it = thrust::lower_bound(thrust::seq, majors.begin(), majors.end(), major);
  if (it == majors.end() || *it != major) { return false; }
  idx = static_cast<size_t>(cuda::std::distance(majors.begin(), it));
  return true;
}

// Binary-search a sorted (major, label) key table.  Returns false when the key is absent.
template <typename vertex_t, typename label_t>
__device__ inline bool try_find_temporal_key_index(raft::device_span<vertex_t const> majors,
                                                   raft::device_span<label_t const> labels,
                                                   vertex_t major,
                                                   label_t label,
                                                   size_t& idx)
{
  auto begin = thrust::make_zip_iterator(majors.begin(), labels.begin());
  auto end   = thrust::make_zip_iterator(majors.end(), labels.end());
  auto it    = thrust::lower_bound(thrust::seq, begin, end, cuda::std::make_tuple(major, label));
  if (it == end || cuda::std::get<0>(*it) != major || cuda::std::get<1>(*it) != label) {
    return false;
  }
  idx = static_cast<size_t>(cuda::std::distance(begin, it));
  return true;
}

}  // namespace detail
}  // namespace cugraph
