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

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cugraph {
namespace detail {

__host__ __device__ inline bool is_temporal_decreasing(
  temporal_sampling_comparison_t temporal_sampling_comparison)
{
  return temporal_sampling_comparison == temporal_sampling_comparison_t::STRICTLY_DECREASING ||
         temporal_sampling_comparison == temporal_sampling_comparison_t::MONOTONICALLY_DECREASING;
}

// Increasing/decreasing walks tighten the frontier window-start to the most recently sampled
// edge's time at every hop.  fixed_window instead keeps applying each seed's original window-start
// at every hop, so it must never be replaced by a sampled edge time.
__host__ __device__ inline bool propagates_sampled_edge_times_as_window_start(bool fixed_window)
{
  return !fixed_window;
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
    case temporal_sampling_comparison_t::LAST: return false;
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

// ---------------------------------------------------------------------------
// LAST (last-n) neighbor selection
// ---------------------------------------------------------------------------
// LAST keeps the same eligible set as RANDOM temporal sampling: the destination
// must be unvisited (per label, when labels are used) and the edge time must
// pass passes_temporal_filter.  Among those edges it deterministically keeps
// fanout K (per edge type when heterogeneous) with the highest last-n rank:
//
//   STRICTLY_INCREASING / MONOTONICALLY_INCREASING
//       later edge_start_time ranks higher
//   STRICTLY_DECREASING / MONOTONICALLY_DECREASING
//       earlier edge_start_time ranks higher
//
// Ties (equal timestamps, or int64 magnitudes outside the exact core that collapse
// under double rounding) have arbitrary order; per_v_top_k_select_transform_outgoing_e
// does not provide a secondary key.
//
// The top-k primitive selects the highest strictly-positive bias and treats
// bias 0 as "not selectable".  last_n_time_bias therefore maps an eligible
// timestamp to (0, +inf) with that rank order.  The return type is
// last_n_bias_t (double): float cannot uniquely represent int32 times.
//
// Exactness of last_n_time_bias:
//   int32: all values rank exactly (signed offset lands in [1, 2^32]).
//   int64: t in [-2^52, 2^52 - 1] ranks exactly (~4.5e15; covers unix s/ms/us).
//     Timestamps outside that core use monotonic outer bands that stay strictly
//     positive (large negative t in (0, 1), large positive t above the core) so
//     LAST still returns fanout edges, but ordering among colliding timestamps
//     is arbitrary.  See temporal_sampling_impl.cuh.
//
// LAST does not use caller edge biases and does not support with_replacement.

using last_n_bias_t = double;

template <typename time_stamp_t>
inline constexpr bool last_n_time_bias_is_exact_v =
  std::is_integral_v<time_stamp_t> &&
  (std::numeric_limits<time_stamp_t>::digits <= std::numeric_limits<double>::digits);

// Strictly positive bias for an *eligible* edge.  Callers must still return
// bias 0 when the edge is visited or outside the temporal window.
//
// int32: cast, optional (max - t) for decreasing, 2^31 offset, +1.
// int64: three bands on the (possibly inverted) rank key:
//   core [-2^52, 2^52 - 1]: t + 2^52 + 1 (exact, bias in [1, 2^53])
//   large t >= 2^52:        (t - 2^52 + 1) * 2^54 (monotonic, bias > 2^53)
//   large t <= -2^52 - 1:   (t - INT64_MIN + 1) / 2^63 (monotonic, bias in (0, 1))
template <typename time_stamp_t>
__device__ inline last_n_bias_t last_n_time_bias(
  time_stamp_t edge_time, temporal_sampling_comparison_t temporal_sampling_comparison)
{
  static_assert(std::is_integral_v<time_stamp_t> && !std::is_same_v<time_stamp_t, bool>,
                "LAST neighbor selection requires an integral timestamp type.");

  if constexpr (sizeof(time_stamp_t) >= sizeof(int64_t) && std::is_signed_v<time_stamp_t>) {
    int64_t rank_key = static_cast<int64_t>(edge_time);
    if (is_temporal_decreasing(temporal_sampling_comparison)) {
      rank_key = std::numeric_limits<int64_t>::max() - rank_key;
    }

    constexpr int64_t k_exact_lo = -(int64_t{1} << 52);
    constexpr int64_t k_exact_hi = (int64_t{1} << 52) - 1;

    if (rank_key >= k_exact_lo && rank_key <= k_exact_hi) {
      return static_cast<last_n_bias_t>(rank_key) + static_cast<last_n_bias_t>(uint64_t{1} << 52) +
             last_n_bias_t{1};
    }
    if (rank_key >= (int64_t{1} << 52)) {
      return (static_cast<last_n_bias_t>(rank_key - (int64_t{1} << 52)) + last_n_bias_t{1}) *
             static_cast<last_n_bias_t>(uint64_t{1} << 54);
    }
    auto const idx = static_cast<last_n_bias_t>(rank_key - std::numeric_limits<int64_t>::min());
    return (idx + last_n_bias_t{1}) / static_cast<last_n_bias_t>(uint64_t{1} << 63);
  } else {
    last_n_bias_t rank = static_cast<last_n_bias_t>(edge_time);
    if (is_temporal_decreasing(temporal_sampling_comparison)) {
      rank = static_cast<last_n_bias_t>(std::numeric_limits<time_stamp_t>::max() - edge_time);
    }
    if constexpr (std::is_signed_v<time_stamp_t>) {
      rank += static_cast<last_n_bias_t>(uint64_t{1} << (sizeof(time_stamp_t) * 8 - 1));
    }
    return rank + last_n_bias_t{1};
  }
}

}  // namespace detail
}  // namespace cugraph
