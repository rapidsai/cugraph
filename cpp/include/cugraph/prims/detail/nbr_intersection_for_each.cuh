/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cugraph {
namespace detail {

constexpr size_t intersection_kernel_block_size = 256;

template <typename edge_t>
__device__ __forceinline__ bool is_edge_unmasked(uint32_t const* edge_mask, edge_t offset)
{
  return edge_mask == nullptr ||
         static_cast<bool>(edge_mask[packed_bool_offset(offset)] & packed_bool_mask(offset));
}

// Thread-per-pair kernel: one thread handles one pair using binary search.
// When use_compact_csr is true, reads from pre-compacted arrays (no mask
// checks) and maps compact offsets back to original CSR via compact_edge_map.
template <bool check_mask,
          bool use_compact_csr,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_low_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask,
  edge_t const* compact_offsets   = nullptr,
  vertex_t const* compact_indices = nullptr,
  edge_t const* compact_edge_map  = nullptr)
{
  auto const tid = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  size_t idx     = tid;

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx = edge_partition.major_offset_from_major_nocheck(p);
    auto q_idx = edge_partition.major_offset_from_major_nocheck(q);

    auto orig_indices = edge_partition.indices();

    bool p_is_short;
    edge_t short_offset, short_degree, long_offset, long_degree;
    vertex_t const* short_indices;
    vertex_t const* long_indices;

    if constexpr (use_compact_csr) {
      edge_t cp = compact_offsets[p_idx], dp = compact_offsets[p_idx + 1] - cp;
      edge_t cq = compact_offsets[q_idx], dq = compact_offsets[q_idx + 1] - cq;
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? cp : cq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? cq : cp;
      long_degree  = p_is_short ? dq : dp;
      short_indices = compact_indices;
      long_indices  = compact_indices;
    } else {
      edge_t op = edge_partition.local_offset(p_idx), dp = edge_partition.local_degree(p_idx);
      edge_t oq = edge_partition.local_offset(q_idx), dq = edge_partition.local_degree(q_idx);
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? op : oq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? oq : op;
      long_degree  = p_is_short ? dq : dp;
      short_indices = orig_indices;
      long_indices  = orig_indices;
    }

    // pq_edge_offset: always in original CSR space
    auto pq_itr = thrust::lower_bound(
      thrust::seq,
      orig_indices + edge_partition.local_offset(p_idx),
      orig_indices + edge_partition.local_offset(p_idx) + edge_partition.local_degree(p_idx),
      q);
    edge_t pq_edge_offset = static_cast<edge_t>(pq_itr - orig_indices);

    // --- ORIGINAL: binary search (commented out for comparison) ---
    // for (edge_t si = 0; si < short_degree; ++si) {
    //   if constexpr (check_mask && !use_compact_csr) {
    //     if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
    //   }
    //   auto r = short_indices[short_offset + si];
    //   edge_t lo = long_offset;
    //   edge_t hi = long_offset + long_degree;
    //   while (lo < hi) {
    //     auto mid = lo + (hi - lo) / 2;
    //     if (long_indices[mid] < r) lo = mid + 1; else hi = mid;
    //   }
    //   if (lo < long_offset + long_degree && long_indices[lo] == r) {
    //     if constexpr (check_mask && !use_compact_csr) {
    //       if (!is_edge_unmasked(edge_mask, lo)) continue;
    //     }
    //     edge_t pr_orig, qr_orig;
    //     if constexpr (use_compact_csr) {
    //       pr_orig = p_is_short ? compact_edge_map[short_offset + si] : compact_edge_map[lo];
    //       qr_orig = p_is_short ? compact_edge_map[lo] : compact_edge_map[short_offset + si];
    //     } else {
    //       pr_orig = p_is_short ? (short_offset + si) : lo;
    //       qr_orig = p_is_short ? lo : (short_offset + si);
    //     }
    //     intersection_op(p, q, r, pq_edge_offset, pr_orig, qr_orig);
    //   }
    // }
    // --- END ORIGINAL ---

    // Merge-walk (two-pointer) intersection
    {
      edge_t si = 0, li = 0;
      while (si < short_degree && li < long_degree) {
        if constexpr (check_mask && !use_compact_csr) {
          if (!is_edge_unmasked(edge_mask, short_offset + si)) { ++si; continue; }
          if (!is_edge_unmasked(edge_mask, long_offset + li)) { ++li; continue; }
        }
        auto sv = short_indices[short_offset + si];
        auto lv = long_indices[long_offset + li];
        if (sv < lv) { ++si; }
        else if (sv > lv) { ++li; }
        else {
          edge_t pr_orig, qr_orig;
          if constexpr (use_compact_csr) {
            // retrieve the original edge indices from the compact edge map
            pr_orig = p_is_short ? compact_edge_map[short_offset + si] : compact_edge_map[long_offset + li];
            qr_orig = p_is_short ? compact_edge_map[long_offset + li] : compact_edge_map[short_offset + si];
          } else {
            pr_orig = p_is_short ? (short_offset + si) : (long_offset + li);
            qr_orig = p_is_short ? (long_offset + li) : (short_offset + si);
          }
          intersection_op(p, q, sv, pq_edge_offset, pr_orig, qr_orig);
          ++si; ++li;
        }
      }
    }

    idx += static_cast<size_t>(gridDim.x) * blockDim.x;
  }
}

// Warp-per-pair kernel: 32 threads cooperate on one pair via parallel binary search.
// Each lane takes a subset of the shorter list and binary searches in the longer list.
template <bool check_mask,
          bool use_compact_csr,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask,
  edge_t const* compact_offsets   = nullptr,
  vertex_t const* compact_indices = nullptr,
  edge_t const* compact_edge_map  = nullptr)
{
  auto const tid     = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  auto const lane_id = static_cast<edge_t>(tid % raft::warp_size());
  size_t idx         = tid / raft::warp_size();

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx = edge_partition.major_offset_from_major_nocheck(p);
    auto q_idx = edge_partition.major_offset_from_major_nocheck(q);

    auto orig_indices = edge_partition.indices();

    edge_t short_offset, short_degree, long_offset, long_degree;
    bool p_is_short;
    vertex_t const* short_indices;
    vertex_t const* long_indices;

    if constexpr (use_compact_csr) {
      edge_t cp = compact_offsets[p_idx], dp = compact_offsets[p_idx + 1] - cp;
      edge_t cq = compact_offsets[q_idx], dq = compact_offsets[q_idx + 1] - cq;
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? cp : cq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? cq : cp;
      long_degree  = p_is_short ? dq : dp;
      short_indices = compact_indices;
      long_indices  = compact_indices;
    } else {
      edge_t op = edge_partition.local_offset(p_idx), dp = edge_partition.local_degree(p_idx);
      edge_t oq = edge_partition.local_offset(q_idx), dq = edge_partition.local_degree(q_idx);
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? op : oq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? oq : op;
      long_degree  = p_is_short ? dq : dp;
      short_indices = orig_indices;
      long_indices  = orig_indices;
    }

    edge_t pq_edge_offset{};
    if (lane_id == 0) {
      auto pq_itr = thrust::lower_bound(
        thrust::seq,
        orig_indices + edge_partition.local_offset(p_idx),
        orig_indices + edge_partition.local_offset(p_idx) + edge_partition.local_degree(p_idx),
        q);
      pq_edge_offset = static_cast<edge_t>(pq_itr - orig_indices);
    }
    pq_edge_offset = __shfl_sync(0xFFFFFFFF, pq_edge_offset, 0);

    for (edge_t si = lane_id; si < short_degree; si += raft::warp_size()) {
      if constexpr (check_mask && !use_compact_csr) {
        if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
      }
      auto r = short_indices[short_offset + si];

      edge_t lo = long_offset;
      edge_t hi = long_offset + long_degree;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (long_indices[mid] < r) lo = mid + 1; else hi = mid;
      }
      if (lo < long_offset + long_degree && long_indices[lo] == r) {
        if constexpr (check_mask && !use_compact_csr) {
          if (!is_edge_unmasked(edge_mask, lo)) continue;
        }
        edge_t pr_orig, qr_orig;
        if constexpr (use_compact_csr) {
          pr_orig = p_is_short ? compact_edge_map[short_offset + si] : compact_edge_map[lo];
          qr_orig = p_is_short ? compact_edge_map[lo] : compact_edge_map[short_offset + si];
        } else {
          pr_orig = p_is_short ? (short_offset + si) : lo;
          qr_orig = p_is_short ? lo : (short_offset + si);
        }
        intersection_op(p, q, r, pq_edge_offset, pr_orig, qr_orig);
      }
    }

    idx += static_cast<size_t>(gridDim.x) * (blockDim.x / raft::warp_size());
  }
}


// EXPERIMENTAL BUT UNUSED. Warp-per-pair merge-path kernel: O(S+L)
// two-pointer merge walk. Each lane handles a stripe of the merged sequence.
template <bool check_mask,
          bool use_compact_csr,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_mid_degree_merge(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask,
  edge_t const* compact_offsets   = nullptr,
  vertex_t const* compact_indices = nullptr,
  edge_t const* compact_edge_map  = nullptr)
{
  auto const tid     = threadIdx.x + static_cast<size_t>(blockIdx.x) * blockDim.x;
  auto const lane_id = static_cast<edge_t>(tid % raft::warp_size());
  size_t idx         = tid / raft::warp_size();

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx = edge_partition.major_offset_from_major_nocheck(p);
    auto q_idx = edge_partition.major_offset_from_major_nocheck(q);

    auto orig_indices = edge_partition.indices();

    edge_t p_off, p_deg, q_off, q_deg;
    vertex_t const* p_indices;
    vertex_t const* q_indices;

    if constexpr (use_compact_csr) {
      p_off = compact_offsets[p_idx];
      p_deg = compact_offsets[p_idx + 1] - p_off;
      q_off = compact_offsets[q_idx];
      q_deg = compact_offsets[q_idx + 1] - q_off;
      p_indices = compact_indices;
      q_indices = compact_indices;
    } else {
      p_off = edge_partition.local_offset(p_idx);
      p_deg = edge_partition.local_degree(p_idx);
      q_off = edge_partition.local_offset(q_idx);
      q_deg = edge_partition.local_degree(q_idx);
      p_indices = orig_indices;
      q_indices = orig_indices;
    }

    edge_t pq_edge_offset{};
    if (lane_id == 0) {
      auto pq_itr = thrust::lower_bound(
        thrust::seq,
        orig_indices + edge_partition.local_offset(p_idx),
        orig_indices + edge_partition.local_offset(p_idx) + edge_partition.local_degree(p_idx),
        q);
      pq_edge_offset = static_cast<edge_t>(pq_itr - orig_indices);
    }
    pq_edge_offset = __shfl_sync(0xFFFFFFFF, pq_edge_offset, 0);

    // Serial merge walk: lane 0 does the two-pointer walk, other lanes idle.
    // This is O(p_deg + q_deg) work per pair.
    if (lane_id == 0) {
      edge_t pi = 0, qi = 0;
      while (pi < p_deg && qi < q_deg) {
        if constexpr (check_mask && !use_compact_csr) {
          while (pi < p_deg && !is_edge_unmasked(edge_mask, p_off + pi)) ++pi;
          while (qi < q_deg && !is_edge_unmasked(edge_mask, q_off + qi)) ++qi;
          if (pi >= p_deg || qi >= q_deg) break;
        }

        vertex_t pv = p_indices[p_off + pi];
        vertex_t qv = q_indices[q_off + qi];

        if (pv < qv) {
          ++pi;
        } else if (pv > qv) {
          ++qi;
        } else {
          edge_t pr_orig, qr_orig;
          if constexpr (use_compact_csr) {
            pr_orig = compact_edge_map[p_off + pi];
            qr_orig = compact_edge_map[q_off + qi];
          } else {
            pr_orig = p_off + pi;
            qr_orig = q_off + qi;
          }
          intersection_op(p, q, pv, pq_edge_offset, pr_orig, qr_orig);
          ++pi;
          ++qi;
        }
      }
    }

    idx += static_cast<size_t>(gridDim.x) * (blockDim.x / raft::warp_size());
  }
}

// Block-per-pair kernel: an entire block cooperates on one pair via parallel binary search.
template <bool check_mask,
          bool use_compact_csr,
          typename vertex_t,
          typename edge_t,
          typename VertexPairIterator,
          typename IntersectionOp>
__global__ static void intersection_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, false> edge_partition,
  VertexPairIterator vertex_pair_first,
  size_t const* pair_index_first,
  size_t num_pairs,
  IntersectionOp intersection_op,
  uint32_t const* edge_mask,
  edge_t const* compact_offsets   = nullptr,
  vertex_t const* compact_indices = nullptr,
  edge_t const* compact_edge_map  = nullptr)
{
  size_t idx = static_cast<size_t>(blockIdx.x);

  while (idx < num_pairs) {
    auto i    = pair_index_first[idx];
    auto pair = *(vertex_pair_first + i);
    auto p    = cuda::std::get<0>(pair);
    auto q    = cuda::std::get<1>(pair);

    auto p_idx = edge_partition.major_offset_from_major_nocheck(p);
    auto q_idx = edge_partition.major_offset_from_major_nocheck(q);

    auto orig_indices = edge_partition.indices();

    edge_t short_offset, short_degree, long_offset, long_degree;
    bool p_is_short;
    vertex_t const* short_indices;
    vertex_t const* long_indices;

    if constexpr (use_compact_csr) {
      edge_t cp = compact_offsets[p_idx], dp = compact_offsets[p_idx + 1] - cp;
      edge_t cq = compact_offsets[q_idx], dq = compact_offsets[q_idx + 1] - cq;
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? cp : cq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? cq : cp;
      long_degree  = p_is_short ? dq : dp;
      short_indices = compact_indices;
      long_indices  = compact_indices;
    } else {
      edge_t op = edge_partition.local_offset(p_idx), dp = edge_partition.local_degree(p_idx);
      edge_t oq = edge_partition.local_offset(q_idx), dq = edge_partition.local_degree(q_idx);
      p_is_short   = (dp <= dq);
      short_offset = p_is_short ? op : oq;
      short_degree = p_is_short ? dp : dq;
      long_offset  = p_is_short ? oq : op;
      long_degree  = p_is_short ? dq : dp;
      short_indices = orig_indices;
      long_indices  = orig_indices;
    }

    __shared__ edge_t shared_pq_offset;
    if (threadIdx.x == 0) {
      auto pq_itr = thrust::lower_bound(
        thrust::seq,
        orig_indices + edge_partition.local_offset(p_idx),
        orig_indices + edge_partition.local_offset(p_idx) + edge_partition.local_degree(p_idx),
        q);
      shared_pq_offset = static_cast<edge_t>(pq_itr - orig_indices);
    }
    __syncthreads();
    edge_t pq_edge_offset = shared_pq_offset;

    for (edge_t si = static_cast<edge_t>(threadIdx.x); si < short_degree;
         si += static_cast<edge_t>(blockDim.x)) {
      if constexpr (check_mask && !use_compact_csr) {
        if (!is_edge_unmasked(edge_mask, short_offset + si)) continue;
      }
      auto r = short_indices[short_offset + si];

      edge_t lo = long_offset;
      edge_t hi = long_offset + long_degree;
      while (lo < hi) {
        auto mid = lo + (hi - lo) / 2;
        if (long_indices[mid] < r) lo = mid + 1; else hi = mid;
      }
      if (lo < long_offset + long_degree && long_indices[lo] == r) {
        if constexpr (check_mask && !use_compact_csr) {
          if (!is_edge_unmasked(edge_mask, lo)) continue;
        }
        edge_t pr_orig, qr_orig;
        if constexpr (use_compact_csr) {
          pr_orig = p_is_short ? compact_edge_map[short_offset + si] : compact_edge_map[lo];
          qr_orig = p_is_short ? compact_edge_map[lo] : compact_edge_map[short_offset + si];
        } else {
          pr_orig = p_is_short ? (short_offset + si) : lo;
          qr_orig = p_is_short ? lo : (short_offset + si);
        }
        intersection_op(p, q, r, pq_edge_offset, pr_orig, qr_orig);
      }
    }

    idx += static_cast<size_t>(gridDim.x);
  }
}

}  // namespace detail
}  // namespace cugraph
