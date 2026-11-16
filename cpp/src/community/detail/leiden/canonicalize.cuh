/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// I1c: canonicalize. Rare path, run only when the input check I1 flags a row that is not strictly
// increasing by column (parallel edges, or unsorted rows).
//
// Every row is sorted by column and every run of parallel edges (u, v) is replaced by one entry
// whose weight is the fp64 sum of the run. The run is summed sequentially in ascending order of
// the weight values, so the canonical CSR is a function of the multiset of (row, column, weight)
// entries only: it does not depend on the order in which the entries are stored. This is what
// makes the multi-GPU path, whose stored order depends on the edge partitioning, bitwise identical
// to the single-GPU path for multigraphs too.
//
// Rows are processed in chunks of whole rows with at most kCanonChunk entries (or one longer row),
// so every CUB call has < 2^31 items. Within a chunk, a stable LSD radix sort of the weight keys
// followed by a stable LSD radix sort of the keys (local row << 30 | column) orders every row by
// column with each run of duplicates in ascending weight order; the chunking is layout only.
//
// Output: int64 offsets (n + 1), int32 indices and fp64 weights (unweighted input: every stored
// entry has the weight 1, so a run sums to its multiplicity).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform_reduce.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cugraph::detail::leiden_engine {

constexpr i64 kCanonChunk = 1ll << 26;
constexpr int kColumnBits = 30;  // n < 2^30

inline i64 canon_chunk_cap(i64 nnz, i64 max_row_nnz)
{
  return std::max<i64>(std::min<i64>(nnz, kCanonChunk), max_row_nnz);
}

struct CanonBufs {
  u64* keys             = nullptr;  // [cap] (local row << 30 | column) of chunk entry i
  u64* sort_a           = nullptr;  // [cap] weight keys, radix sort double buffer
  u64* sort_b           = nullptr;
  int* vals_a           = nullptr;  // [cap] entry index within the chunk
  int* vals_b           = nullptr;
  i64* loff             = nullptr;  // [n + 2] distinct columns per row of a chunk
  i64* loff2            = nullptr;  // [n + 2] chunk-local output offsets
  i64* base             = nullptr;  // [1] output entries written so far
  void* cub             = nullptr;
  std::size_t cub_bytes = 0;

  static std::size_t cub_temp(i64 n, i64 cap)
  {
    std::size_t b = 0, best = 0;
    if (cap > 0) {
      u64* k = nullptr;
      int* v = nullptr;
      cub::DoubleBuffer<u64> keys(k, k);
      cub::DoubleBuffer<int> vals(v, v);
      RAFT_CUDA_TRY(
        cub::DeviceRadixSort::SortPairs(nullptr, b, keys, vals, cub_items(cap, "canonical sort")));
      best = b;
    }
    b        = 0;
    i64* off = nullptr;
    RAFT_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(nullptr, b, off, off, cub_items(n + 2, "canonical scan")));
    return std::max(best, b);
  }

  void carve(Carver& c, i64 n, i64 cap)
  {
    keys      = c.take<u64>(cap);
    sort_a    = c.take<u64>(cap);
    sort_b    = c.take<u64>(cap);
    vals_a    = c.take<int>(cap);
    vals_b    = c.take<int>(cap);
    loff      = c.take<i64>(n + 2);
    loff2     = c.take<i64>(n + 2);
    base      = c.take<i64>(1);
    cub_bytes = cub_temp(n, cap);
    cub       = c.bytes(cub_bytes);
  }
};

// Device scratch of run_canonicalize for a graph with n vertices, nnz stored entries and at most
// max_row_nnz entries per row.
inline std::size_t canonicalize_scratch_bytes(i64 n, i64 nnz, i64 max_row_nnz)
{
  CUGRAPH_EXPECTS(n >= 0 && nnz >= 0 && max_row_nnz >= 0,
                  "leiden: n, nnz and max_row_nnz must be >= 0.");
  CUGRAPH_EXPECTS(max_row_nnz < (1ll << 31),
                  "leiden: vertices with >= 2^31 stored edges are not supported.");
  Carver c;
  CanonBufs b;
  b.carve(c, n, canon_chunk_cap(nnz, max_row_nnz));
  return c.used();
}

// Warp per row: keys[i] = (local row << 30 | column) and vals[i] = i for chunk entry i; with
// weights, wkeys[i] = IEEE bits of the weight (monotone in the value for weights >= 0, which the
// input check I1 has established before canonicalize runs).
template <typename IP, typename IX, typename WI>
__global__ void canon_keys_kernel(const IP* __restrict__ indptr,
                                  const IX* __restrict__ indices,
                                  const WI* __restrict__ data,
                                  i64 r0,
                                  i64 r1,
                                  i64 e0,
                                  u64* __restrict__ keys,
                                  u64* __restrict__ wkeys,
                                  int* __restrict__ vals)
{
  const int lane         = threadIdx.x & (kWarp - 1);
  const i64 warp0        = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps       = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  constexpr u64 kColMask = (1ull << kColumnBits) - 1ull;
  for (i64 v = r0 + warp0; v < r1; v += nwarps) {
    const i64 b = static_cast<i64>(indptr[v]);
    const i64 e = static_cast<i64>(indptr[v + 1]);
    for (i64 j = b + lane; j < e; j += kWarp) {
      keys[j - e0] =
        (static_cast<u64>(v - r0) << kColumnBits) | (static_cast<u64>(indices[j]) & kColMask);
      if (data) wkeys[j - e0] = weight_bits(data[j]);
      vals[j - e0] = static_cast<int>(j - e0);
    }
  }
}

// out[i] = keys[order[i]]: the row / column keys in ascending weight order.
__global__ static void canon_gather_keys_kernel(const u64* __restrict__ keys,
                                                const int* __restrict__ order,
                                                i64 m,
                                                u64* __restrict__ out)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < m; i += stride)
    out[i] = keys[order[i]];
}

// Thread per row of the chunk: loff[i] = distinct columns of row r0 + i.
template <typename IP>
__global__ void canon_count_kernel(const IP* __restrict__ indptr,
                                   i64 r0,
                                   i64 rows,
                                   i64 e0,
                                   const u64* __restrict__ keys,
                                   i64* __restrict__ loff)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < rows; i += stride) {
    const i64 b = static_cast<i64>(indptr[r0 + i]) - e0;
    const i64 e = static_cast<i64>(indptr[r0 + i + 1]) - e0;
    i64 k       = 0;
    for (i64 j = b; j < e; ++j)
      k += (j == b || keys[j] != keys[j - 1]);
    loff[i] = k;
    if (i == rows - 1) loff[rows] = 0;
  }
}

// Thread per row: write the canonical row at base + loff2[i]; every run of duplicates is summed
// in fp64, sequentially, in the sorted (ascending weight) order.
template <typename IP, typename WI>
__global__ void canon_write_kernel(const IP* __restrict__ indptr,
                                   i64 r0,
                                   i64 rows,
                                   i64 e0,
                                   const u64* __restrict__ keys,
                                   const int* __restrict__ vals,
                                   const WI* __restrict__ data,
                                   const i64* __restrict__ loff2,
                                   const i64* __restrict__ base,
                                   i64* __restrict__ indptr_out,
                                   int* __restrict__ idx_out,
                                   double* __restrict__ data_out)
{
  constexpr u64 kColMask = (1ull << kColumnBits) - 1ull;
  const i64 b0           = base[0];
  const i64 stride       = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < rows; i += stride) {
    const i64 b        = static_cast<i64>(indptr[r0 + i]) - e0;
    const i64 e        = static_cast<i64>(indptr[r0 + i + 1]) - e0;
    i64 out            = b0 + loff2[i];
    indptr_out[r0 + i] = out;
    i64 j              = b;
    while (j < e) {
      const u64 key = keys[j];
      const i64 src = e0 + vals[j];
      double sum    = data ? static_cast<double>(data[src]) : 1.0;
      for (++j; j < e && keys[j] == key; ++j) {
        const i64 sj = e0 + vals[j];
        sum          = __dadd_rn(sum, data ? static_cast<double>(data[sj]) : 1.0);
      }
      idx_out[out]  = static_cast<int>(key & kColMask);
      data_out[out] = sum;
      ++out;
    }
  }
}

__global__ static void canon_advance_kernel(
  const i64* __restrict__ loff2, i64 rows, i64* base, i64* indptr_out, i64 n, int last)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    base[0] += loff2[rows];
    if (last) indptr_out[n] = base[0];
  }
}

inline int bit_length(u64 x)
{
  int b = 0;
  while (x) {
    ++b;
    x >>= 1;
  }
  return b;
}

template <typename IP>
struct CanonRowLength {
  const IP* indptr;
  __host__ __device__ i64 operator()(i64 v) const
  {
    return static_cast<i64>(indptr[v + 1]) - static_cast<i64>(indptr[v]);
  }
};

template <typename IP>
struct CanonWiden {
  __host__ __device__ i64 operator()(IP x) const { return static_cast<i64>(x); }
};

// Longest row of the CSR (offsets are on the device).
template <typename IP>
i64 canon_max_row_nnz(const IP* indptr, i64 n, cudaStream_t s)
{
  if (n == 0) return 0;
  return thrust::transform_reduce(rmm::exec_policy(s),
                                  thrust::make_counting_iterator<i64>(0),
                                  thrust::make_counting_iterator<i64>(n),
                                  CanonRowLength<IP>{indptr},
                                  i64{0},
                                  thrust::maximum<i64>{});
}

// Canonicalizes the CSR (indptr, indices, data) of n vertices and nnz stored entries (data ==
// nullptr: unit weights) into (indptr_out [n + 1], idx_out [nnz], data_out [nnz]) and returns the
// canonical number of entries. `scratch` must hold canonicalize_scratch_bytes(n, nnz,
// max_row_nnz) bytes. The input must have passed I1 with valid structure and finite, non-negative
// weights. Synchronizes the stream (chunk boundaries and the output size are read back).
template <typename IP, typename IX, typename WI>
i64 run_canonicalize(const IP* indptr,
                     const IX* indices,
                     const WI* data,
                     i64 n,
                     i64 nnz,
                     i64 max_row_nnz,
                     i64* indptr_out,
                     int* idx_out,
                     double* data_out,
                     void* scratch,
                     std::size_t scratch_bytes,
                     cudaStream_t s)
{
  CUGRAPH_EXPECTS(n >= 0 && n < kMaxVertices,
                  "leiden: the number of vertices must be in [0, 2^30).");
  const i64 cap = canon_chunk_cap(nnz, max_row_nnz);
  Carver c{static_cast<char*>(scratch), 0};
  CanonBufs b;
  b.carve(c, n, cap);
  CUGRAPH_EXPECTS(c.used() <= scratch_bytes, "leiden: canonicalize scratch too small.");
  RAFT_CUDA_TRY(cudaMemsetAsync(b.base, 0, sizeof(i64), s));
  if (n == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(indptr_out, 0, sizeof(i64), s));
    RAFT_CUDA_TRY(cudaStreamSynchronize(s));
    return 0;
  }
  auto const widened = thrust::make_transform_iterator(indptr, CanonWiden<IP>{});
  i64 r0             = 0;
  while (r0 < n) {
    // Greedy chunk of whole rows [r0, r1) with <= cap entries (at least one row): r1 is the last
    // row boundary with indptr[r1] - indptr[r0] <= cap.
    IP h_e0{};
    RAFT_CUDA_TRY(cudaMemcpyAsync(&h_e0, indptr + r0, sizeof(IP), cudaMemcpyDeviceToHost, s));
    RAFT_CUDA_TRY(cudaStreamSynchronize(s));
    const i64 e0 = static_cast<i64>(h_e0);
    auto const it =
      thrust::upper_bound(rmm::exec_policy(s), widened + r0 + 1, widened + n + 1, e0 + cap);
    const i64 r1 = std::max<i64>(r0 + 1, static_cast<i64>(it - widened) - 1);
    IP h_e1{};
    RAFT_CUDA_TRY(cudaMemcpyAsync(&h_e1, indptr + r1, sizeof(IP), cudaMemcpyDeviceToHost, s));
    RAFT_CUDA_TRY(cudaStreamSynchronize(s));
    const i64 rows = r1 - r0, m = static_cast<i64>(h_e1) - e0;

    if (m > 0) {
      canon_keys_kernel<IP, IX, WI><<<grid_rows(rows), kBlock, 0, s>>>(
        indptr, indices, data, r0, r1, e0, b.keys, b.sort_a, b.vals_a);
      RAFT_CHECK_CUDA(s);
    }
    cub::DoubleBuffer<int> vals(b.vals_a, b.vals_b);
    const u64* row_keys = b.keys;
    if (m > 1) {
      cub::DoubleBuffer<u64> keys(b.keys, b.sort_b);
      if (data) {
        // Stable sort of the entries by weight, then the row / column keys in that order.
        cub::DoubleBuffer<u64> wkeys(b.sort_a, b.sort_b);
        std::size_t tb = b.cub_bytes;
        RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(b.cub,
                                                      tb,
                                                      wkeys,
                                                      vals,
                                                      cub_items(m, "canonical weight sort"),
                                                      0,
                                                      std::is_same_v<WI, float> ? 32 : 64,
                                                      s));
        canon_gather_keys_kernel<<<grid_items(m), kBlock, 0, s>>>(
          b.keys, vals.Current(), m, wkeys.Alternate());
        RAFT_CHECK_CUDA(s);
        keys = cub::DoubleBuffer<u64>(wkeys.Alternate(), wkeys.Current());
      }
      // Stable sort by (row, column): every run of duplicates stays in ascending weight order.
      const int end_bit = kColumnBits + bit_length(static_cast<u64>(rows - 1));
      std::size_t tb    = b.cub_bytes;
      RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
        b.cub, tb, keys, vals, cub_items(m, "canonical row sort"), 0, end_bit, s));
      row_keys = keys.Current();
    }
    const unsigned g = grid_items(rows);
    canon_count_kernel<IP><<<g, kBlock, 0, s>>>(indptr, r0, rows, e0, row_keys, b.loff);
    RAFT_CHECK_CUDA(s);
    std::size_t tb = b.cub_bytes;
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      b.cub, tb, b.loff, b.loff2, cub_items(rows + 1, "canonical offsets"), s));
    canon_write_kernel<IP, WI><<<g, kBlock, 0, s>>>(indptr,
                                                    r0,
                                                    rows,
                                                    e0,
                                                    row_keys,
                                                    vals.Current(),
                                                    data,
                                                    b.loff2,
                                                    b.base,
                                                    indptr_out,
                                                    idx_out,
                                                    data_out);
    RAFT_CHECK_CUDA(s);
    canon_advance_kernel<<<1, 1, 0, s>>>(b.loff2, rows, b.base, indptr_out, n, r1 == n ? 1 : 0);
    RAFT_CHECK_CUDA(s);
    r0 = r1;
  }
  i64 out = 0;
  RAFT_CUDA_TRY(cudaMemcpyAsync(&out, b.base, sizeof(i64), cudaMemcpyDeviceToHost, s));
  RAFT_CUDA_TRY(cudaStreamSynchronize(s));
  return out;
}

}  // namespace cugraph::detail::leiden_engine
