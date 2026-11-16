/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Exclusive integer scans as three plain kernels (tile sums, their scan, the
// tiles plus their offset) that can be graph nodes, unlike a CUB device scan;
// integer sums are exact, so the result equals CUB's ExclusiveSum.

#include "community/detail/leiden/graph.cuh"

#include <cugraph/utilities/error.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cugraph::detail::leiden_engine {

constexpr int kScanBlock     = 256;
constexpr int kScanItems     = 16;
constexpr i64 kScanTile      = static_cast<i64>(kScanBlock) * kScanItems;
constexpr int kScanSumsBlock = 1024;  // S2

template <typename T, typename Out>
struct ScanArray {  // Out(a[i])
  const T* a;
  __host__ __device__ Out operator()(i64 i) const { return static_cast<Out>(a[i]); }
};
struct ScanUsed {  // csize[i] > 0 for i < n2, else 0 (COMPACT flags)
  const int* csize;
  i64 n2;
  __host__ __device__ int operator()(i64 i) const { return (i < n2 && csize[i] > 0) ? 1 : 0; }
};

template <typename In, typename Out>
__global__ void __launch_bounds__(kScanBlock)
  scan_tile_reduce_kernel(In in, i64 items, Out* __restrict__ tile_sums)
{
  using Load   = cub::BlockLoad<Out, kScanBlock, kScanItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Reduce = cub::BlockReduce<Out, kScanBlock>;
  __shared__ union {
    typename Load::TempStorage load;
    typename Reduce::TempStorage reduce;
  } tmp;
  const i64 b0    = static_cast<i64>(blockIdx.x) * kScanTile;
  const i64 valid = items - b0 < kScanTile ? items - b0 : kScanTile;
  auto it         = thrust::make_transform_iterator(thrust::counting_iterator<i64>(b0), in);
  Out x[kScanItems];
  Load(tmp.load).Load(it, x, static_cast<int>(valid), Out(0));
  __syncthreads();
  Out s = 0;
#pragma unroll
  for (int k = 0; k < kScanItems; ++k)
    s += x[k];
  s = Reduce(tmp.reduce).Sum(s);
  if (threadIdx.x == 0) tile_sums[blockIdx.x] = s;
}

// One block: tile_sums[0 .. tiles) <- their exclusive scan.
template <typename Out>
__global__ void __launch_bounds__(kScanSumsBlock)
  scan_block_sums_kernel(Out* __restrict__ tile_sums, i64 tiles)
{
  using Scan = cub::BlockScan<Out, kScanSumsBlock>;
  __shared__ typename Scan::TempStorage tmp;
  Out carry = 0;
  for (i64 c0 = 0; c0 < tiles; c0 += kScanSumsBlock) {
    const i64 i = c0 + threadIdx.x;
    const Out x = i < tiles ? tile_sums[i] : Out(0);
    Out ex, total;
    Scan(tmp).ExclusiveSum(x, ex, total);
    if (i < tiles) tile_sums[i] = carry + ex;
    carry += total;
    __syncthreads();
  }
}

template <typename In, typename Out>
__global__ void __launch_bounds__(kScanBlock)
  scan_tile_apply_kernel(In in, i64 items, const Out* __restrict__ tile_offs, Out* __restrict__ out)
{
  using Load  = cub::BlockLoad<Out, kScanBlock, kScanItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Store = cub::BlockStore<Out, kScanBlock, kScanItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using Scan  = cub::BlockScan<Out, kScanBlock>;
  __shared__ union {
    typename Load::TempStorage load;
    typename Store::TempStorage store;
    typename Scan::TempStorage scan;
  } tmp;
  const i64 b0    = static_cast<i64>(blockIdx.x) * kScanTile;
  const i64 valid = items - b0 < kScanTile ? items - b0 : kScanTile;
  auto it         = thrust::make_transform_iterator(thrust::counting_iterator<i64>(b0), in);
  Out x[kScanItems];
  Load(tmp.load).Load(it, x, static_cast<int>(valid), Out(0));
  __syncthreads();
  Out y[kScanItems];
  Scan(tmp.scan).ExclusiveSum(x, y);
  __syncthreads();
  const Out off = tile_offs[blockIdx.x];
#pragma unroll
  for (int k = 0; k < kScanItems; ++k)
    y[k] += off;
  Store(tmp.store).Store(out + b0, y, static_cast<int>(valid));
}

template <typename Out>
inline std::size_t scan_scratch_bytes(i64 items)
{
  const i64 tiles = items > 0 ? (items + kScanTile - 1) / kScanTile : 1;
  return static_cast<std::size_t>(tiles) * sizeof(Out);
}

// The three calls of a scan of in(0 .. items) into out (scratch: tile sums).
template <typename In, typename Out>
void scan_calls(KernelCall* c, In in, Out* out, i64 items, void* scratch, std::size_t scratch_bytes)
{
  const i64 tiles = items > 0 ? (items + kScanTile - 1) / kScanTile : 1;
  CUGRAPH_EXPECTS(scan_scratch_bytes<Out>(items) <= scratch_bytes,
                  "leiden: scan scratch too small.");
  CUGRAPH_EXPECTS(tiles < (1ll << 31), "leiden: scan too large.");
  Out* sums     = static_cast<Out*>(scratch);
  const bool on = items > 0;
  c[0].set(on,
           scan_tile_reduce_kernel<In, Out>,
           dim3(static_cast<unsigned>(tiles)),
           dim3(kScanBlock),
           0,
           in,
           items,
           sums);
  c[1].set(on, scan_block_sums_kernel<Out>, dim3(1), dim3(kScanSumsBlock), 0, sums, tiles);
  c[2].set(on,
           scan_tile_apply_kernel<In, Out>,
           dim3(static_cast<unsigned>(tiles)),
           dim3(kScanBlock),
           0,
           in,
           items,
           static_cast<const Out*>(sums),
           out);
}

}  // namespace cugraph::detail::leiden_engine
