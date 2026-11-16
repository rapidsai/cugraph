/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// AGGREGATE: contraction of level l by the refined partition (cmap). A1 member
// counts and windows (sums of member degrees); A2 members grouped by coarse id,
// listed by gather class; A3 gathers into hash tables; A4 scan, stats and
// compaction of the holey rows into the arena bottom; A6 P_{l+1} and the level
// metadata. A coarse row holds the exact int64 sums of its members' counted
// entries that agg_drop_entry keeps, per coarse neighbour, as fp32 (exact
// integers >= 1), in hash-slot order (which nothing reads).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/graph.cuh"
#include "community/detail/leiden/kernels_refine.cuh"
#include "community/detail/leiden/kernels_scan.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <initializer_list>
#include <vector>

namespace cugraph::detail::leiden_engine {

// Gather classes by window, and the distinct keys per multi-pass pass.
struct GatherClasses {
  int warp          = 128;   // <= 128: warp, 256-slot table (load <= 0.5)
  int wide          = 256;   // <= 256: warp, 512-slot table (load <= 0.5)
  int block         = 2048;  // <= 2048: block, one pass
  int pass_capacity = 3072;  // multi-pass: distinct keys per pass (0.75)
};
// clist: warp class from the front, block class from the back; wide: own list.
enum : int { kGatherWarp = 0, kGatherBlock = 1, kGatherWide = 2 };

struct AggregateOptions {
  GatherClasses gather;
  ClassThresholds move;          // degree classes reported for level l + 1 (SL2)
  GraphChain* stage1 = nullptr;  // aggregate_begin as one graph replay (null: launches)
};

constexpr u32 kFlagGatherInvariant = 1u << 25;

constexpr int kAggWarpSlots  = 256;  // warp class: 8 warps per block
constexpr int kAggWideSlots  = 512;  // wide-warp class: 4 warps per block
constexpr int kAggWideWarps  = 4;
constexpr int kAggBlockSlots = 4096;
constexpr std::size_t kAggBlockSmem =
  kAggBlockSlots * (sizeof(u64) + sizeof(int));  // 48 KB dynamic
__device__ __forceinline__ int agg_pow2(i64 x)
{
  return x <= 1 ? 1 : 1 << (32 - __clz(static_cast<int>(x - 1)));
}

// Output weight of a coarse entry from its exact int64 sum: fp32 (one rounding)
// or, for the partial rows of a partitioned level 0, the sum itself.
template <typename OW>
__device__ __forceinline__ OW agg_weight(i64 x);
template <>
__device__ __forceinline__ float agg_weight<float>(i64 x)
{
  return __ll2float_rn(x);
}
template <>
__device__ __forceinline__ i64 agg_weight<i64>(i64 x)
{
  return x;
}

// agg_drop_entry: whether the entry (v, u) of a member v of coarse vertex c,
// with cu = cmap[u], is left out of the coarse row of c. Coarse node weights
// are carried separately (khat_next[c] = the summed k_hat of c's members).
// Implemented: cu == c, the self-loops (C, C) of the multiset E of Traag et
// al. 2019, Algorithm A.2 lines 44-47 (AggregateGraph), whose weight k_hat
// already carries: Gilbert & Madduri 2026 (arXiv:2608.01503), Section IV-B
// ("Edges inducing self-loops in the coarse graph are ignored, as cluster
// sizes are tracked separately") and Algorithm 4 line 11 (C[v] != C[u]).
__device__ __forceinline__ bool agg_drop_entry(int cu, int c) { return cu == c; }

// Zeroes [0, n] of the member counts, windows, row lengths and the counters.
__global__ void agg_init_kernel(
  i64 n, int* __restrict__ mcnt, i64* __restrict__ win, i64* __restrict__ cdeg, Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i <= n; i += stride) {
    mcnt[i] = 0;
    win[i]  = 0;
    cdeg[i] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    for (int k = 0; k < 3; ++k)
      ctl->gather_list_count[k] = 0;
    ctl->nnz_next = 0;
    for (int k = 0; k < kNumClasses; ++k)
      ctl->coarse_class_count[k] = 0;
  }
}

__global__ void agg_count_kernel(const i64* __restrict__ indptr,
                                 const int* __restrict__ cmap,
                                 i64 n,
                                 int* __restrict__ mcnt,
                                 i64* __restrict__ win,
                                 Control* ctl)
{
  const i64 nn     = ctl->n_next;
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = cmap[v];
    if (c < 0 || c >= nn) {
      atomicOr(&ctl->flags, kFlagBadLabel);
      continue;
    }
    atomicAdd(&mcnt[c], 1);
    const i64 d = indptr[v + 1] - indptr[v];
    if (d) atomicAdd(reinterpret_cast<u64*>(&win[c]), static_cast<u64>(d));
  }
}

__global__ void agg_scatter_kernel(const int* __restrict__ cmap,
                                   i64 n,
                                   const i64* __restrict__ mstart,
                                   int* mcnt,
                                   int* __restrict__ members,
                                   const i64* __restrict__ woff,
                                   GatherClasses gc,
                                   int* __restrict__ clist,
                                   int* __restrict__ clist_wide,
                                   Control* ctl)
{
  const i64 nn     = ctl->n_next;
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 i = base + (threadIdx.x & (kWarp - 1));
    int cls     = -1;
    if (i < n) {
      const int c = cmap[i];
      if (c >= 0 && c < nn) {
        const i64 pos = mstart[c] + atomicSub(&mcnt[c], 1) - 1;
        members[pos]  = static_cast<int>(i);
      }
      if (i < nn) {
        const i64 win = woff[i + 1] - woff[i];
        cls           = win <= gc.warp ? kGatherWarp : win <= gc.wide ? kGatherWide : kGatherBlock;
      }
    }
    int* cnt     = ctl->gather_list_count;
    const int iw = refine_list_append(cls == kGatherWarp, &cnt[kGatherWarp]);
    const int id = refine_list_append(cls == kGatherWide, &cnt[kGatherWide]);
    const int ib = refine_list_append(cls == kGatherBlock, &cnt[kGatherBlock]);
    if (iw >= 0) clist[iw] = static_cast<int>(i);
    if (id >= 0) clist_wide[id] = static_cast<int>(i);
    if (ib >= 0) clist[n - 1 - ib] = static_cast<int>(i);
  }
}

// Warp per coarse vertex: members flattened into 32-edge chunks; match_any
// pre-aggregates equal coarse neighbours, the group leader inserts.
template <WKind K, int kSlots, int kWarps, typename OW = float>
__global__ void __launch_bounds__(kWarps* kWarp)
  agg_gather_warp_kernel(const i64* __restrict__ indptr,
                         const int* __restrict__ indices,
                         EdgeW<K> w,
                         const i64* __restrict__ khat,
                         const int* __restrict__ cmap,
                         const i64* __restrict__ mstart,
                         const int* __restrict__ members,
                         const i64* __restrict__ woff,
                         const int* __restrict__ list,
                         int list_class,
                         const i64* __restrict__ out_off,
                         int* __restrict__ out_idx,
                         OW* __restrict__ out_w,
                         i64* __restrict__ cdeg,
                         i64* __restrict__ khat_next,
                         Control* ctl)
{
  // int keys (-1 = free, restored by the output scan): 12 B per slot
  __shared__ int tkey[kWarps][kSlots];
  __shared__ i64 tval[kWarps][kSlots];
  __shared__ i64 wsum[kWarps][kWarp];
  const int wib = threadIdx.x / kWarp, lane = threadIdx.x & (kWarp - 1);
  volatile int* vkey = tkey[wib];
  for (int s = lane; s < kSlots; s += kWarp)
    tkey[wib][s] = -1;
  __syncwarp();
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarps + wib;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarps;
  const i64 count  = ctl->gather_list_count[list_class];
  for (i64 i = warp0; i < count; i += nwarps) {
    const int c          = list[i];
    const i64 win        = woff[c + 1] - woff[c];
    int tsize            = agg_pow2(2 * win);
    tsize                = tsize < kWarp ? kWarp : (tsize > kSlots ? kSlots : tsize);
    const unsigned tmask = static_cast<unsigned>(tsize - 1);
    const i64 m0 = mstart[c], m1 = mstart[c + 1];
    const i64 ob = out_off[c];
    i64 ks       = 0;
    for (i64 mb = m0; mb < m1; mb += kWarp) {
      const int nm = static_cast<int>(m1 - mb < kWarp ? m1 - mb : kWarp);
      int v        = -1;
      i64 eb = 0, dg = 0;
      if (lane < nm) {
        v  = members[mb + lane];
        eb = indptr[v];
        dg = indptr[v + 1] - eb;
        ks += khat[v];
      }
      i64 incl = dg;
#pragma unroll
      for (int o = 1; o < kWarp; o <<= 1) {
        const i64 y = __shfl_up_sync(kFullMask, incl, o);
        if (lane >= o) incl += y;
      }
      const i64 total = __shfl_sync(kFullMask, incl, kWarp - 1);
      const i64 excl  = incl - dg;
      for (i64 t0 = 0; t0 < total; t0 += kWarp) {
        const i64 t = t0 + lane;
        int pos     = 0;  // largest member m < nm with excl[m] <= t
#pragma unroll
        for (int step = kWarp / 2; step > 0; step >>= 1) {
          const int cand = pos + step;
          const i64 ex   = __shfl_sync(kFullMask, excl, cand & (kWarp - 1));
          if (cand < nm && ex <= t) pos = cand;
        }
        const i64 ebm = __shfl_sync(kFullMask, eb, pos);
        const i64 exm = __shfl_sync(kFullMask, excl, pos);
        const int vm  = __shfl_sync(kFullMask, v, pos);
        int cu        = -1;
        i64 x         = 0;
        if (t < total) {
          const i64 j = ebm + (t - exm);
          const int u = __ldcs(&indices[j]);
          if (u != vm) {
            x = w.cs(j);
            if (x > 0) {
              cu = cmap[u];
              if (agg_drop_entry(cu, c)) cu = -1;
            }
          }
        }
        if (cu < 0) x = 0;
        const unsigned m = __match_any_sync(kFullMask, cu);
        wsum[wib][lane]  = x;
        __syncwarp();
        if (cu >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += wsum[wib][__ffs(mm) - 1];
          warp_table_add(tkey[wib], tval[wib], tmask, cu, sum);
        }
        __syncwarp();
      }
    }
    ks      = warp_sum(ks);
    int out = 0;
    for (int s0 = 0; s0 < tsize; s0 += kWarp) {
      const int s       = s0 + lane;
      const int kk      = vkey[s];
      const bool f      = kk != -1;
      const unsigned bm = __ballot_sync(kFullMask, f);
      if (f) {
        const i64 o = ob + out + __popc(bm & ((1u << lane) - 1u));
        out_idx[o]  = kk;
        out_w[o]    = agg_weight<OW>(tval[wib][s]);
        vkey[s]     = -1;  // the next coarse vertex starts empty
      }
      out += __popc(bm);
    }
    if (lane == 0) {
      cdeg[c]      = out;
      khat_next[c] = ks;
    }
    __syncwarp();
  }
}

// Inserts into the block table (values zeroed, inserters add). With a capacity
// (multi-pass) a new key is claimed only while fewer than `cap` are, else *ovf
// is set and the pass discarded; at most one claim per thread passes the check
// (< cap + kBlock keys), so *ovf means > cap keys and npass doubling ends.
__device__ __forceinline__ void agg_block_insert(
  int* tk, u64* tv, unsigned tmask, int key, i64 val, int* cnt, int* ovf, int cap)
{
  volatile int* vk  = tk;
  const bool capped = cap < INT_MAX;
  unsigned h        = fmix32(static_cast<u32>(key)) & tmask;
  while (true) {
    const int cur = vk[h];
    if (cur == key) break;
    if (cur == -1) {
      if (capped && atomicAdd(cnt, 0) >= cap) {
        atomicOr(ovf, 1);
        return;
      }
      const int prev = atomicCAS(&tk[h], -1, key);
      if (prev == -1) {
        if (capped) atomicAdd(cnt, 1);
        break;
      }
      if (prev == key) break;
    }
    h = (h + 1) & tmask;
  }
  atomicAdd(&tv[h], static_cast<u64>(val));
}

// Block per coarse vertex: members in batches of kBlock, edges in chunks of
// kBlock (binary search of the owner). Windows <= gc.block run one pass, larger
// ones hash-range passes (table_pass(key, npass) == p), npass doubling.
template <WKind K, typename OW = float>
__global__ void __launch_bounds__(kBlock, 4)
  agg_gather_block_kernel(const i64* __restrict__ indptr,
                          const int* __restrict__ indices,
                          EdgeW<K> w,
                          const i64* __restrict__ khat,
                          const int* __restrict__ cmap,
                          const i64* __restrict__ mstart,
                          const int* __restrict__ members,
                          const i64* __restrict__ woff,
                          const int* __restrict__ clist,
                          i64 n_list,
                          GatherClasses gc,
                          const i64* __restrict__ out_off,
                          int* __restrict__ out_idx,
                          OW* __restrict__ out_w,
                          i64* __restrict__ cdeg,
                          i64* __restrict__ khat_next,
                          Control* ctl,
                          int slots,
                          i64 win_lo,
                          i64 win_hi)
{
  // `slots` entries; this launch takes the windows in [win_lo, win_hi]
  extern __shared__ __align__(16) unsigned char agg_smem[];
  u64* tv      = reinterpret_cast<u64*>(agg_smem);
  int* tk      = reinterpret_cast<int*>(agg_smem + slots * sizeof(u64));
  using Scan   = cub::BlockScan<i64, kBlock>;
  using Reduce = cub::BlockReduce<i64, kBlock>;
  __shared__ union {
    typename Scan::TempStorage scan;
    typename Reduce::TempStorage reduce;
  } tmp;
  __shared__ i64 s_excl[kBlock];
  __shared__ i64 s_eb[kBlock];
  __shared__ int s_v[kBlock];
  __shared__ i64 wsum[kWarpsPerBlock][kWarp];
  __shared__ int s_cnt, s_ovf, s_out;
  const int tid = threadIdx.x, wib = tid / kWarp, lane = tid & (kWarp - 1);
  const i64 count = ctl->gather_list_count[kGatherBlock];
  for (i64 i = blockIdx.x; i < count; i += gridDim.x) {
    const int c   = clist[n_list - 1 - i];
    const i64 win = woff[c + 1] - woff[c];
    if (win < win_lo || win > win_hi) continue;  // block-uniform
    const i64 m0 = mstart[c], m1 = mstart[c + 1];
    i64 ks = 0;
    for (i64 m = m0 + tid; m < m1; m += kBlock)
      ks += khat[members[m]];
    ks = Reduce(tmp.reduce).Sum(ks);
    __syncthreads();
    const bool single    = win <= gc.block;
    const int tsize      = single ? agg_pow2(win + 1) : slots;
    const unsigned tmask = static_cast<unsigned>(tsize - 1);
    const int cap        = single ? INT_MAX : gc.pass_capacity;
    const i64 ob         = out_off[c];
    unsigned npass       = 1;
    while (true) {  // until a full set of passes ran without overflow
      if (tid == 0) s_out = 0;
      bool overflow = false;
      for (unsigned p = 0; p < npass; ++p) {
        for (int s = tid; s < tsize; s += kBlock) {
          tk[s] = -1;
          tv[s] = 0;
        }
        if (tid == 0) {
          s_cnt = 0;
          s_ovf = 0;
        }
        __syncthreads();
        for (i64 mb = m0; mb < m1; mb += kBlock) {
          const int nm = static_cast<int>(m1 - mb < kBlock ? m1 - mb : kBlock);
          int v        = -1;
          i64 eb = 0, dg = 0;
          if (tid < nm) {
            v  = members[mb + tid];
            eb = indptr[v];
            dg = indptr[v + 1] - eb;
          }
          i64 excl, total;
          Scan(tmp.scan).ExclusiveSum(dg, excl, total);
          s_excl[tid] = excl;
          s_eb[tid]   = eb;
          s_v[tid]    = v;
          __syncthreads();
          for (i64 t0 = 0; t0 < total; t0 += kBlock) {
            const i64 t = t0 + tid;
            int cu      = -1;
            i64 x       = 0;
            if (t < total) {
              int lo = 0, hi = nm - 1;  // largest m: excl <= t
              while (lo < hi) {
                const int mid = (lo + hi + 1) >> 1;
                if (s_excl[mid] <= t)
                  lo = mid;
                else
                  hi = mid - 1;
              }
              const i64 j = s_eb[lo] + (t - s_excl[lo]);
              const int u = __ldcs(&indices[j]);
              if (u != s_v[lo]) {
                x = w.cs(j);
                if (x > 0) {
                  cu = cmap[u];
                  if (agg_drop_entry(cu, c) || (npass > 1 && table_pass(cu, npass) != p)) cu = -1;
                }
              }
            }
            if (cu < 0) x = 0;
            const unsigned m = __match_any_sync(kFullMask, cu);
            wsum[wib][lane]  = x;
            __syncwarp();
            if (cu >= 0 && lane == __ffs(m) - 1) {
              i64 sum = 0;
              for (unsigned mm = m; mm; mm &= mm - 1)
                sum += wsum[wib][__ffs(mm) - 1];
              agg_block_insert(tk, tv, tmask, cu, sum, &s_cnt, &s_ovf, cap);
            }
            __syncwarp();
          }
          __syncthreads();  // s_excl / s_eb / s_v are rewritten
        }
        if (s_ovf) {  // block-uniform (read after the barrier)
          overflow = true;
          break;
        }
        for (int s0 = 0; s0 < tsize; s0 += kBlock) {
          const int s       = s0 + tid;
          const bool f      = s < tsize && tk[s] != -1;
          const unsigned bm = __ballot_sync(kFullMask, f);
          int base          = 0;
          if (lane == 0 && bm) base = atomicAdd(&s_out, __popc(bm));
          base = __shfl_sync(kFullMask, base, 0);
          if (f) {
            const i64 o = ob + base + __popc(bm & ((1u << lane) - 1u));
            out_idx[o]  = tk[s];
            out_w[o]    = agg_weight<OW>(static_cast<i64>(tv[s]));
          }
        }
        __syncthreads();
      }
      if (!overflow) break;
      if (npass >= (1u << 31)) {  // cannot happen for cap >= 2
        if (tid == 0) atomicOr(&ctl->flags, kFlagGatherInvariant);
        break;
      }
      npass *= 2;
      __syncthreads();
    }
    if (tid == 0) {
      cdeg[c]      = s_out;
      khat_next[c] = ks;
    }
    __syncthreads();
  }
}

__global__ void agg_stats_kernel(const i64* __restrict__ cdeg,
                                 const i64* __restrict__ indptr_next,
                                 ClassThresholds th,
                                 Control* ctl)
{
  __shared__ i64 s_cls[kWarpsPerBlock][kNumClasses];
  const i64 nn         = ctl->n_next;
  i64 cls[kNumClasses] = {};
  const i64 stride     = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; c < nn; c += stride) {
    cls[degree_class(cdeg[c], th)] += 1;
  }
  const int wib = threadIdx.x / kWarp, lane = threadIdx.x & (kWarp - 1);
#pragma unroll
  for (int k = 0; k < kNumClasses; ++k)
    cls[k] = warp_sum(cls[k]);
  if (lane == 0)
    for (int k = 0; k < kNumClasses; ++k)
      s_cls[wib][k] = cls[k];
  __syncthreads();
  if (threadIdx.x < kNumClasses) {
    i64 t = 0;
    for (int q = 0; q < kWarpsPerBlock; ++q)
      t += s_cls[q][threadIdx.x];
    if (t)
      atomicAdd(reinterpret_cast<u64*>(&ctl->coarse_class_count[threadIdx.x]), static_cast<u64>(t));
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->nnz_next = indptr_next[nn];
}

// Holey rows (window offsets, arena top) -> compacted rows (arena bottom).
template <typename W = float>
__global__ void agg_compact_kernel(const i64* __restrict__ woff,
                                   const i64* __restrict__ cdeg,
                                   const i64* __restrict__ indptr_next,
                                   i64 nn,
                                   const int* __restrict__ h_idx,
                                   const W* __restrict__ h_w,
                                   int* __restrict__ idx,
                                   W* __restrict__ wt)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 base = warp0 * kWarp; base < nn; base += nwarps * kWarp) {
    const i64 c = base + lane;
    i64 src = 0, dst = 0, d = 0;
    if (c < nn) {
      src = woff[c];
      dst = indptr_next[c];
      d   = cdeg[c];
    }
    const int rows = nn - base < kWarp ? static_cast<int>(nn - base) : kWarp;
    for (int q = 0; q < rows; ++q) {
      const i64 sq = __shfl_sync(kFullMask, src, q);
      const i64 dq = __shfl_sync(kFullMask, dst, q);
      const i64 nq = __shfl_sync(kFullMask, d, q);
      for (i64 t = lane; t < nq; t += kWarp) {
        idx[dq + t] = h_idx[sq + t];
        wt[dq + t]  = h_w[sq + t];
      }
    }
  }
}

// A6: P_{l+1}[cmap_v] = P_l[v] (members share a move community, so racing
// stores agree); indptr_{l+1} and k_hat_{l+1} into exact-size arena slots.
__global__ void agg_finish_kernel(i64 n,
                                  i64 nn,
                                  const int* __restrict__ cmap,
                                  const int* __restrict__ P,
                                  int* __restrict__ P_next,
                                  const i64* __restrict__ indptr_scr,
                                  const i64* __restrict__ khat_scr,
                                  i64* __restrict__ indptr_dst,
                                  i64* __restrict__ khat_dst)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  const i64 end    = n > nn + 1 ? n : nn + 1;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < end; i += stride) {
    if (P && i < n) P_next[cmap[i]] = P[i];
    if (i <= nn) indptr_dst[i] = indptr_scr[i];
    if (i < nn) khat_dst[i] = khat_scr[i];
  }
}

__global__ void agg_set_n_next_kernel(Control* ctl, i64 n_next) { ctl->n_next = n_next; }

inline std::size_t agg_bottom_end(std::size_t bottom, std::initializer_list<std::size_t> sizes)
{
  std::size_t b = bottom;
  for (std::size_t x : sizes)
    b = align_up(b) + x;
  return b;
}

template <typename OW = float>
struct AggregatePlan {
  bool holey = false;    // the holey rows have their block (false: not allocated)
  int* h_idx = nullptr;  // holey rows at the arena top (nnz_l entries)
  OW* h_w    = nullptr;
};

struct CoarseLevel {
  i64* indptr    = nullptr;  // [n + 1]
  i64* khat      = nullptr;  // [n]
  int* indices   = nullptr;  // [nnz]
  float* weights = nullptr;
  i64 n          = 0;
  i64 nnz        = 0;
  bool ok        = true;  // false: a block was not allocated (LevelArena::failed)
};

// at every use: a device reset drops the opt-in (CUDA 12)
template <WKind K, typename OW>
inline void agg_block_smem_optin()
{
  RAFT_CUDA_TRY(cudaFuncSetAttribute(agg_gather_block_kernel<K, OW>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(kAggBlockSmem)));
}

// The gather classes over device-resident counts; the block class runs on 1024
// slots (four blocks per SM) below kAggSmallWindow, else 4096. Rows disjoint.
constexpr int kAggSmallSlots  = 1024;
constexpr i64 kAggSmallWindow = kAggSmallSlots - 1;  // nextpow2(win + 1) fits

template <WKind K, typename OW = float>
void gather_calls(KernelCall* c,
                  bool on,
                  const i64* indptr,
                  const int* indices,
                  EdgeW<K> w,
                  const i64* khat,
                  i64 n,
                  const int* cmap,
                  const AggregateBufs& ab,
                  const GatherClasses& gc,
                  const i64* out_off,
                  int* out_idx,
                  OW* out_w,
                  Control* ctl)
{
  c[0].set(
    on,
    agg_gather_warp_kernel<K, kAggWarpSlots, kWarpsPerBlock, OW>,
    dim3(grid_occ(agg_gather_warp_kernel<K, kAggWarpSlots, kWarpsPerBlock, OW>, n, kWarpsPerBlock)),
    dim3(kBlock),
    0,
    indptr,
    indices,
    w,
    khat,
    cmap,
    ab.member_start,
    ab.members,
    ab.window_offset,
    ab.clist,
    static_cast<int>(kGatherWarp),
    out_off,
    out_idx,
    out_w,
    ab.cdeg,
    ab.khat_next,
    ctl);
  // windows > gc.warp are known on the device only: all resident warps run
  c[1].set(on && gc.wide > gc.warp,
           agg_gather_warp_kernel<K, kAggWideSlots, kAggWideWarps, OW>,
           dim3(grid_occ(agg_gather_warp_kernel<K, kAggWideSlots, kAggWideWarps, OW>,
                         n,
                         kAggWideWarps,
                         kAggWideWarps * kWarp)),
           dim3(kAggWideWarps * kWarp),
           0,
           indptr,
           indices,
           w,
           khat,
           cmap,
           ab.member_start,
           ab.members,
           ab.window_offset,
           ab.clist_wide,
           static_cast<int>(kGatherWide),
           out_off,
           out_idx,
           out_w,
           ab.cdeg,
           ab.khat_next,
           ctl);
  agg_block_smem_optin<K, OW>();
  const i64 small_hi           = std::min<i64>(kAggSmallWindow, static_cast<i64>(gc.block));
  const std::size_t small_smem = kAggSmallSlots * (sizeof(u64) + sizeof(int));
  const i64 gs                 = grid_occ(agg_gather_block_kernel<K, OW>, n, 1, kBlock, small_smem);
  const i64 gb = grid_occ(agg_gather_block_kernel<K, OW>, n, 1, kBlock, kAggBlockSmem);
  c[2].set(on,
           agg_gather_block_kernel<K, OW>,
           dim3(static_cast<unsigned>(gs)),
           dim3(kBlock),
           static_cast<unsigned>(kAggSmallSlots * (sizeof(u64) + sizeof(int))),
           indptr,
           indices,
           w,
           khat,
           cmap,
           ab.member_start,
           ab.members,
           ab.window_offset,
           ab.clist,
           n,
           gc,
           out_off,
           out_idx,
           out_w,
           ab.cdeg,
           ab.khat_next,
           ctl,
           kAggSmallSlots,
           i64{0},
           small_hi);
  c[3].set(on,
           agg_gather_block_kernel<K, OW>,
           dim3(static_cast<unsigned>(gb)),
           dim3(kBlock),
           static_cast<unsigned>(kAggBlockSmem),
           indptr,
           indices,
           w,
           khat,
           cmap,
           ab.member_start,
           ab.members,
           ab.window_offset,
           ab.clist,
           n,
           gc,
           out_off,
           out_idx,
           out_w,
           ab.cdeg,
           ab.khat_next,
           ctl,
           kAggBlockSlots,
           small_hi + 1,
           i64{LLONG_MAX});
}

// Stage 1 (A1-A4a): level l is gathered into the arena top (the gathers are
// disabled if its block was not allocated). One call list after `pre` (R9);
// the caller reads n_next, nnz_next and the classes at SL2.
template <WKind K, typename OW = float>
AggregatePlan<OW> aggregate_begin(const i64* indptr,
                                  const int* indices,
                                  EdgeW<K> w,
                                  const i64* khat,
                                  i64 n,
                                  i64 nnz,
                                  const int* cmap,
                                  const AggregateOptions& opt,
                                  const AggregateBufs& ab,
                                  LevelArena& arena,
                                  void* cub,
                                  std::size_t cub_bytes,
                                  Control* ctl,
                                  cudaStream_t s,
                                  std::vector<KernelCall> pre = {})
{
  AggregatePlan<OW> plan;
  const std::size_t ne = static_cast<std::size_t>(nnz);
  arena.reserve_top(align_up(4 * ne) + align_up(sizeof(OW) * ne));  // one block
  plan.h_idx = arena.push_top<int>(ne);
  plan.h_w   = arena.push_top<OW>(ne);
  arena.reserve_top(0);
  plan.holey                = plan.h_idx && plan.h_w;
  std::vector<KernelCall> c = std::move(pre);
  std::size_t k             = c.size();
  c.resize(k + 18);
  const bool on = n > 0;
  c[k++].set(true,
             agg_init_kernel,
             dim3(grid_items(n + 1)),
             dim3(kBlock),
             0,
             n,
             ab.member_count,
             ab.window,
             ab.cdeg,
             ctl);
  c[k++].set(on,
             agg_count_kernel,
             dim3(grid_items(n)),
             dim3(kBlock),
             0,
             indptr,
             cmap,
             n,
             ab.member_count,
             ab.window,
             ctl);
  scan_calls(&c[k], ScanArray<int, i64>{ab.member_count}, ab.member_start, n + 1, cub, cub_bytes);
  k += 3;
  scan_calls(&c[k], ScanArray<i64, i64>{ab.window}, ab.window_offset, n + 1, cub, cub_bytes);
  k += 3;
  c[k++].set(on,
             agg_scatter_kernel,
             dim3(grid_items(n)),
             dim3(kBlock),
             0,
             cmap,
             n,
             ab.member_start,
             ab.member_count,
             ab.members,
             ab.window_offset,
             opt.gather,
             ab.clist,
             ab.clist_wide,
             ctl);
  gather_calls<K, OW>(&c[k],
                      on && plan.holey,
                      indptr,
                      indices,
                      w,
                      khat,
                      n,
                      cmap,
                      ab,
                      opt.gather,
                      ab.window_offset,
                      plan.h_idx,
                      plan.h_w,
                      ctl);
  k += 4;
  scan_calls(&c[k], ScanArray<i64, i64>{ab.cdeg}, ab.indptr_next, n + 1, cub, cub_bytes);
  k += 3;
  c[k++].set(true,
             agg_stats_kernel,
             dim3(grid_items(n)),
             dim3(kBlock),
             0,
             ab.cdeg,
             ab.indptr_next,
             opt.move,
             ctl);
  c.resize(k);
  launch_chain(c, opt.stage1, s);
  return plan;
}

// Stage 2, after SL2: exact metadata and compacted rows (arena bottom, one
// block with room for the next level's cmap), then P_{l+1} if P is given; !ok
// if a block (the holey rows' or this one) was not allocated.
inline CoarseLevel aggregate_finish(i64 n,
                                    const int* cmap,
                                    const AggregatePlan<>& plan,
                                    i64 n_next,
                                    i64 nnz_next,
                                    const AggregateBufs& ab,
                                    LevelArena& arena,
                                    const int* P,
                                    int* P_next,
                                    cudaStream_t s)
{
  CoarseLevel lv;
  lv.n                  = n_next;
  lv.nnz                = nnz_next;
  const std::size_t nn  = static_cast<std::size_t>(n_next);
  const std::size_t ne  = static_cast<std::size_t>(nnz_next);
  const std::size_t end = agg_bottom_end(arena.bottom, {8 * (nn + 1), 8 * nn, 4 * ne, 4 * ne});
  lv.ok                 = plan.holey;
  if (!lv.ok) return lv;
  arena.reserve_bottom(end - align_up(arena.bottom) + align_up(4 * nn) + kAlign);
  lv.indptr  = arena.push_bottom<i64>(nn + 1);
  lv.khat    = arena.push_bottom<i64>(nn);
  lv.indices = arena.push_bottom<int>(ne);
  lv.weights = arena.push_bottom<float>(ne);
  arena.reserve_bottom(0);
  lv.ok = lv.indptr && lv.khat && lv.indices && lv.weights;
  if (!lv.ok) return lv;
  if (n_next > 0) {
    agg_compact_kernel<<<grid_rows(n_next), kBlock, 0, s>>>(ab.window_offset,
                                                            ab.cdeg,
                                                            ab.indptr_next,
                                                            n_next,
                                                            plan.h_idx,
                                                            plan.h_w,
                                                            lv.indices,
                                                            lv.weights);
    RAFT_CHECK_CUDA(s);
  }
  arena.release_top();
  agg_finish_kernel<<<grid_items((n > n_next ? n : n_next) + 1), kBlock, 0, s>>>(
    n, n_next, cmap, P, P_next, ab.indptr_next, ab.khat_next, lv.indptr, lv.khat);
  RAFT_CHECK_CUDA(s);
  return lv;
}

}  // namespace cugraph::detail::leiden_engine
