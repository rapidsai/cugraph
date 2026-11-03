/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// PLEIDEN_R refinement: a spanning forest inside each move community S (every
// vertex picks its heaviest-modularity edge), then the longest valid prefix of
// every tree in ord order is committed. R1 ext_v and the R-test U_v; R2 the
// best U neighbour hn_v; R3 / R4 chases to the roots; R5 members by root; R7
// inner_v; R8 the prefix; R9 coarse ids. Decisions read exact int64 values and
// orders are bijective hashes (no ties), so neither interleaving, launch
// geometry nor the commit class changes a result.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/graph.cuh"
#include "community/detail/leiden/kernels_scan.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda_runtime.h>

#include <vector>

namespace cugraph::detail::leiden_engine {

constexpr u32 kFlagForestInvariant = 1u << 24;

struct TreeClasses {
  int thread = 8;   // <= 8 members: thread per tree (sorting network)
  int warp   = 32;  // <= 32 members: warp per tree (bitonic sort)
};  // larger trees: segmented sort + chunked block scan
// tree_list: thread class from the front, warp class from the back.
enum : int { kTreeThread = 0, kTreeWarp = 1, kTreeLarge = 2 };
constexpr int kTinyTree = 8;  // register arrays of the thread class

constexpr int kCommitBlock = 256;               // threads of the large commit kernel
constexpr u64 kOrdPad      = (1ull << 34) - 1;  // sorts after every ord (<= 2^33)

__device__ __forceinline__ i64 refine_ord(u64 ctx_order, int v, bool flipped)
{
  return order_key(order_f(ctx_order, v), flipped);
}

// Slot of a warp-aggregated append (-1 if !pred); every lane must call it.
__device__ __forceinline__ int refine_list_append(bool pred, int* counter)
{
  const unsigned m = __ballot_sync(kFullMask, pred);
  if (!m) return -1;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(counter, __popc(m));
  base = __shfl_sync(kFullMask, base, leader);
  return pred ? base + __popc(m & ((1u << lane) - 1u)) : -1;
}

// Row kernels (R1, R2, R7): a warp walks up to 32 consecutive rows.
__device__ __forceinline__ int refine_rows_per_warp(i64 n, i64 nwarps)
{
  const i64 r = (n + nwarps - 1) / nwarps;
  return r < 1 ? 1 : (r > kWarp ? kWarp : static_cast<int>(r));
}

// refine_vertex_ok: whether v, in move community S, may be merged by the
// refinement (the vertex condition of the refinement). ext_v = W(v, S - v)
// (exact int64), K_S = the summed node weight of S (v included), k_v = v's node
// weight, lam = lambda_hat (pen(K, make_mult(k, lam)) = lambda_hat * k * K).
// Implemented: ext_v >= pen(K_S - k_v, make_mult(k_v, lam)), the set R of
// Traag et al. 2019, Algorithm A.2 line 34 (MergeNodesSubset: E(v, S - v) >=
// gamma ||v|| (||S|| - ||v||), for modularity ||v|| = k_v and gamma rescaled
// by 2m, Appendix A eq. (A5)), as NetworKit ParallelLeiden.cpp:435-436
// (R-Set Condition).
__device__ __forceinline__ bool refine_vertex_ok(i64 ext_v, i64 K_S, i64 k_v, double lam)
{
  return ext_v >= pen(K_S - k_v, make_mult(k_v, lam));
}

// R1: ext_v and the vertex test U_v (refine_vertex_ok).
template <WKind K>
__global__ void refine_ext_u_kernel(const i64* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    EdgeW<K> w,
                                    const i64* __restrict__ khat,
                                    const int* __restrict__ S,
                                    const i64* __restrict__ KS,
                                    i64 n,
                                    i64 C_host,
                                    const i64* __restrict__ C_dev,
                                    double lam,
                                    i64* __restrict__ ext,
                                    unsigned char* __restrict__ U,
                                    int* __restrict__ SU,
                                    Control* ctl)
{
  const i64 C      = C_dev ? *C_dev : C_host;
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const int rpw    = refine_rows_per_warp(n, nwarps);
  for (i64 base = warp0 * rpw; base < n; base += nwarps * rpw) {
    const i64 v    = base + lane;
    const bool own = lane < rpw && v < n;
    i64 b = 0, e = 0;
    int s = -1;
    if (own) {
      b = indptr[v];
      e = indptr[v + 1];
      s = S[v];
      if (s < 0 || s >= C) {
        atomicOr(&ctl->flags, kFlagBadLabel);
        s = -1;
      }
    }
    const int rows = n - base < rpw ? static_cast<int>(n - base) : rpw;
    i64 mine       = 0;
    for (int q = 0; q < rows; ++q) {
      const int sq = __shfl_sync(kFullMask, s, q);
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const i64 vq = base + q;
      i64 acc      = 0;
      if (sq >= 0) {  // warp-uniform
        for (i64 j = bq + lane; j < eq; j += kWarp) {
          const int u = __ldcs(&indices[j]);
          if (u == vq) continue;
          const i64 x = w.cs(j);
          if (x > 0 && S[u] == sq) acc += x;
        }
      }
      acc = warp_sum(acc);
      if (lane == q) mine = acc;
    }
    if (own) {
      ext[v]    = mine;
      bool u_ok = false;
      if (s >= 0) {
        const i64 kv = khat[v];
        u_ok         = refine_vertex_ok(mine, KS[s], kv, lam);
      }
      U[v]  = u_ok ? 1 : 0;
      SU[v] = u_ok ? s : -1;  // R2's neighbour test in one gather
    }
  }
}

// R2: hn_v = argmax of (key, -u), key = w_vu - pen_v(k_u) >= 0, over counted U
// neighbours u in S_v (SU = U_u ? S_u : -1); v if U_v fails or there is none.
template <WKind K>
__global__ void refine_choose_kernel(const i64* __restrict__ indptr,
                                     const int* __restrict__ indices,
                                     EdgeW<K> w,
                                     const i64* __restrict__ khat,
                                     const int* __restrict__ S,
                                     const unsigned char* __restrict__ U,
                                     const int* __restrict__ SU,
                                     i64 n,
                                     double lam,
                                     int* __restrict__ hn,
                                     int* __restrict__ root)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const int rpw    = refine_rows_per_warp(n, nwarps);
  for (i64 base = warp0 * rpw; base < n; base += nwarps * rpw) {
    const i64 v    = base + lane;
    const bool own = lane < rpw && v < n;
    i64 b = 0, e = 0;
    int s = -1;  // -1: v fails the R-test (no edge choice)
    Mult mv{0ull, 0};
    if (own && U[v]) {
      b  = indptr[v];
      e  = indptr[v + 1];
      s  = S[v];
      mv = make_mult(khat[v], lam);
    }
    const int rows = n - base < rpw ? static_cast<int>(n - base) : rpw;
    int mine       = -1;
    for (int q = 0; q < rows; ++q) {
      const int sq = __shfl_sync(kFullMask, s, q);
      if (sq < 0) continue;  // warp-uniform
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const Mult mq{__shfl_sync(kFullMask, mv.M, q), __shfl_sync(kFullMask, mv.sh, q)};
      const i64 vq = base + q;
      i64 bk       = -1;       // best key (valid keys are >= 0)
      int bv       = INT_MAX;  // its vertex
      for (i64 j = bq + lane; j < eq; j += kWarp) {
        const int u = __ldcs(&indices[j]);
        if (u == vq || SU[u] != sq) continue;  // S_u == S_v, U_u
        const i64 x = w.cs(j);
        if (x <= 0) continue;  // not counted
        const i64 key = x - pen(khat[u], mq);
        if (key > bk || (key == bk && u < bv)) {
          bk = key;
          bv = u;
        }
      }
#pragma unroll
      for (int o = kWarp / 2; o > 0; o >>= 1) {
        const i64 ok = __shfl_xor_sync(kFullMask, bk, o);
        const int ou = __shfl_xor_sync(kFullMask, bv, o);
        if (ok > bk || (ok == bk && ou < bv)) {
          bk = ok;
          bv = ou;
        }
      }
      if (lane == q && bk >= 0) mine = bv;
    }
    if (own) {
      const int h = mine >= 0 ? mine : static_cast<int>(v);
      hn[v]       = h;
      root[v]     = h == v ? static_cast<int>(v) : -1;
    }
  }
}

// R7: inner_v = sum of counted w_vu with root_u = root_v and ord(u) < ord(v);
// rf = root | flipped << 31 makes the test one gather.
template <WKind K>
__global__ void refine_inner_kernel(const i64* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    EdgeW<K> w,
                                    const int* __restrict__ root,
                                    const unsigned char* __restrict__ flipped,
                                    const int* __restrict__ rf,
                                    const int* __restrict__ tree_offset,
                                    i64 n,
                                    u64 ctx_order,
                                    i64* __restrict__ inner)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const int rpw    = refine_rows_per_warp(n, nwarps);
  for (i64 base = warp0 * rpw; base < n; base += nwarps * rpw) {
    const i64 v    = base + lane;
    const bool own = lane < rpw && v < n;
    i64 b = 0, e = 0, ov = 0;
    int rv = -1;  // -1: singleton tree (inner = 0)
    if (own) {
      const int r0 = root[v];
      if (tree_offset[r0 + 1] - tree_offset[r0] > 1) {
        rv = r0;
        b  = indptr[v];
        e  = indptr[v + 1];
        ov = refine_ord(ctx_order, static_cast<int>(v), flipped[v] != 0);
      }
    }
    const int rows = n - base < rpw ? static_cast<int>(n - base) : rpw;
    i64 mine       = 0;
    for (int q = 0; q < rows; ++q) {
      const int rq = __shfl_sync(kFullMask, rv, q);
      if (rq < 0) continue;  // warp-uniform
      const i64 bq = __shfl_sync(kFullMask, b, q);
      const i64 eq = __shfl_sync(kFullMask, e, q);
      const i64 oq = __shfl_sync(kFullMask, ov, q);
      const i64 vq = base + q;
      i64 acc      = 0;
      for (i64 j = bq + lane; j < eq; j += kWarp) {
        const int u = __ldcs(&indices[j]);
        if (u == vq) continue;
        const int ru = rf[u];
        if ((ru & 0x7fffffff) != rq) continue;
        if (refine_ord(ctx_order, u, ru < 0) >= oq) continue;
        const i64 x = w.cs(j);
        if (x > 0) acc += x;
      }
      acc = warp_sum(acc);
      if (lane == q) mine = acc;
    }
    if (own) inner[v] = mine;
  }
}

// R3: for hn_v != v, chase towards lower f; if the chase moved, root_v and
// root_now <- now. Roots are never read and -1 never written, and a chase
// endpoint never moves, so every interleaving gives the same roots.
__global__ void forest_chase1_kernel(const int* __restrict__ hn,
                                     i64 n,
                                     u64 ctx_order,
                                     int* __restrict__ root)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int v = static_cast<int>(i);
    int nxt     = hn[v];
    if (nxt == v) continue;
    int now  = v;
    u32 fnow = order_f(ctx_order, now), fnxt = order_f(ctx_order, nxt);
    while (fnow > fnxt) {
      now  = nxt;
      fnow = fnxt;
      nxt  = hn[now];
      fnxt = order_f(ctx_order, nxt);
    }
    if (now != v) {
      root[v]   = now;
      root[now] = now;
    }
  }
}

// R4a: flipped <- (root == -1) (R4b's snapshot); zeroes the tree counters.
__global__ void forest_flip_kernel(const int* __restrict__ root,
                                   i64 n,
                                   unsigned char* __restrict__ flipped,
                                   int* __restrict__ tree_count,
                                   Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v <= n; v += stride) {
    tree_count[v] = 0;
    if (v < n) flipped[v] = root[v] == -1 ? 1 : 0;
  }
  if (blockIdx.x == 0 && threadIdx.x < 3) ctl->tree_list_count[threadIdx.x] = 0;
}

// R4b + R5 count: a flipped v chases while the current vertex is flipped and
// ord decreases, to a rooted vertex (final root); tree_count[root_v] += 1.
__global__ void forest_chase2_kernel(const int* __restrict__ hn,
                                     const unsigned char* __restrict__ flipped,
                                     i64 n,
                                     u64 ctx_order,
                                     int* root,
                                     int* __restrict__ tree_count,
                                     Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const int v = static_cast<int>(i);
    int rv;
    if (flipped[v]) {
      int now = v, nxt = hn[v];
      i64 onow  = refine_ord(ctx_order, now, true);
      bool fnow = true;
      while (fnow) {
        const bool fnxt = flipped[nxt] != 0;
        const i64 onxt  = refine_ord(ctx_order, nxt, fnxt);
        if (onow <= onxt) break;
        now  = nxt;
        onow = onxt;
        fnow = fnxt;
        nxt  = hn[now];
      }
      if (fnow) {  // invariant 1 violated: ended at an unrooted vertex
        atomicOr(&ctl->flags, kFlagForestInvariant);
        rv = v;
      } else {
        rv = root[now];
      }
      root[v] = rv;
    } else {
      rv = root[v];
    }
    atomicAdd(&tree_count[rv], 1);
  }
}

// R5: members grouped by root, r_v <- v (R8 overwrites the committed ones),
// trees of >= 2 members listed by class (large trees: segmented-sort input).
__global__ void tree_scatter_kernel(const int* __restrict__ root,
                                    const unsigned char* __restrict__ flipped,
                                    const int* __restrict__ tree_offset,
                                    i64 n,
                                    u64 ctx_order,
                                    TreeClasses tc,
                                    int* tree_count,
                                    int* __restrict__ members,
                                    int* __restrict__ r,
                                    int* __restrict__ tree_list,
                                    int* __restrict__ large_list,
                                    i64* __restrict__ seg_keys,
                                    int* __restrict__ seg_vals,
                                    int* __restrict__ rf,
                                    Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + (threadIdx.x & (kWarp - 1));
    int cls     = -1;
    if (v < n) {
      const int rv   = root[v];
      const int o    = tree_offset[rv];
      const int size = tree_offset[rv + 1] - o;
      const int pos  = o + atomicSub(&tree_count[rv], 1) - 1;
      members[pos]   = static_cast<int>(v);
      rf[v]          = rv | (flipped[v] ? static_cast<int>(0x80000000u) : 0);
      r[v]           = static_cast<int>(v);
      if (size > tc.warp) {
        seg_keys[pos] = refine_ord(ctx_order, static_cast<int>(v), flipped[v] != 0);
        seg_vals[pos] = static_cast<int>(v);
      }
      if (rv == v && size >= 2)
        cls = size <= tc.thread ? kTreeThread : size <= tc.warp ? kTreeWarp : kTreeLarge;
    }
    int* cnt     = ctl->tree_list_count;
    const int it = refine_list_append(cls == kTreeThread, &cnt[kTreeThread]);
    const int iw = refine_list_append(cls == kTreeWarp, &cnt[kTreeWarp]);
    const int il = refine_list_append(cls == kTreeLarge, &cnt[kTreeLarge]);
    if (it >= 0) tree_list[it] = static_cast<int>(v);
    if (iw >= 0) tree_list[n - 1 - iw] = static_cast<int>(v);
    if (il >= 0) large_list[il] = static_cast<int>(v);
  }
}

// R8: the members of a tree are visited in sorted order; member x at position
// j >= 1 joins the community formed by the members at positions < j (the
// prefix) iff refine_commit_ok holds. kpre = summed node weight of the prefix,
// extpre = W(prefix, S - prefix) (exact; the exclusive prefix sum of
// ext - 2 inner, with inner_x = W(x, members before x) and inner_root = 0),
// ks = K_S, kx = k_x, ix = inner_x = W(x, prefix), lam = lambda_hat.
// Implemented: ix >= pen(kpre, make_mult(kx, lam)) (Delta H(x -> prefix) >= 0,
// Traag et al. 2019, Algorithm A.2 line 38) and extpre >= pen(ks - kpre,
// make_mult(kpre, lam)) (the prefix C is in T: E(C, S - C) >= gamma ||C||
// (||S|| - ||C||), line 37), the two checks of the proof of Lemma 1 of pLeiden
// (Gilbert & Madduri, IPDPS 2026, arXiv:2608.01503: "it verifies w'(v, Cx) >=
// 0 and w'(Cx, B[Cx]) >= 0"); NetworKit ParallelLeiden.cpp:457 and 462-464.
__device__ __forceinline__ bool refine_commit_ok(
  i64 kpre, i64 extpre, i64 ks, i64 kx, i64 ix, double lam)
{
  const bool gain = ix >= pen(kpre, make_mult(kx, lam));
  const bool well = extpre >= pen(ks - kpre, make_mult(kpre, lam));
  return gain & well;
}

template <typename T>
__device__ __forceinline__ T refine_warp_scan(T x)
{
  const int lane = threadIdx.x & (kWarp - 1);
  T incl         = x;
#pragma unroll
  for (int o = 1; o < kWarp; o <<= 1) {
    const T y = __shfl_up_sync(kFullMask, incl, o);
    if (lane >= o) incl += y;
  }
  return incl - x;
}

// Ascending bitonic sort of (key, val) across the lanes (pads carry kOrdPad).
__device__ __forceinline__ void refine_warp_sort(u64& key, int& val)
{
  const int lane = threadIdx.x & (kWarp - 1);
#pragma unroll
  for (int k = 2; k <= kWarp; k <<= 1) {
#pragma unroll
    for (int j = k >> 1; j > 0; j >>= 1) {
      const u64 ok        = __shfl_xor_sync(kFullMask, key, j);
      const int ov        = __shfl_xor_sync(kFullMask, val, j);
      const bool keep_min = ((lane & j) == 0) == ((lane & k) == 0);
      if (keep_min ? ok < key : ok > key) {
        key = ok;
        val = ov;
      }
    }
  }
}

// Thread per tree of <= 8 members: odd-even sort in registers, prefix walk.
__global__ void tree_commit_thread_kernel(const int* __restrict__ tree_list,
                                          const int* __restrict__ tree_offset,
                                          const int* __restrict__ members,
                                          const unsigned char* __restrict__ flipped,
                                          const i64* __restrict__ khat,
                                          const i64* __restrict__ ext,
                                          const i64* __restrict__ inner,
                                          const int* __restrict__ S,
                                          const i64* __restrict__ KS,
                                          double lam,
                                          u64 ctx_order,
                                          int* __restrict__ r,
                                          Control* ctl)
{
  const i64 count  = ctl->tree_list_count[kTreeThread];
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
    const int t    = tree_list[i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    u64 key[kTinyTree];
    int x[kTinyTree];
#pragma unroll
    for (int k = 0; k < kTinyTree; ++k) {
      key[k] = kOrdPad;
      x[k]   = -1;
      if (k < size) {
        x[k]   = members[o + k];
        key[k] = static_cast<u64>(refine_ord(ctx_order, x[k], flipped[x[k]] != 0));
      }
    }
#pragma unroll
    for (int rnd = 0; rnd < kTinyTree; ++rnd) {
#pragma unroll
      for (int k = rnd & 1; k + 1 < kTinyTree; k += 2) {
        if (key[k] > key[k + 1]) {
          const u64 tk = key[k];
          key[k]       = key[k + 1];
          key[k + 1]   = tk;
          const int tx = x[k];
          x[k]         = x[k + 1];
          x[k + 1]     = tx;
        }
      }
    }
    const int x0 = x[0];
    const i64 ks = KS[S[x0]];
    i64 kpre     = khat[x0];
    i64 extpre   = ext[x0] - 2 * inner[x0];
    bool open    = true;
#pragma unroll
    for (int k = 1; k < kTinyTree; ++k) {
      if (open && k < size) {
        const int xk = x[k];
        const i64 kx = khat[xk], ix = inner[xk];
        if (refine_commit_ok(kpre, extpre, ks, kx, ix, lam)) {
          r[xk] = x0;
          kpre += kx;
          extpre += ext[xk] - 2 * ix;
        } else {
          open = false;
        }
      }
    }
    if (x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
  }
}

__global__ void tree_commit_warp_kernel(const int* __restrict__ tree_list,
                                        i64 n,
                                        const int* __restrict__ tree_offset,
                                        const int* __restrict__ members,
                                        const unsigned char* __restrict__ flipped,
                                        const i64* __restrict__ khat,
                                        const i64* __restrict__ ext,
                                        const i64* __restrict__ inner,
                                        const int* __restrict__ S,
                                        const i64* __restrict__ KS,
                                        double lam,
                                        u64 ctx_order,
                                        int* __restrict__ r,
                                        Control* ctl)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 count  = ctl->tree_list_count[kTreeWarp];
  for (i64 i = warp0; i < count; i += nwarps) {
    const int t    = tree_list[n - 1 - i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    u64 key        = kOrdPad;
    int x          = -1;
    if (lane < size) {
      x   = members[o + lane];
      key = static_cast<u64>(refine_ord(ctx_order, x, flipped[x] != 0));
    }
    refine_warp_sort(key, x);
    const bool valid = lane < size;
    i64 kx = 0, dx = 0, ix = 0;
    if (valid) {
      kx = khat[x];
      ix = inner[x];
      dx = ext[x] - 2 * ix;
    }
    const i64 kpre      = refine_warp_scan(kx);
    const i64 extpre    = refine_warp_scan(dx);
    const int x0        = __shfl_sync(kFullMask, x, 0);
    const i64 ks        = KS[S[x0]];
    const bool ok       = !valid || lane == 0 || refine_commit_ok(kpre, extpre, ks, kx, ix, lam);
    const unsigned fail = __ballot_sync(kFullMask, !ok);
    const int p         = fail ? __ffs(fail) - 1 : size;
    if (valid && lane >= 1 && lane < p) r[x] = x0;
    if (lane == 0 && x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
  }
}

// Block per large tree (members sorted by ord): chunks with carried prefixes.
__global__ void __launch_bounds__(kCommitBlock)
  tree_commit_large_kernel(const int* __restrict__ large_list,
                           const int* __restrict__ tree_offset,
                           const int* __restrict__ sorted_vals,
                           const i64* __restrict__ khat,
                           const i64* __restrict__ ext,
                           const i64* __restrict__ inner,
                           const int* __restrict__ S,
                           const i64* __restrict__ KS,
                           double lam,
                           int* __restrict__ r,
                           Control* ctl)
{
  using Scan = cub::BlockScan<i64, kCommitBlock>;
  __shared__ typename Scan::TempStorage tmp;
  __shared__ int s_fail;
  const int tid   = threadIdx.x;
  const i64 count = ctl->tree_list_count[kTreeLarge];
  for (i64 i = blockIdx.x; i < count; i += gridDim.x) {
    const int t    = large_list[i];
    const int o    = tree_offset[t];
    const int size = tree_offset[t + 1] - o;
    const int x0   = sorted_vals[o];
    const i64 ks   = KS[S[x0]];
    i64 carry_k = 0, carry_d = 0;
    for (int c0 = 0; c0 < size; c0 += kCommitBlock) {
      const int j      = c0 + tid;
      const bool valid = j < size;
      int x            = -1;
      i64 kx = 0, dx = 0, ix = 0;
      if (valid) {
        x  = sorted_vals[o + j];
        kx = khat[x];
        ix = inner[x];
        dx = ext[x] - 2 * ix;
      }
      if (tid == 0) s_fail = INT_MAX;
      i64 ek, ed, tk, td;
      Scan(tmp).ExclusiveSum(kx, ek, tk);
      __syncthreads();
      Scan(tmp).ExclusiveSum(dx, ed, td);
      const i64 kpre = carry_k + ek, extpre = carry_d + ed;
      if (valid && j >= 1 && !refine_commit_ok(kpre, extpre, ks, kx, ix, lam))
        atomicMin(&s_fail, j);
      __syncthreads();
      const int f = s_fail;
      if (valid && j >= 1 && j < f) r[x] = x0;
      __syncthreads();
      if (f != INT_MAX) break;  // block-uniform
      carry_k += tk;
      carry_d += td;
    }
    if (tid == 0 && x0 != t) atomicOr(&ctl->flags, kFlagForestInvariant);
    __syncthreads();
  }
}

// Segments of the large trees for DeviceSegmentedSort.
struct LargeSegmentBegin {
  const int* list;
  const int* count;
  const int* offset;
  __host__ __device__ int operator()(int i) const { return i < *count ? offset[list[i]] : 0; }
};
struct LargeSegmentEnd {
  const int* list;
  const int* count;
  const int* offset;
  __host__ __device__ int operator()(int i) const { return i < *count ? offset[list[i] + 1] : 0; }
};

__global__ void host_flags_kernel(const int* __restrict__ r, i64 n, int* __restrict__ flags)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v <= n; v += stride)
    flags[v] = v < n && r[v] == v ? 1 : 0;
}

// R9: cmap_v = cid[r_v]; n_{l+1} = cid[n].
__global__ void cmap_kernel(const int* __restrict__ r,
                            const int* __restrict__ cid,
                            i64 n,
                            int* __restrict__ cmap,
                            Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    cmap[v] = cid[r[v]];
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_next = cid[n];
}

struct RefineParams {
  double lam    = 0.0;  // lambda_hat = gamma / 2m_hat
  u64 ctx_order = 0;    // ctx(seed64, ORDER, it, l, 0, 0)
  TreeClasses trees;
};

inline void refine_check_cub(std::size_t need, std::size_t have, const char* what)
{
  CUGRAPH_EXPECTS(need <= have,
                  "leiden: internal error: CUB temp storage for %s exceeds the carved region.",
                  what);
}

// PLEIDEN_R of a level with move partition S (ids < C) and volumes KS: cmap
// and ctl->n_next without a host sync. `pre` (COMPACT) and R1-R8 run as one
// call list, R9's calls go to `tail` (AGGREGATE stage 1) or launch.
template <WKind K>
void run_refine(const i64* indptr,
                const int* indices,
                EdgeW<K> w,
                const i64* khat,
                const int* S,
                const i64* KS,
                i64 n,
                i64 C,
                const RefineParams& rp,
                const RefineBufs& rb,
                int* cmap,
                void* cub,
                std::size_t cub_bytes,
                Control* ctl,
                cudaStream_t s,
                const i64* C_dev              = nullptr,
                GraphChain* graph             = nullptr,
                std::vector<KernelCall>* tail = nullptr,
                std::vector<KernelCall> pre   = {})
{
  if (n == 0) {
    launch_chain(pre, nullptr, s);
    RAFT_CUDA_TRY(cudaMemsetAsync(&ctl->n_next, 0, sizeof(i64), s));
    return;
  }
  const unsigned gi         = grid_items(n);
  const TreeClasses& tc     = rp.trees;
  std::vector<KernelCall> c = std::move(pre);
  std::size_t k             = c.size();
  c.resize(k + 16);
  // rb.cid (free until R9) carries SU from R1 to R2 and rf from R5 to R7
  c[k++].set(true,
             refine_ext_u_kernel<K>,
             dim3(grid_occ(refine_ext_u_kernel<K>, n, kWarpsPerBlock)),
             dim3(kBlock),
             0,
             indptr,
             indices,
             w,
             khat,
             S,
             KS,
             n,
             C,
             C_dev,
             rp.lam,
             rb.ext,
             rb.U,
             rb.cid,
             ctl);
  c[k++].set(true,
             refine_choose_kernel<K>,
             dim3(grid_occ(refine_choose_kernel<K>, n, kWarpsPerBlock)),
             dim3(kBlock),
             0,
             indptr,
             indices,
             w,
             khat,
             S,
             rb.U,
             rb.cid,
             n,
             rp.lam,
             rb.hn,
             rb.root);
  c[k++].set(
    true, forest_chase1_kernel, dim3(gi), dim3(kBlock), 0, rb.hn, n, rp.ctx_order, rb.root);
  c[k++].set(true,
             forest_flip_kernel,
             dim3(grid_items(n + 1)),
             dim3(kBlock),
             0,
             rb.root,
             n,
             rb.flipped,
             rb.tree_count,
             ctl);
  c[k++].set(true,
             forest_chase2_kernel,
             dim3(gi),
             dim3(kBlock),
             0,
             rb.hn,
             rb.flipped,
             n,
             rp.ctx_order,
             rb.root,
             rb.tree_count,
             ctl);
  scan_calls(&c[k], ScanArray<int, int>{rb.tree_count}, rb.tree_offset, n + 1, cub, cub_bytes);
  k += 3;
  c[k++].set(true,
             tree_scatter_kernel,
             dim3(gi),
             dim3(kBlock),
             0,
             rb.root,
             rb.flipped,
             rb.tree_offset,
             n,
             rp.ctx_order,
             rp.trees,
             rb.tree_count,
             rb.members,
             rb.r,
             rb.tree_list,
             rb.large_list,
             rb.sort_keys_a,
             rb.sort_vals_a,
             rb.cid,
             ctl);
  c[k++].set(true,
             refine_inner_kernel<K>,
             dim3(grid_occ(refine_inner_kernel<K>, n, kWarpsPerBlock)),
             dim3(kBlock),
             0,
             indptr,
             indices,
             w,
             rb.root,
             rb.flipped,
             rb.cid,
             rb.tree_offset,
             n,
             rp.ctx_order,
             rb.inner);
  // R8 classes over device-resident counts (impossible classes disabled)
  c[k++].set(n >= 2,
             tree_commit_thread_kernel,
             dim3(grid_occ(tree_commit_thread_kernel, n / 2 + 1, kBlock)),
             dim3(kBlock),
             0,
             rb.tree_list,
             rb.tree_offset,
             rb.members,
             rb.flipped,
             khat,
             rb.ext,
             rb.inner,
             S,
             KS,
             rp.lam,
             rp.ctx_order,
             rb.r,
             ctl);
  c[k++].set(n > tc.thread,
             tree_commit_warp_kernel,
             dim3(grid_rows(n / (tc.thread + 1) + 1)),
             dim3(kBlock),
             0,
             rb.tree_list,
             n,
             rb.tree_offset,
             rb.members,
             rb.flipped,
             khat,
             rb.ext,
             rb.inner,
             S,
             KS,
             rp.lam,
             rp.ctx_order,
             rb.r,
             ctl);
  c.resize(static_cast<std::size_t>(k));
  launch_chain(c, graph, s);
  if (n > tc.warp) {  // trees larger than a warp can exist
    const int max_large = static_cast<int>(n / (static_cast<i64>(tc.warp) + 1));
    const int* cnt      = &ctl->tree_list_count[kTreeLarge];
    auto begin          = thrust::make_transform_iterator(
      thrust::counting_iterator<int>(0), LargeSegmentBegin{rb.large_list, cnt, rb.tree_offset});
    auto end         = thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                               LargeSegmentEnd{rb.large_list, cnt, rb.tree_offset});
    const int items  = cub_items(n, "large-tree sort");
    std::size_t need = 0;
    RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(nullptr,
                                                      need,
                                                      rb.sort_keys_a,
                                                      rb.sort_keys_b,
                                                      rb.sort_vals_a,
                                                      rb.sort_vals_b,
                                                      items,
                                                      max_large,
                                                      begin,
                                                      end,
                                                      s));
    refine_check_cub(need, cub_bytes, "the large-tree sort");
    std::size_t tb = cub_bytes;
    RAFT_CUDA_TRY(cub::DeviceSegmentedSort::SortPairs(cub,
                                                      tb,
                                                      rb.sort_keys_a,
                                                      rb.sort_keys_b,
                                                      rb.sort_vals_a,
                                                      rb.sort_vals_b,
                                                      items,
                                                      max_large,
                                                      begin,
                                                      end,
                                                      s));
    tree_commit_large_kernel<<<grid_for(max_large, 1, kCommitBlock), kCommitBlock, 0, s>>>(
      rb.large_list,
      rb.tree_offset,
      rb.sort_vals_b,
      khat,
      rb.ext,
      rb.inner,
      S,
      KS,
      rp.lam,
      rb.r,
      ctl);
    RAFT_CHECK_CUDA(s);
  }
  // R9 (the host flags reuse tree_count)
  std::vector<KernelCall> r9(5);
  r9[0].set(
    true, host_flags_kernel, dim3(grid_items(n + 1)), dim3(kBlock), 0, rb.r, n, rb.tree_count);
  scan_calls(&r9[1], ScanArray<int, int>{rb.tree_count}, rb.cid, n + 1, cub, cub_bytes);
  r9[4].set(true, cmap_kernel, dim3(gi), dim3(kBlock), 0, rb.r, rb.cid, n, cmap, ctl);
  if (tail) {
    tail->insert(tail->end(), r9.begin(), r9.end());
  } else {
    launch_chain(r9, nullptr, s);
  }
}

}  // namespace cugraph::detail::leiden_engine
