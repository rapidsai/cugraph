/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Finalize: F1 cc_hook (ECL-CC hooking of intra-community counted edges), F2
// cc_flatten (root = component minimum, then COMPACT), F3 quality_rows (exact
// L_hat, K_hat), F4 quality_tree (TREE1024 sum and Q), F5 size-ordered labels.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <climits>
#include <utility>

namespace cugraph::detail::leiden_engine {

// CC split (ECL-CC, Jaiganesh & Burtscher 2018). parent[x] <= x always, so
// chains decrease to a root; concurrent shortcuts only store another ancestor
// (benign races; volatile reads).
__device__ __forceinline__ int cc_find(int x, volatile int* parent)
{
  int curr = parent[x];
  if (curr != x) {
    int next, prev = x;
    while (curr > (next = parent[curr])) {
      parent[prev] = next;
      prev         = curr;
      curr         = next;
    }
  }
  return curr;
}

__device__ __forceinline__ void cc_union(int a, int b, int* parent)
{
  volatile int* vp = parent;
  int ra = cc_find(a, vp), rb = cc_find(b, vp);
  bool repeat;
  do {
    repeat = false;
    if (ra != rb) {
      if (ra < rb) {
        const int ret = atomicCAS(&parent[rb], rb, ra);
        if (ret != rb) {
          rb     = ret;
          repeat = true;
        }
      } else {
        const int ret = atomicCAS(&parent[ra], ra, rb);
        if (ret != ra) {
          ra     = ret;
          repeat = true;
        }
      }
    }
  } while (repeat);
}

__global__ void cc_init_kernel(i64 n, int* __restrict__ parent)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    parent[v] = static_cast<int>(v);
}

// Warp per row: every counted intra-community edge is hooked once, from its
// larger endpoint. Per 32-entry chunk the minimum mn of the representatives
// takes every other one r by CAS (mn < r keeps parent[x] <= x), else union.
template <WKind K>
__global__ void cc_hook_kernel(const i64* __restrict__ indptr,
                               const int* __restrict__ indices,
                               EdgeW<K> w,
                               const int* __restrict__ P,
                               i64 n,
                               int* parent)
{
  volatile int* vp = parent;
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 v = warp0; v < n; v += nwarps) {
    const i64 b = indptr[v], e = indptr[v + 1];
    const int pv = P[v];
    for (i64 j0 = b; j0 < e; j0 += kWarp) {
      const i64 j = j0 + lane;
      int u       = -1;
      if (j < e) {
        const int x = __ldcs(&indices[j]);
        if (x < v && P[x] == pv && w.cs(j) > 0) u = x;
      }
      const unsigned any = __ballot_sync(kFullMask, u >= 0);
      if (!any) continue;  // warp-uniform
      const int vrep = cc_find(static_cast<int>(v), vp);
      const int rep  = u >= 0 ? cc_find(u, vp) : INT_MAX;
      int mn         = rep < vrep ? rep : vrep;
#pragma unroll
      for (int o = kWarp / 2; o > 0; o >>= 1) {
        const int y = __shfl_xor_sync(kFullMask, mn, o);
        mn          = y < mn ? y : mn;
      }
      const unsigned grp = __match_any_sync(kFullMask, rep);
      if (u >= 0 && rep != mn && lane == __ffs(grp) - 1 && atomicCAS(&parent[rep], rep, mn) != rep)
        cc_union(u, mn, parent);
      if (lane == 0 && vrep != mn && atomicCAS(&parent[vrep], vrep, mn) != vrep)
        cc_union(static_cast<int>(v), mn, parent);
    }
  }
}

// flags[v] = 1 iff v is a root (its component's minimum id), flags[n] = 0.
__global__ void cc_flatten_kernel(i64 n, int* parent, int* __restrict__ flags)
{
  volatile int* vp = parent;
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    int curr      = vp[v];
    const int old = curr;
    int next;
    while (curr > (next = vp[curr]))
      curr = next;
    if (curr != old) vp[v] = curr;
    flags[v] = curr == v ? 1 : 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) flags[n] = 0;
}

__global__ void cc_relabel_kernel(i64 n,
                                  const int* __restrict__ parent,
                                  const int* __restrict__ rank,
                                  int* __restrict__ P,
                                  Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    P[v] = rank[parent[v]];
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_components = rank[n];
}

// CC_SPLIT: P <- COMPACT(component minimum).
template <WKind K>
void run_cc_split(const i64* indptr,
                  const int* indices,
                  EdgeW<K> w,
                  int* P,
                  i64 n,
                  const FinalBufs& fb,
                  void* cub,
                  std::size_t cub_bytes,
                  Control* ctl,
                  cudaStream_t s)
{
  if (n == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(&ctl->n_components, 0, sizeof(i64), s));
    return;
  }
  const unsigned gi = grid_items(n);
  cc_init_kernel<<<gi, kBlock, 0, s>>>(n, fb.parent);
  RAFT_CHECK_CUDA(s);
  cc_hook_kernel<K><<<grid_rows(n), kBlock, 0, s>>>(indptr, indices, w, P, n, fb.parent);
  RAFT_CHECK_CUDA(s);
  cc_flatten_kernel<<<gi, kBlock, 0, s>>>(n, fb.parent, fb.flags);
  RAFT_CHECK_CUDA(s);
  std::size_t tb = cub_bytes;
  RAFT_CUDA_TRY(
    cub::DeviceScan::ExclusiveSum(cub, tb, fb.flags, fb.rank, cub_items(n + 1, "cc compact"), s));
  cc_relabel_kernel<<<gi, kBlock, 0, s>>>(n, fb.parent, fb.rank, P, ctl);
  RAFT_CHECK_CUDA(s);
}

// Community volumes are warp-aggregated into a block-privatised histogram for
// C <= 1024, else into global memory; integer sums are exact and order-free.
constexpr int kSmemCommunities = 1024;

// Warp per union row x = r n + v of the single-copy graph: L_hat[r] +=
// intra-community counted weight, K[P[x]] += k_hat_v, repmask[P[x]] |= 1 << r.
template <WKind K, bool kSmem>
__global__ void quality_rows_kernel(const i64* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    EdgeW<K> w,
                                    const i64* __restrict__ khat,
                                    const int* __restrict__ P,
                                    i64 n,
                                    int R,
                                    i64 C,
                                    i64* __restrict__ Kc,
                                    int* __restrict__ repmask,
                                    Control* ctl)
{
  extern __shared__ i64 sK[];
  __shared__ i64 s_l[kWarpsPerBlock][kMaxReplicas];
  __shared__ i64 s_k[kWarpsPerBlock][kWarp];
  __shared__ int s_r[kWarpsPerBlock][kWarp];
  if constexpr (kSmem) {
    for (i64 c = threadIdx.x; c < C; c += blockDim.x)
      sK[c] = 0;
    __syncthreads();
  }
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  const i64 warp0        = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + wib;
  const i64 nwarps       = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 rows         = static_cast<i64>(R) * n;
  i64 lacc[kMaxReplicas] = {0, 0, 0, 0};
  u32 bad                = 0;
  for (i64 base = warp0; base < rows; base += kWarp * nwarps) {
    int my_c = -1, my_r = 0;
    i64 my_k = 0;
    for (int i = 0; i < kWarp; ++i) {
      const i64 x = base + static_cast<i64>(i) * nwarps;
      if (x >= rows) break;  // warp-uniform
      const int r      = static_cast<int>(x / n);
      const i64 v      = x - static_cast<i64>(r) * n;
      const i64 off    = static_cast<i64>(r) * n;
      const int pv     = P[x];
      const bool valid = pv >= 0 && pv < C;
      const i64 b = indptr[v], e = indptr[v + 1];
      i64 l = 0;
      for (i64 j = b + lane; j < e; j += kWarp) {
        const int u = __ldcs(&indices[j]);
        if (u == v || P[off + u] != pv) continue;
        l += w.cs(j);
      }
      l = warp_sum(l);
      if (!valid) {
        bad = kFlagBadLabel;
      } else {
        if (lane == 0) lacc[r] += l;
        if (lane == i) {
          my_c = pv;
          my_r = r;
          my_k = khat[v];
        }
      }
    }
    const unsigned m = __match_any_sync(kFullMask, my_c);
    s_k[wib][lane]   = my_k;
    s_r[wib][lane]   = 1 << my_r;
    __syncwarp();
    if (my_c >= 0 && lane == __ffs(m) - 1) {
      i64 sum   = 0;
      int rmask = 0;
      for (unsigned mm = m; mm; mm &= mm - 1) {
        sum += s_k[wib][__ffs(mm) - 1];
        rmask |= s_r[wib][__ffs(mm) - 1];
      }
      if constexpr (kSmem) {
        atomicAdd(reinterpret_cast<u64*>(&sK[my_c]), static_cast<u64>(sum));
      } else {
        atomicAdd(reinterpret_cast<u64*>(&Kc[my_c]), static_cast<u64>(sum));
      }
      if (R > 1) atomicOr(&repmask[my_c], rmask);
    }
    __syncwarp();
  }
  if (lane == 0) {
    for (int r = 0; r < kMaxReplicas; ++r)
      s_l[wib][r] = lacc[r];
    if (bad) atomicOr(&ctl->flags, bad);
  }
  __syncthreads();
  if (threadIdx.x < R) {
    i64 t = 0;
    for (int i = 0; i < kWarpsPerBlock; ++i)
      t += s_l[i][threadIdx.x];
    if (t) atomicAdd(reinterpret_cast<u64*>(&ctl->l_hat[threadIdx.x]), static_cast<u64>(t));
  }
  if constexpr (kSmem) {
    for (i64 c = threadIdx.x; c < C; c += blockDim.x)
      if (sK[c]) atomicAdd(reinterpret_cast<u64*>(&Kc[c]), static_cast<u64>(sK[c]));
  }
}

// t_c = (K_c / 2m)^2, grid-wide (the divisions stay off the tree block).
__global__ void volume_terms_kernel(const i64* __restrict__ Kc,
                                    i64 C,
                                    i64 two_m,
                                    double* __restrict__ t)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; c < C; c += stride)
    t[c] = volume_term(Kc[c], two_m);
}

// A block per replica: TREE1024 over its communities' t_c, ascending ids.
__global__ void __launch_bounds__(kTreeThreads) quality_tree_kernel(const double* __restrict__ t,
                                                                    const int* __restrict__ repmask,
                                                                    i64 C,
                                                                    int R,
                                                                    i64 two_m,
                                                                    double gamma,
                                                                    Control* ctl)
{
  using Scan = cub::BlockScan<int, kTreeThreads>;
  __shared__ typename Scan::TempStorage scan_tmp;
  __shared__ double buf[kTreeThreads];
  __shared__ double tree[kTreeThreads];
  const int r = blockIdx.x;
  const int j = threadIdx.x;
  double acc  = 0.0;
  if (two_m > 0) {
    if (R == 1) {
      for (i64 c = j; c < C; c += kTreeThreads)
        acc = __dadd_rn(acc, t[c]);
    } else {
      i64 seq = 0;  // items of replica r before this chunk
      for (i64 c0 = 0; c0 < C; c0 += kTreeThreads) {
        const i64 c    = c0 + j;
        const int mask = c < C ? repmask[c] : 0;
        if (r == 0 && __popc(mask) > 1) atomicOr(&ctl->flags, kFlagBadLabel);  // spans replicas
        const int f = (mask & (1 << r)) ? 1 : 0;
        int pos = 0, total = 0;
        Scan(scan_tmp).ExclusiveSum(f, pos, total);
        if (f) buf[pos] = t[c];
        __syncthreads();
        const int k = static_cast<int>((j - seq % kTreeThreads + kTreeThreads) % kTreeThreads);
        if (k < total) acc = __dadd_rn(acc, buf[k]);
        seq += total;
        __syncthreads();
      }
    }
  }
  const double sum = tree1024_finish(acc, tree);
  if (j == 0) {
    const i64 l   = ctl->l_hat[r];
    ctl->sum_t[r] = sum;
    ctl->q[r]     = two_m > 0 ? __dsub_rn(__ddiv_rn(__ll2double_rn(l), __ll2double_rn(two_m)),
                                      __dmul_rn(gamma, sum))
                              : 0.0;
  }
}

// Q of partition P (compact ids < C; R copies of n vertices, the graph and
// k_hat the single copy) with one copy's 2m_hat: ctl->l_hat / sum_t / q [r].
template <WKind K>
void run_quality(const i64* indptr,
                 const int* indices,
                 EdgeW<K> w,
                 const i64* khat,
                 const int* P,
                 i64 n,
                 int R,
                 i64 C,
                 i64 two_m,
                 double gamma,
                 const FinalBufs& fb,
                 Control* ctl,
                 cudaStream_t s)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(ctl->l_hat, 0, sizeof(ctl->l_hat), s));
  if (C > 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(fb.K, 0, C * sizeof(i64), s));
    RAFT_CUDA_TRY(cudaMemsetAsync(fb.rep, 0, C * sizeof(int), s));
  }
  if (n > 0 && C > 0) {
    const i64 rows = static_cast<i64>(R) * n;
    if (C <= kSmemCommunities) {
      const std::size_t smem = C * sizeof(i64);
      quality_rows_kernel<K, true>
        <<<grid_occ(quality_rows_kernel<K, true>, rows, kWarpsPerBlock, kBlock, smem),
           kBlock,
           smem,
           s>>>(indptr, indices, w, khat, P, n, R, C, fb.K, fb.rep, ctl);
    } else {
      quality_rows_kernel<K, false>
        <<<grid_occ(quality_rows_kernel<K, false>, rows, kWarpsPerBlock), kBlock, 0, s>>>(
          indptr, indices, w, khat, P, n, R, C, fb.K, fb.rep, ctl);
    }
    RAFT_CHECK_CUDA(s);
  }
  if (C > 0 && two_m > 0) {
    volume_terms_kernel<<<grid_items(C), kBlock, 0, s>>>(fb.K, C, two_m, fb.t);
    RAFT_CHECK_CUDA(s);
  }
  quality_tree_kernel<<<R, kTreeThreads, 0, s>>>(fb.t, fb.rep, C, R, two_m, gamma, ctl);
  RAFT_CHECK_CUDA(s);
}

constexpr int kSmemLabelCommunities = 4096;

// size_c and minv_c (the group leader, the lowest lane, holds the minimum id).
template <bool kSmem>
__global__ void label_count_kernel(const int* __restrict__ P,
                                   i64 n,
                                   i64 C,
                                   int* __restrict__ size,
                                   int* __restrict__ minv,
                                   Control* ctl)
{
  extern __shared__ int s_lab[];  // [2 C]: sizes, minimum ids
  if constexpr (kSmem) {
    for (i64 c = threadIdx.x; c < C; c += blockDim.x) {
      s_lab[c]     = 0;
      s_lab[C + c] = INT_MAX;
    }
    __syncthreads();
  }
  u32 bad          = 0;
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + lane;
    int c       = -1;
    if (v < n) {
      c = P[v];
      if (c < 0 || c >= C) {
        bad = kFlagBadLabel;
        c   = -1;
      }
    }
    const unsigned m = __match_any_sync(kFullMask, c);
    if (c >= 0 && lane == __ffs(m) - 1) {
      const int cnt = __popc(m);
      if constexpr (kSmem) {
        atomicAdd(&s_lab[c], cnt);
        atomicMin(&s_lab[C + c], static_cast<int>(v));
      } else {
        atomicAdd(&size[c], cnt);
        atomicMin(&minv[c], static_cast<int>(v));
      }
    }
  }
  if (bad) atomicOr(&ctl->flags, bad);
  if constexpr (kSmem) {
    __syncthreads();
    for (i64 c = threadIdx.x; c < C; c += blockDim.x) {
      if (s_lab[c]) {
        atomicAdd(&size[c], s_lab[c]);
        atomicMin(&minv[c], s_lab[C + c]);
      }
    }
  }
}

// key_c = ((2^31 - 1 - size_c) << 32) | minv_c: ascending = size-descending,
// ties by the smallest vertex id.
__global__ void label_keys_kernel(i64 C,
                                  const int* __restrict__ size,
                                  const int* __restrict__ minv,
                                  u64* __restrict__ keys,
                                  int* __restrict__ vals)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; c < C; c += stride) {
    keys[c] = (static_cast<u64>(0x7fffffffu - static_cast<u32>(size[c])) << 32) |
              static_cast<u64>(static_cast<u32>(minv[c]));
    vals[c] = static_cast<int>(c);
  }
}

__global__ void label_rank_kernel(i64 C,
                                  const int* __restrict__ sorted_vals,
                                  int* __restrict__ label_of)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < C; i += stride)
    label_of[sorted_vals[i]] = static_cast<int>(i);
}

__global__ void label_gather_kernel(const int* __restrict__ P,
                                    i64 n,
                                    i64 C,
                                    const int* __restrict__ label_of,
                                    int* __restrict__ labels_out)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c   = P[v];
    labels_out[v] = (c >= 0 && c < C) ? label_of[c] : -1;  // -1: flagged
  }
}

inline void run_rank_labels(const int* P,
                            i64 n,
                            i64 C,
                            int* labels_out,
                            const FinalBufs& fb,
                            void* cub,
                            std::size_t cub_bytes,
                            Control* ctl,
                            cudaStream_t s)
{
  if (n == 0 || C == 0) return;
  RAFT_CUDA_TRY(cudaMemsetAsync(fb.size, 0, C * sizeof(int), s));
  RAFT_CUDA_TRY(cudaMemsetAsync(fb.minv, 0x7f, C * sizeof(int), s));
  const unsigned gi = grid_items(n);
  if (C <= kSmemLabelCommunities) {
    label_count_kernel<true>
      <<<gi, kBlock, 2 * C * sizeof(int), s>>>(P, n, C, fb.size, fb.minv, ctl);
  } else {
    label_count_kernel<false><<<gi, kBlock, 0, s>>>(P, n, C, fb.size, fb.minv, ctl);
  }
  RAFT_CHECK_CUDA(s);
  const unsigned gc = grid_items(C);
  label_keys_kernel<<<gc, kBlock, 0, s>>>(C, fb.size, fb.minv, fb.keys_in, fb.vals_in);
  RAFT_CHECK_CUDA(s);
  std::size_t tb = cub_bytes;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(cub,
                                                tb,
                                                fb.keys_in,
                                                fb.keys_out,
                                                fb.vals_in,
                                                fb.vals_out,
                                                cub_items(C, "label sort"),
                                                0,
                                                63,
                                                s));
  label_rank_kernel<<<gc, kBlock, 0, s>>>(C, fb.vals_out, fb.label_of);
  RAFT_CHECK_CUDA(s);
  label_gather_kernel<<<gi, kBlock, 0, s>>>(P, n, C, fb.label_of, labels_out);
  RAFT_CHECK_CUDA(s);
}

// Calls fn with the EdgeW of fp32 (scaled on the fly), int64 or unit weights.
template <typename Fn>
void dispatch_weights(const float* wf, const i64* wq, double scale, i64 unit_weight, Fn&& fn)
{
  if (wf) {
    fn(EdgeW<WKind::F32>{wf, static_cast<float>(scale)});
  } else if (wq) {
    fn(EdgeW<WKind::I64>{wq});
  } else {
    fn(EdgeW<WKind::UNIT>{unit_weight});
  }
}

}  // namespace cugraph::detail::leiden_engine
