/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// LOCAL_MOVE + COMPACT: M0 K_hat / csize, M1 active vertices into buckets
// [sub-round][class], M2 DECIDE per degree class with fused activation, M3
// apply, M5 order-preserving relabel, V1 projection. DECIDE takes a total-order
// argmax (score, -c) over exact integers of its sub-round's frozen state, so
// bucket, list and slot orders never reach a decision; no kernel reads a
// workspace byte this LOCAL_MOVE has not written.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/graph.cuh"
#include "community/detail/leiden/kernels_scan.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <vector>

namespace cugraph::detail::leiden_engine {

constexpr int kMoveLightBlocksPerSm = 5;          // light kernel __launch_bounds__
constexpr int kMoveMidSlots         = 256;        // per warp: 3 KB
constexpr int kMoveMidBlock         = 128;        // mid kernel: 4-warp blocks (13 KB)
constexpr int kMoveBlockSlots       = 2048;       // per block: 24 KB
constexpr int kMoveXlCap            = 1536;       // distinct keys per pass (load 0.75)
constexpr int kMoveSmemIds          = 4096;       // M0 block-privatised histogram
constexpr i64 kNoScore              = LLONG_MIN;  // "no candidate" (scores > -2^63)
constexpr int kDeferIdle            = -1;         // deferred-activation slot: unused
constexpr int kDeferMulti           = -2;         //   two movers with different targets
constexpr int kEmptySlot            = -1;         // free hash-table slot

enum : int { kMoveErrLabel = 1, kMoveErrBucket = 2 };

template <typename T>
struct MoveWKindOf;
template <WKind K>
struct MoveWKindOf<EdgeW<K>> {
  static constexpr WKind value = K;
};

__device__ __forceinline__ i64 move_group_sum(i64 x, unsigned m)
{
  i64 s = 0;
  for (unsigned mm = m; mm; mm &= mm - 1)
    s += __shfl_sync(m, x, __ffs(mm) - 1);
  return s;
}

__device__ __forceinline__ int move_sub_round(u64 ctx_move, int v, int S)
{
  return static_cast<int>(static_cast<u32>(mix64(ctx_move ^ static_cast<u64>(v)) >> 32) %
                          static_cast<u32>(S));
}

__device__ __forceinline__ bool move_better(i64 g1, int c1, i64 g2, int c2)
{
  return g1 > g2 || (g1 == g2 && c1 < c2);
}

__device__ __forceinline__ void move_warp_argmax(i64& g, int& c)
{
#pragma unroll
  for (int o = kWarp / 2; o > 0; o >>= 1) {
    const i64 og = __shfl_xor_sync(kFullMask, g, o);
    const int oc = __shfl_xor_sync(kFullMask, c, o);
    if (move_better(og, oc, g, c)) {
      g = og;
      c = oc;
    }
  }
}

// Table size for <= `keys` distinct keys at load <= 0.5 (>= 32), capped.
__device__ __forceinline__ int move_table_size(i64 keys, int slots)
{
  if (keys <= 16) return 32;
  const int t = 1 << (32 - __clz(static_cast<int>(2 * keys - 1)));
  return t < slots ? t : slots;
}

// DECIDE's code in dest's low word: the community, plus kMoveFreshBit for a
// move into an empty one (v's id or n + v; M3 stores instead of adding).
constexpr u32 kMoveFreshBit = 0x80000000u;
constexpr u32 kMoveIdMask   = 0x7fffffffu;

// Fixed point (numerics.cuh): edge weights and node weights k_hat are exact
// int64 counts; with Mult mu = make_mult(k_hat_v, lambda_hat), pen(K, mu) is
// lambda_hat * k_hat_v * K in the same units, exactly. Every decision compares
// int64 values only.

// move_score: the local-moving score of moving v into a community c != d (v's
// current community), from w_vc = W(v, c) (the summed weight between v and the
// members of c) and K_c (the summed node weight of c). The scores of all of v's
// options (move_score for each c != d, staying in d, an empty community) must
// be on one scale, so that DECIDE can take the best one.
// Implemented: w_vc - pen(K_c, mu) = W(v, c) - lambda_hat k_v K_c, the part of
// Delta H(v -> c) that depends on c, for modularity written as in Traag et al.
// 2019, Appendix A eq. (A5) (||v|| = k_v, gamma rescaled by 2m; Algorithm A.2
// line 17), as NetworKit ParallelLeiden.hpp:32-33 (modularityDelta: cutD -
// gamma * degreeV * volD / vol(V)). Staying in d scores move_score(W(v, d),
// K_d - k_v, mu) (ParallelLeiden.hpp:36-37, modularityThreshold) and an empty
// community move_score(0, 0, mu) = 0, so Delta H(v -> c) = score(c) - stay.
__device__ __forceinline__ i64 move_score(i64 w_vc, i64 K_c, Mult mu)
{
  return w_vc - pen(K_c, mu);
}

// move_decide_final: DECIDE(v), the local-moving decision for vertex v in
// community d. Inputs:
//   own   = W(v, d), the weight between v and the other members of d;
//   Kd    = the summed node weight of d (v included); kv = v's node weight;
//   mu    = make_mult(kv, lambda_hat);
//   bg/bc = the best move_score over the neighbouring communities c != d and
//           its id; among equal scores bc is the smallest id; (kNoScore,
//           INT_MAX) if v has no neighbouring community other than d;
//   csize = current member counts per community id, ids in [0, 2n).
// Returns the target as a u32: d (stay), bc, or an empty community encoded as
// (id | kMoveFreshBit). Empty-community ids: id v if csize[v] == 0, else id
// n + v if csize[n + v] == 0 (no other vertex can claim n + v); if neither is
// free, an empty community is not an option for v in this sub-round.
// Implemented: the argmax of the scores stay = move_score(own, Kd - kv, mu)
// (d), bg (bc) and 0 (an empty community); v leaves d only for a score
// strictly above stay, Traag, Waltman & van Eck 2019 (arXiv:1810.08473),
// Algorithm A.2 lines 17-18 ("Perform only strictly positive node
// movements"). The empty community needs 0 > stay and 0 > bg, so ties go to
// d, then to bc: NetworKit ParallelLeiden.cpp:237-239 ("if (0 > modThreshold
// || maxDelta > modThreshold)", "if (0 > maxDelta) // move node to empty
// community"). Ties depend on the inputs only.
__device__ __forceinline__ u32 move_decide_final(int v,
                                                 int d,
                                                 i64 n,
                                                 i64 own,
                                                 i64 Kd,
                                                 i64 kv,
                                                 Mult mu,
                                                 i64 bg,
                                                 int bc,
                                                 const int* __restrict__ csize)
{
  const i64 stay = move_score(own, Kd - kv, mu);
  if (bg < 0 && stay < 0) {  // an empty community (score 0) is the best
    const int s0 = csize[v], s1 = csize[n + v];
    if (s0 == 0) return static_cast<u32>(v) | kMoveFreshBit;
    if (s1 == 0) return static_cast<u32>(n + v) | kMoveFreshBit;
  }
  return static_cast<u32>(bg > stay ? bc : d);
}

// Sweep buckets: row r of `lists` ([S][n]) per sub-round, a segment per class.
struct MoveBuckets {
  int* lists           = nullptr;
  int* counts          = nullptr;  // [S][kNumClasses]
  int* err             = nullptr;
  i64 n                = 0;
  int off[kNumClasses] = {};
  int cap[kNumClasses] = {};

  __host__ __device__ int* list(int r, int c) const
  {
    return lists + static_cast<i64>(r) * n + off[c];
  }
  __host__ __device__ int* count(int r, int c) const { return counts + r * kNumClasses + c; }
};

// Bucket counts (cleared every sweep); movers and error bits are in Control.
struct MoveCounters {
  int* base    = nullptr;
  Control* ctl = nullptr;
  int S        = kNumSubrounds;
  int* buckets() const { return base; }
  // movers of sub-round r of the k-th sweep of the current chunk
  int* movers(int k, int r) const { return &ctl->sweep_movers[k][r]; }
  int* err() const { return &ctl->move_err; }
};

// Warp-aggregated append of v to bucket `key` = r * kNumClasses + c (-1: none).
__device__ __forceinline__ void move_bucket_append(const MoveBuckets& B, int key, int v)
{
  const unsigned m = __match_any_sync(kFullMask, key);
  if (key < 0) return;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(B.counts + key, __popc(m));
  base          = __shfl_sync(m, base, leader);
  const int pos = base + __popc(m & ((1u << lane) - 1u));
  const int r = key / kNumClasses, c = key - r * kNumClasses;
  if (pos < B.cap[c]) {
    B.list(r, c)[pos] = v;
  } else {
    atomicOr(B.err, kMoveErrBucket);
  }
}

__device__ __forceinline__ void move_list_append(int* list, int* count, bool mv, int v)
{
  const unsigned m = __ballot_sync(kFullMask, mv);
  if (!m) return;
  const int lane   = threadIdx.x & (kWarp - 1);
  const int leader = __ffs(m) - 1;
  int base         = 0;
  if (lane == leader) base = atomicAdd(count, __popc(m));
  base = __shfl_sync(kFullMask, base, leader);
  if (mv) list[base + __popc(m & ((1u << lane) - 1u))] = v;
}

// M0: K[c] = sum of k_hat, csize[c] = |{v : P[v] == c}| for c < id_bound
// (zeroed by the caller); resets the other csize and the deferred slots.
template <bool kSmem>
__global__ void __launch_bounds__(kBlock) move_level_init_kernel(const int* __restrict__ P,
                                                                 const i64* __restrict__ khat,
                                                                 i64 n,
                                                                 i64 id_bound,
                                                                 i64* __restrict__ Kc,
                                                                 int* __restrict__ csize,
                                                                 int* __restrict__ defer_slot,
                                                                 int* __restrict__ err)
{
  extern __shared__ __align__(16) unsigned char move_init_smem[];
  i64* sK = reinterpret_cast<i64*>(move_init_smem);
  int* sC = reinterpret_cast<int*>(sK + (kSmem ? id_bound : 0));
  if constexpr (kSmem) {
    for (i64 c = threadIdx.x; c < id_bound; c += blockDim.x) {
      sK[c] = 0;
      sC[c] = 0;
    }
    __syncthreads();
  }
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  int bad          = 0;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + lane;
    int c       = -1;
    i64 k       = 0;
    if (v < n) {
      c = P[v];
      k = khat[v];
      if (c < 0 || c >= id_bound) {
        bad = 1;
        c   = -1;
      }
      csize[n + v] = 0;
      if (v >= id_bound) csize[v] = 0;
      defer_slot[v] = kDeferIdle;
    }
    const unsigned m = __match_any_sync(kFullMask, c);
    const i64 sum    = move_group_sum(k, m);
    if (c >= 0 && lane == __ffs(m) - 1) {
      if constexpr (kSmem) {
        atomicAdd(reinterpret_cast<u64*>(&sK[c]), static_cast<u64>(sum));
        atomicAdd(&sC[c], __popc(m));
      } else {
        atomicAdd(reinterpret_cast<u64*>(&Kc[c]), static_cast<u64>(sum));
        atomicAdd(&csize[c], __popc(m));
      }
    }
  }
  if (bad) atomicOr(err, kMoveErrLabel);
  if constexpr (kSmem) {
    __syncthreads();
    for (i64 c = threadIdx.x; c < id_bound; c += blockDim.x) {
      if (sC[c]) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[c]), static_cast<u64>(sK[c]));
        atomicAdd(&csize[c], sC[c]);
      }
    }
  }
}

// The `active` flag of a sweep as three bitmaps (in MoveBufs::rank): A (active
// at the start = R of the previous sweep; null: all), R (activated after the
// vertex's sub-round) and Ap (appended to a later sub-round of this sweep): in
// sub-round r, active[u] iff (sub(u) <= r ? R[u] : A[u] | Ap[u]).
struct MoveFlags {
  u32* A  = nullptr;  // null: all-active sweep (completed by M1, see MoveAct)
  u32* R  = nullptr;
  u32* Ap = nullptr;
};

__device__ __forceinline__ bool move_bit(const u32* bm, int v)
{
  return (bm[v >> 5] >> (v & 31)) & 1u;
}

// move_activates: whether neighbour u of a vertex that just moved into
// community cv must be evaluated again, given c_u = u's community after the
// sub-round of the move (the callers guarantee that timing). Implemented:
// c_u != cv, the neighbours outside the mover's new community, Traag et al.
// 2019, Algorithm A.2 lines 20-21 (N <- {u | (u, v) in E(G), u not in C'};
// Q.add(N - Q)) and Section III ("all neighbours of the node that do not
// belong to the node's new community"), as NetworKit ParallelLeiden.cpp:279
// (result[neighbor] != bestCommunity).
__device__ __forceinline__ bool move_activates(int c_u, int cv) { return c_u != cv; }

// Activation, fused into M2: when v (sub-round r) moves to cv, each counted
// neighbour u with move_activates(P_post[u], cv) becomes active: appended to
// its later sub-round's bucket if sub(u) > r (once per sweep), else its R bit
// is set. P_post[u] (u's community after sub-round r) == P[u] unless
// sub(u) == r; then the target is deferred to the next sweep's M1 (two
// different targets: kDeferMulti). Skipped in a phase's first and last sweep
// and in TOP sweeps.
struct MoveState {
  int k;          // current sweep, advanced by the last node of a sweep
  int chunk0;     // first sweep of the launch chunk (sweep_movers row 0)
  int chunk_end;  // the chunk's sweeps are [chunk0, chunk_end)
  int pad;
};

// A phase has ended once its chunk reaches the cap or holds a zero-move sweep;
// a TOP phase's COMPACT runs behind this gate.
struct MovePhaseGate {
  const MoveState* st = nullptr;  // null: the phase has ended
  int S               = 0;
  int cap             = 0;
};

__device__ __forceinline__ bool move_phase_done(const MovePhaseGate& g, const Control* ctl)
{
  if (!g.st || g.st->chunk_end >= g.cap) return true;
  const int len = g.st->chunk_end - g.st->chunk0;
  for (int j = 0; j < len; ++j) {
    int moved = 0;
    for (int r = 0; r < g.S; ++r)
      moved |= ctl->sweep_movers[j][r];
    if (!moved) return true;
  }
  return false;
}

// Phase-constant arguments of M1, M2 and M3 for sub-round r.
struct MoveAct {
  MoveBuckets B;
  ClassThresholds th;
  u32* bm0            = nullptr;  // activity bitmaps: R of sweep k is bm[k & 1], A
  u32* bm1            = nullptr;  // of sweep k is R of sweep k - 1 (MoveFlags)
  u32* ap             = nullptr;  // appended in the current sweep
  int* slot           = nullptr;  // [n] deferred target per vertex
  Control* ctl        = nullptr;
  const MoveState* st = nullptr;
  u64 s64             = 0;  // seed64
  u32 it = 0, level = 0, phase = 0, sweep0 = 0;
  int cap = 0;  // sweeps of the phase
  int S   = kNumSubrounds;
  int r   = 0;
  int top = 0;  // TOP phase: every sweep all-active, no activation

  __device__ __forceinline__ int sweep() const { return st->k; }
  __device__ __forceinline__ bool do_low(int k) const { return !top && k + 1 < cap; }
  __device__ __forceinline__ bool do_high(int k) const { return !top && k > 0; }
  __device__ __forceinline__ bool on(int k) const { return do_low(k) || do_high(k); }
  __device__ __forceinline__ MoveFlags flags(int k) const
  {
    MoveFlags F;
    F.A  = (top || k == 0) ? nullptr : ((k & 1) ? bm0 : bm1);
    F.R  = (k & 1) ? bm1 : bm0;
    F.Ap = ap;
    return F;
  }
  __device__ __forceinline__ u64 ctx_move(int k) const
  {
    return ctx(s64, kTagMove, it, level, phase, sweep0 + static_cast<u32>(k));
  }
  __device__ __forceinline__ u64 stamp_hi(int k) const
  {
    return static_cast<u64>(1u + static_cast<u32>(k) * S + r) << 32;
  }
  __device__ __forceinline__ int* mover_count(int k) const
  {
    return &ctl->sweep_movers[k - st->chunk0][r];
  }
};

// Activation for the counted entry (u, c = frozen P[u]) of a mover with target
// cv (u = -1: none); every lane of the warp calls it.
__device__ __forceinline__ void move_activate_entry(
  const MoveAct& t, int k, const i64* __restrict__ indptr, int u, int c, int cv)
{
  int key = -1;
  if (u >= 0) {
    const int su      = move_sub_round(t.ctx_move(k), u, t.S);
    const MoveFlags F = t.flags(k);
    u32* rw           = F.R + (u >> 5);
    const u32 bit     = 1u << (u & 31);
    if (su < t.r) {
      if (t.do_low(k) && move_activates(c, cv) && !(*rw & bit)) atomicOr(rw, bit);
    } else if (su > t.r) {
      if (t.do_high(k) && move_activates(c, cv) && !((F.A[u >> 5] | F.Ap[u >> 5]) & bit) &&
          !(atomicOr(&F.Ap[u >> 5], bit) & bit)) {
        // u has the counted entry (u, v), so its row is empty only in
        // a partition that does not own u (partition.cuh)
        const i64 deg = indptr[u + 1] - indptr[u];
        if (deg > 0) key = su * kNumClasses + degree_class(deg, t.th);
      }
    } else if (t.do_low(k) && !(*rw & bit)) {
      const int old = atomicCAS(&t.slot[u], kDeferIdle, cv);
      if (old != kDeferIdle && old != cv && old != kDeferMulti) atomicExch(&t.slot[u], kDeferMulti);
    }
  }
  move_bucket_append(t.B, key, u);
}

// M1 over vertices [v0, v1) (v0 a multiple of 32): applies the deferred
// activations, buckets the active vertices (F.A null: all) and clears their
// words of R and Ap; a no-op after a zero-move sweep of the chunk.
__global__ void __launch_bounds__(kBlock) move_bucket_kernel(
  const i64* __restrict__ indptr, i64 v0, i64 v1, MoveAct t, const int* __restrict__ P)
{
  const int k = t.sweep();
  const int S = t.S;
  if (k > t.st->chunk0) {
    const int* prev = t.ctl->sweep_movers[k - 1 - t.st->chunk0];
    int moved       = 0;
    for (int r = 0; r < S; ++r)
      moved |= prev[r];
    if (!moved) return;
  }
  const MoveFlags F        = t.flags(k);
  const u64 ctx_move       = t.ctx_move(k);
  const ClassThresholds th = t.th;
  const MoveBuckets& B     = t.B;
  int* __restrict__ slot   = t.slot;
  const int lane           = threadIdx.x & (kWarp - 1);
  const i64 stride         = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = v0 + static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < v1;
       base += stride) {
    const i64 v = base + lane;
    if (lane == 0) {
      F.R[base >> 5]  = 0;
      F.Ap[base >> 5] = 0;
    }
    int key  = -1;
    bool act = false;
    if (v < v1) {
      act = !F.A || move_bit(F.A, static_cast<int>(v));
      if (F.A) {
        const int sl = slot[v];
        if (sl != kDeferIdle) {
          slot[v] = kDeferIdle;
          act     = act || sl == kDeferMulti || move_activates(P[v], sl);
        }
      }
      if (act) {
        const int r = move_sub_round(ctx_move, static_cast<int>(v), S);
        const int c = degree_class(indptr[v + 1] - indptr[v], th);
        key         = r * kNumClasses + c;
      }
    }
    if (F.A) {  // this warp owns word base / 32
      const unsigned m = __ballot_sync(kFullMask, act);
      if (lane == 0) F.A[base >> 5] = m;
    }
    move_bucket_append(B, key, static_cast<int>(v));
  }
}

template <WKind K>
struct MoveEval {
  const i64* indptr;
  const int* indices;
  EdgeW<K> w;
  const i64* khat;
  const int* P;
  const i64* Kc;
  const int* csize;
  i64* dest;
  int* movers;
  i64 n;
  double lam;
};

// Vertices per warp of a group prefetch (layout only).
__device__ __forceinline__ int move_vpw(i64 cnt, i64 nwarps)
{
  const i64 rounds = (cnt + nwarps * kWarp - 1) / (nwarps * kWarp);
  if (rounds < 1) return 1;
  const i64 v = (cnt + nwarps * rounds - 1) / (nwarps * rounds);
  return v < 1 ? 1 : (v > kWarp ? kWarp : static_cast<int>(v));
}

// move_stays: exact pruning for the common case. Given own = W(v, d),
// oth = W(v, V \ d) (v's total weight to vertices outside d), Kd and kv, return
// true only if move_decide_final would return d whatever the neighbouring
// communities' weights and sizes are (it runs before the per-community sums
// exist). Returning false is always correct, only slower.
// Implemented: move_score(own, Kd - kv, mu) >= oth. Every other option scores
// at most oth: move_score(w_vc, K_c, mu) <= w_vc <= oth (pen >= 0) and an
// empty community 0 <= oth, and DECIDE leaves d only for a strictly higher
// score (Traag et al. 2019, Algorithm A.2 line 18; NetworKit
// ParallelLeiden.cpp:237). The bound is attained (w_vc = oth, K_c = 0).
__device__ __forceinline__ bool move_stays(i64 own, i64 oth, i64 Kd, i64 kv, Mult mu)
{
  return move_score(own, Kd - kv, mu) >= oth;
}

template <WKind K>
__device__ __forceinline__ void move_light_entry(
  const MoveEval<K>& a, int v, i64 b, i64 e, int lane, int& u, int& c, i64& x)
{
  u           = -1;
  c           = -1;
  x           = 0;
  const i64 j = b + lane;
  if (j < e) {
    const int y = __ldcs(&a.indices[j]);
    if (y != v) {
      const i64 q = a.w.cs(j);
      if (q > 0) {
        u = y;
        c = a.P[y];
        x = q;
      }
    }
  }
}

// DECIDE of the vertex held by lane `src` (pk, pKd: its k_hat and K_d) from
// every lane's row entry (c, x); lane src updates res.
template <WKind K, bool kPrune>
__device__ __forceinline__ void move_light_decide(const MoveEval<K>& a,
                                                  i64* s_w,
                                                  int lane,
                                                  int src,
                                                  int v,
                                                  int d,
                                                  int c,
                                                  i64 x,
                                                  Mult pm,
                                                  i64 pk,
                                                  i64 pKd,
                                                  u32& res)
{
  Mult mu;
  mu.M  = __shfl_sync(kFullMask, pm.M, src);
  mu.sh = __shfl_sync(kFullMask, pm.sh, src);
  if constexpr (kPrune) {
    const i64 own = warp_sum(c == d ? x : 0ll);
    const i64 oth = warp_sum(c != d ? x : 0ll);  // x = 0 if c < 0
    bool stays    = false;
    if (lane == src) stays = move_stays(own, oth, pKd, pk, mu);
    if (__shfl_sync(kFullMask, stays, src)) return;  // res = d
  }
  const unsigned m = __match_any_sync(kFullMask, c);
  s_w[lane]        = x;
  __syncwarp();
  i64 own = 0, bg = kNoScore;
  int bc = INT_MAX;
  if (c >= 0 && lane == __ffs(m) - 1) {
    i64 sum = 0;
    for (unsigned mm = m; mm; mm &= mm - 1)
      sum += s_w[__ffs(mm) - 1];
    if (c == d) {
      own = sum;
    } else {
      bg = move_score(sum, a.Kc[c], mu);
      bc = c;
    }
  }
  __syncwarp();
  own = warp_sum(own);
  move_warp_argmax(bg, bc);
  if (lane == src) res = move_decide_final(v, d, a.n, own, pKd, pk, mu, bg, bc, a.csize);
}

// light: warp per vertex, one entry per lane, match_any groups and a group
// prefetch; two vertices per step keep both rows' dependent loads in flight.
template <WKind K, bool kPrune>
__global__ void __launch_bounds__(kBlock, kMoveLightBlocksPerSm) move_eval_light_kernel(
  MoveEval<K> a, const int* __restrict__ list, const int* __restrict__ count, int cap, MoveAct t)
{
  __shared__ i64 s_w[kWarpsPerBlock][kWarp];
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  const int k    = t.sweep();
  const bool act = t.on(k);
  const i64 cnt  = min(*count, cap);
  const i64 nw   = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  const i64 gw   = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + wib;
  const int vpw  = move_vpw(cnt, nw);
  for (i64 base = gw * vpw; base < cnt; base += nw * vpw) {
    int pv = -1, pd = 0;
    i64 pb = 0, pe = 0, pk = 0, pKd = 0;
    Mult pm{0ull, 0};
    if (lane < vpw && base + lane < cnt) {
      pv  = list[base + lane];
      pb  = a.indptr[pv];
      pe  = a.indptr[pv + 1];
      pd  = a.P[pv];
      pk  = a.khat[pv];
      pKd = a.Kc[pd];
      pm  = make_mult(pk, a.lam);
    }
    u32 res       = static_cast<u32>(pd);
    unsigned todo = __ballot_sync(kFullMask, pv >= 0);
    while (todo) {  // two vertices per step (warp-uniform)
      const int s1 = __ffs(todo) - 1;
      todo &= todo - 1;
      const int s2 = todo ? __ffs(todo) - 1 : -1;
      if (s2 >= 0) todo &= todo - 1;
      const int t2 = s2 >= 0 ? s2 : s1;
      const int v1 = __shfl_sync(kFullMask, pv, s1);
      const i64 b1 = __shfl_sync(kFullMask, pb, s1);
      const i64 e1 = __shfl_sync(kFullMask, pe, s1);
      const int d1 = __shfl_sync(kFullMask, pd, s1);
      const int v2 = __shfl_sync(kFullMask, pv, t2);
      const i64 b2 = __shfl_sync(kFullMask, pb, t2);
      const i64 e2 = s2 >= 0 ? __shfl_sync(kFullMask, pe, t2) : b2;
      const int d2 = __shfl_sync(kFullMask, pd, t2);
      int u1, u2, c1, c2;
      i64 x1, x2;
      move_light_entry(a, v1, b1, e1, lane, u1, c1, x1);
      move_light_entry(a, v2, b2, e2, lane, u2, c2, x2);
      move_light_decide<K, kPrune>(a, s_w[wib], lane, s1, v1, d1, c1, x1, pm, pk, pKd, res);
      if (s2 >= 0)
        move_light_decide<K, kPrune>(a, s_w[wib], lane, s2, v2, d2, c2, x2, pm, pk, pKd, res);
      if (act) {  // activation by the movers (rows in registers)
        const u32 r1 = __shfl_sync(kFullMask, res, s1);
        if (r1 != static_cast<u32>(d1))  // warp-uniform
          move_activate_entry(t, k, a.indptr, u1, c1, static_cast<int>(r1 & kMoveIdMask));
        if (s2 >= 0) {
          const u32 r2 = __shfl_sync(kFullMask, res, s2);
          if (r2 != static_cast<u32>(d2))
            move_activate_entry(t, k, a.indptr, u2, c2, static_cast<int>(r2 & kMoveIdMask));
        }
      }
    }
    if (pv >= 0) { a.dest[pv] = t.stamp_hi(k) | static_cast<u64>(res); }
    move_list_append(a.movers, t.mover_count(k), pv >= 0 && res != static_cast<u32>(pd), pv);
  }
}

// mid: warp per vertex, 256-slot table per warp.
template <WKind K, bool kPrune>
__global__ void __launch_bounds__(kMoveMidBlock, 7) move_eval_mid_kernel(
  MoveEval<K> a, const int* __restrict__ list, const int* __restrict__ count, int cap, MoveAct t)
{
  constexpr int kWarps = kMoveMidBlock / kWarp;
  __shared__ int s_key[kWarps][kMoveMidSlots];
  __shared__ i64 s_val[kWarps][kMoveMidSlots];
  __shared__ i64 s_w[kWarps][kWarp];
  const int lane = threadIdx.x & (kWarp - 1), wib = threadIdx.x / kWarp;
  int* keys = s_key[wib];
  i64* vals = s_val[wib];
  for (int s = lane; s < kMoveMidSlots; s += kWarp) {
    keys[s] = kEmptySlot;
    vals[s] = 0;
  }
  __syncwarp();
  const int k    = t.sweep();
  const bool act = t.on(k);
  const i64 cnt  = min(*count, cap);
  const i64 nw   = static_cast<i64>(gridDim.x) * kWarps;
  const i64 gw   = static_cast<i64>(blockIdx.x) * kWarps + wib;
  const int vpw  = move_vpw(cnt, nw);
  for (i64 base = gw * vpw; base < cnt; base += nw * vpw) {
    int pv = -1, pd = 0;
    i64 pb = 0, pe = 0, pk = 0, pKd = 0;
    Mult pm{0ull, 0};
    if (lane < vpw && base + lane < cnt) {
      pv  = list[base + lane];
      pb  = a.indptr[pv];
      pe  = a.indptr[pv + 1];
      pd  = a.P[pv];
      pk  = a.khat[pv];
      pKd = a.Kc[pd];
      pm  = make_mult(pk, a.lam);
    }
    u32 res       = static_cast<u32>(pd);
    unsigned todo = __ballot_sync(kFullMask, pv >= 0);
    while (todo) {
      const int src = __ffs(todo) - 1;
      todo &= todo - 1;
      const int v = __shfl_sync(kFullMask, pv, src);
      const i64 b = __shfl_sync(kFullMask, pb, src);
      const i64 e = __shfl_sync(kFullMask, pe, src);
      const int d = __shfl_sync(kFullMask, pd, src);
      Mult mu;
      mu.M  = __shfl_sync(kFullMask, pm.M, src);
      mu.sh = __shfl_sync(kFullMask, pm.sh, src);
      if constexpr (kPrune) {  // first pass: own / oth only
        i64 own = 0, oth = 0;
        for (i64 j = b + lane; j < e; j += kWarp) {
          const int u = __ldcs(&a.indices[j]);
          if (u == v) continue;
          const i64 q = a.w.cs(j);
          if (q <= 0) continue;
          if (a.P[u] == d)
            own += q;
          else
            oth += q;
        }
        own        = warp_sum(own);
        oth        = warp_sum(oth);
        bool stays = false;
        if (lane == src) stays = move_stays(own, oth, pKd, pk, mu);
        if (__shfl_sync(kFullMask, stays, src)) continue;  // res = d
      }
      const int tsize = move_table_size(e - b, kMoveMidSlots);
      const u32 mask  = static_cast<u32>(tsize - 1);
      for (i64 j0 = b; j0 < e; j0 += kWarp) {
        const i64 j = j0 + lane;
        int c       = -1;
        i64 x       = 0;
        if (j < e) {
          const int u = __ldcs(&a.indices[j]);
          if (u != v) {
            const i64 q = a.w.cs(j);
            if (q > 0) {
              c = a.P[u];
              x = q;
            }
          }
        }
        const unsigned m = __match_any_sync(kFullMask, c);
        s_w[wib][lane]   = x;
        __syncwarp();
        if (c >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += s_w[wib][__ffs(mm) - 1];
          warp_table_add(keys, vals, mask, c, sum);
        }
        __syncwarp();
      }
      i64 own = 0, bg = kNoScore;
      int bc = INT_MAX;
      for (int s = lane; s < tsize; s += kWarp) {
        const int c = keys[s];
        if (c == kEmptySlot) continue;
        const i64 sum = vals[s];
        keys[s]       = kEmptySlot;  // the next vertex starts empty
        if (c == d) {
          own = sum;
        } else {
          const i64 g = move_score(sum, a.Kc[c], mu);
          if (move_better(g, c, bg, bc)) {
            bg = g;
            bc = c;
          }
        }
      }
      own = warp_sum(own);
      move_warp_argmax(bg, bc);
      if (lane == src) res = move_decide_final(v, d, a.n, own, pKd, pk, mu, bg, bc, a.csize);
      __syncwarp();  // the scan finishes before the next claims
      if (act) {     // activation by a mover (its row is cached)
        const u32 rv = __shfl_sync(kFullMask, res, src);
        if (rv != static_cast<u32>(d)) {  // warp-uniform
          const int cv = static_cast<int>(rv & kMoveIdMask);
          for (i64 j0 = b; j0 < e; j0 += kWarp) {
            const i64 j = j0 + lane;
            int u = -1, c = -1;
            if (j < e) {
              const int y = a.indices[j];
              if (y != v && a.w(j) > 0) {
                u = y;
                c = a.P[y];
              }
            }
            move_activate_entry(t, k, a.indptr, u, c, cv);
          }
        }
      }
    }
    if (pv >= 0) { a.dest[pv] = t.stamp_hi(k) | static_cast<u64>(res); }
    move_list_append(a.movers, t.mover_count(k), pv >= 0 && res != static_cast<u32>(pd), pv);
  }
}

// Insert into the block table (several warps may hold a key; values start at
// 0, inserters add). Claims, then counts, so a key racing with its own claim
// never splits a pass. False if the pass must split (> cap keys or no slot).
__device__ __forceinline__ bool move_block_table_add(
  int* keys, i64* vals, u32 mask, int c, i64 sum, int* claimed, int cap)
{
  volatile int* vk = keys;
  u32 h            = fmix32(static_cast<u32>(c)) & mask;
  for (u32 probes = 0; probes <= mask;) {
    const int cur = vk[h];
    if (cur == c) {
      atomicAdd(reinterpret_cast<u64*>(&vals[h]), static_cast<u64>(sum));
      return true;
    }
    if (cur == kEmptySlot) {
      const int prev = atomicCAS(&keys[h], kEmptySlot, c);
      if (prev == kEmptySlot) {
        atomicAdd(reinterpret_cast<u64*>(&vals[h]), static_cast<u64>(sum));
        return atomicAdd(claimed, 1) < cap;
      }
      continue;  // claimed by another inserter: re-read slot h
    }
    h = (h + 1) & mask;
    ++probes;
  }
  return false;
}

// block: block per vertex, shared-memory table in hash-range passes; npass
// starts at ceil(deg / pass_cap) and doubles when a pass overflows. The argmax
// and W(v, d) carry across passes, so npass never changes the result.
template <WKind K, bool kPrune>
__global__ void __launch_bounds__(kBlock, 3) move_eval_block_kernel(MoveEval<K> a,
                                                                    const int* __restrict__ list,
                                                                    const int* __restrict__ count,
                                                                    int cap,
                                                                    int pass_cap,
                                                                    MoveAct t)
{
  __shared__ int s_key[kMoveBlockSlots];
  __shared__ i64 s_val[kMoveBlockSlots];
  __shared__ i64 s_w[kWarpsPerBlock][kWarp];
  __shared__ i64 s_g[kWarpsPerBlock];
  __shared__ int s_c[kWarpsPerBlock];
  __shared__ i64 s_own;
  __shared__ int s_claimed;
  __shared__ int s_ovf;
  __shared__ u32 s_res;
  const int tid  = threadIdx.x;
  const int lane = tid & (kWarp - 1), wib = tid / kWarp;
  for (int s = tid; s < kMoveBlockSlots; s += blockDim.x) {
    s_key[s] = kEmptySlot;
    s_val[s] = 0;
  }
  const i64 cnt  = min(*count, cap);
  const int k    = t.sweep();
  const bool act = t.on(k);
  __syncthreads();
  for (i64 i = blockIdx.x; i < cnt; i += gridDim.x) {
    const int v = list[i];
    const i64 b = a.indptr[v], e = a.indptr[v + 1];
    const int d   = a.P[v];
    const i64 kv  = a.khat[v];
    const Mult mu = make_mult(kv, a.lam);
    const i64 deg = e - b;
    const int tsize =
      move_table_size(deg < pass_cap ? deg : static_cast<i64>(pass_cap), kMoveBlockSlots);
    const u32 mask = static_cast<u32>(tsize - 1);
    u64 npass      = deg > pass_cap ? static_cast<u64>((deg + pass_cap - 1) / pass_cap) : 1ull;
    if constexpr (kPrune) {  // first pass: own / oth only
      i64 own = 0, oth = 0;
      for (i64 j = b + tid; j < e; j += blockDim.x) {
        const int u = __ldcs(&a.indices[j]);
        if (u == v) continue;
        const i64 q = a.w.cs(j);
        if (q <= 0) continue;
        if (a.P[u] == d)
          own += q;
        else
          oth += q;
      }
      own = warp_sum(own);
      oth = warp_sum(oth);
      if (lane == 0) {
        s_g[wib]    = own;
        s_w[wib][0] = oth;
      }
      __syncthreads();
      bool stays = false;
      if (tid == 0) {
        i64 o = 0, t = 0;
        for (int q = 0; q < kWarpsPerBlock; ++q) {
          o += s_g[q];
          t += s_w[q][0];
        }
        stays = move_stays(o, t, a.Kc[d], kv, mu);
        s_ovf = stays ? 1 : 0;
      }
      __syncthreads();
      const bool skip = s_ovf != 0;
      __syncthreads();  // every thread has read s_ovf
      if (skip) {
        if (tid == 0) a.dest[v] = t.stamp_hi(k) | static_cast<u64>(d);
        continue;
      }
    }
    u64 p  = 0;
    i64 rg = kNoScore;  // running argmax (thread 0)
    int rc = INT_MAX;
    if (tid == 0) s_own = 0;
    while (p < npass) {
      if (tid == 0) {
        s_claimed = 0;
        s_ovf     = 0;
      }
      __syncthreads();
      for (i64 j0 = b + static_cast<i64>(wib) * kWarp; j0 < e; j0 += blockDim.x) {
        const i64 j = j0 + lane;
        int c       = -1;
        i64 x       = 0;
        if (j < e) {
          const int u = __ldcs(&a.indices[j]);
          if (u != v) {
            const i64 q = a.w.cs(j);
            if (q > 0) {
              const int cu = a.P[u];
              if (npass == 1 || table_pass(cu, npass) == p) {
                c = cu;
                x = q;
              }
            }
          }
        }
        const unsigned m = __match_any_sync(kFullMask, c);
        s_w[wib][lane]   = x;
        __syncwarp();
        if (c >= 0 && lane == __ffs(m) - 1) {
          i64 sum = 0;
          for (unsigned mm = m; mm; mm &= mm - 1)
            sum += s_w[wib][__ffs(mm) - 1];
          if (!move_block_table_add(s_key, s_val, mask, c, sum, &s_claimed, pass_cap))
            atomicOr(&s_ovf, 1);  // read after the barrier
        }
        __syncwarp();
      }
      __syncthreads();
      const bool ovf = s_ovf != 0;
      __syncthreads();  // every thread has read s_ovf
      if (ovf) {        // split: clear the pass, resume at 2p of 2 npass
        for (int s = tid; s < tsize; s += blockDim.x) {
          s_key[s] = kEmptySlot;
          s_val[s] = 0;
        }
        npass *= 2;
        p *= 2;
        continue;  // the loop head synchronises before the next pass
      }
      i64 g  = kNoScore;
      int cc = INT_MAX;
      for (int s = tid; s < tsize; s += blockDim.x) {
        const int c = s_key[s];
        if (c == kEmptySlot) continue;
        const i64 sum = s_val[s];
        s_key[s]      = kEmptySlot;  // the next pass / vertex starts empty
        s_val[s]      = 0;
        if (c == d) {
          s_own = sum;
        } else {
          const i64 sc = move_score(sum, a.Kc[c], mu);
          if (move_better(sc, c, g, cc)) {
            g  = sc;
            cc = c;
          }
        }
      }
      move_warp_argmax(g, cc);
      if (lane == 0) {
        s_g[wib] = g;
        s_c[wib] = cc;
      }
      __syncthreads();
      if (wib == 0) {
        g  = lane < kWarpsPerBlock ? s_g[lane] : kNoScore;
        cc = lane < kWarpsPerBlock ? s_c[lane] : INT_MAX;
        move_warp_argmax(g, cc);
        if (lane == 0 && move_better(g, cc, rg, rc)) {
          rg = g;
          rc = cc;
        }
      }
      __syncthreads();
      ++p;
    }
    if (tid == 0) {
      const u32 res = move_decide_final(v, d, a.n, s_own, a.Kc[d], kv, mu, rg, rc, a.csize);
      a.dest[v]     = t.stamp_hi(k) | static_cast<u64>(res);
      if (res != static_cast<u32>(d)) a.movers[atomicAdd(t.mover_count(k), 1)] = v;
      s_res = res;
    }
    __syncthreads();
    const u32 rv = s_res;
    if (act && rv != static_cast<u32>(d)) {  // activation (block-wide)
      const int cv = static_cast<int>(rv & kMoveIdMask);
      for (i64 j0 = b + static_cast<i64>(wib) * kWarp; j0 < e; j0 += blockDim.x) {
        const i64 j = j0 + lane;
        int u = -1, c = -1;
        if (j < e) {
          const int y = a.indices[j];
          if (y != v && a.w(j) > 0) {
            u = y;
            c = a.P[y];
          }
        }
        move_activate_entry(t, k, a.indptr, u, c, cv);
      }
    }
    __syncthreads();  // s_own, s_res
  }
}

// M3, thread per mover: K_hat / csize moves aggregated per source and target
// (match_any); a move into an empty community stores (k_v, 1). P[v] <- c_v.
__global__ void __launch_bounds__(kBlock) move_apply_kernel(const i64* __restrict__ khat,
                                                            int* __restrict__ P,
                                                            i64* __restrict__ Kc,
                                                            int* __restrict__ csize,
                                                            const i64* __restrict__ dest,
                                                            const int* __restrict__ movers,
                                                            MoveAct t,
                                                            int mover_cap)
{
  const int* __restrict__ mover_count = t.mover_count(t.sweep());
  const int lane                      = threadIdx.x & (kWarp - 1);
  const i64 cnt                       = min(*mover_count, mover_cap);
  const i64 stride                    = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < cnt;
       base += stride) {
    const i64 i = base + lane;
    int v = -1, d = -1, c = -1;
    bool fresh = false;
    i64 k      = 0;
    if (i < cnt) {
      v              = movers[i];
      d              = P[v];
      const u32 code = static_cast<u32>(dest[v]);
      c              = static_cast<int>(code & kMoveIdMask);
      fresh          = (code & kMoveFreshBit) != 0;
      k              = khat[v];
    }
    {
      const unsigned m = __match_any_sync(kFullMask, d);
      const i64 s      = move_group_sum(k, m);
      if (d >= 0 && lane == __ffs(m) - 1) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[d]), static_cast<u64>(-s));
        atomicSub(&csize[d], __popc(m));
      }
    }
    {
      const int ca     = fresh ? -1 : c;
      const unsigned m = __match_any_sync(kFullMask, ca);
      const i64 s      = move_group_sum(k, m);
      if (ca >= 0 && lane == __ffs(m) - 1) {
        atomicAdd(reinterpret_cast<u64*>(&Kc[ca]), static_cast<u64>(s));
        atomicAdd(&csize[ca], __popc(m));
      }
    }
    if (fresh) {
      Kc[c]    = k;
      csize[c] = 1;
    }
    if (v >= 0) P[v] = c;
  }
}

// M5: P[v] <- rank[P[v]], rank = exclusive_scan(csize > 0) over [0, 2n],
// KS[rank[c]] <- K[c], move_C <- C; behind a closed gate only move_C <- -1.
__global__ void move_compact_kernel(const int* __restrict__ rank,
                                    const int* __restrict__ csize,
                                    const i64* __restrict__ Kc,
                                    i64 n,
                                    int* __restrict__ P,
                                    i64* __restrict__ KS,
                                    Control* ctl,
                                    MovePhaseGate gate)
{
  if (gate.st) {  // uniform over the grid
    __shared__ int done;
    if (threadIdx.x == 0) done = move_phase_done(gate, ctl);
    __syncthreads();
    if (!done) {
      if (blockIdx.x == 0 && threadIdx.x == 0) ctl->move_C = -1;
      return;
    }
  }
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * n; i += stride) {
    if (csize[i] > 0) KS[rank[i]] = Kc[i];
    if (i < n) P[i] = rank[P[i]];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->move_C = rank[2 * n];
}

__global__ void move_clear_kernel(int* __restrict__ p, int count)
{
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    p[i] = 0;
}

// Starts a launch chunk of sweeps [k0, k_end): sweep counter, cleared movers.
__global__ void move_chunk_kernel(MoveState* __restrict__ st,
                                  int k0,
                                  int k_end,
                                  Control* __restrict__ ctl)
{
  if (threadIdx.x == 0) {
    st->k         = k0;
    st->chunk0    = k0;
    st->chunk_end = k_end;
  }
  for (int i = threadIdx.x; i < kSweepChunk * kMaxSubrounds; i += blockDim.x)
    (&ctl->sweep_movers[0][0])[i] = 0;
}

__global__ void move_sweep_end_kernel(MoveState* __restrict__ st)
{
  if (threadIdx.x == 0) st->k += 1;
}

#if LEIDEN_GRAPH_LOOP
// Sweep end in a WHILE node: loop while the sweep moved and the chunk has more.
__global__ void move_sweep_end_loop_kernel(MoveState* __restrict__ st,
                                           const Control* __restrict__ ctl,
                                           int S,
                                           cudaGraphConditionalHandle h)
{
  if (threadIdx.x != 0) return;
  const int k    = st->k;
  const int* row = ctl->sweep_movers[k - st->chunk0];
  int moved      = 0;
  for (int r = 0; r < S; ++r)
    moved |= row[r];
  st->k = k + 1;
  cudaGraphSetConditional(h, (moved && k + 1 < st->chunk_end) ? 1u : 0u);
}
#endif

__global__ void move_project_kernel(const int* __restrict__ Pc,
                                    const int* __restrict__ cmap,
                                    i64 n_fine,
                                    int* __restrict__ Pf)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n_fine; v += stride)
    Pf[v] = Pc[cmap[v]];
}

struct MoveParams {
  double lam = 0.0;  // lam_hat = gamma / 2m_hat (one host division)
  u32 seed   = 0;
  u32 it     = 0;
  u32 level  = 0;
  u32 phase  = kPhaseDown;
  u32 sweep0 = 0;  // TOP: TOP sweeps already run at this level
  int cap    = 4;  // DOWN / UP 4, TOP 32
  int S      = kNumSubrounds;
  ClassThresholds th{};
  int xl_cap   = kMoveXlCap;  // distinct keys per hash-range pass
  i64 id_bound = -1;          // P ids in [0, id_bound) on entry; -1: n
  // sweeps of the first chunk (replayed TOP phases: 1, they mostly certify)
  int first_chunk = kSweepChunk;
  bool defer_rest = false;  // TOP: continue_local_move launches the rest
};

struct MoveStats {
  i64 moves  = 0;
  int sweeps = 0;
  i64 last   = 0;  // movers of the last executed sweep
};

// A launched LOCAL_MOVE; the last chunk's movers arrive with the caller's next
// readback (finish_local_move). A deferred phase launched only its first chunk.
struct MoveRun {
  MoveStats st;
  int S            = kNumSubrounds;
  int len          = 0;        // sweeps of the pending chunk (0: nothing pending)
  bool stopped     = false;    // a sweep moved nothing (later sweeps are no-ops)
  int next         = 0;        // first sweep not launched yet
  int cap          = 0;        // sweeps of the phase
  MoveState* state = nullptr;  // the device sweep state of the phase

  bool more_sweeps() const { return !stopped && next < cap; }
  // the gate for this phase's COMPACT while sweeps may remain unlaunched
  MovePhaseGate gate() const
  {
    return more_sweeps() ? MovePhaseGate{state, S, cap} : MovePhaseGate{};
  }
};

struct MoveLaunch {
  GraphChain* graph = nullptr;
  bool loop         = false;      // sweeps of a chunk in one WHILE graph launch
  std::vector<KernelCall> calls;  // the sweep's calls
  GraphDeps deps;                 // their graph dependencies
  MoveAct act;                    // the phase's arguments (sub-round 0)
};

// Folds the pending chunk's movers (from a readback) into the statistics.
inline void move_fold_chunk(MoveRun& mr, const Control& h)
{
  CUGRAPH_EXPECTS(!(h.move_err & kMoveErrLabel),
                  "leiden: internal error: local move community ids out of range.");
  CUGRAPH_EXPECTS(!(h.move_err & kMoveErrBucket),
                  "leiden: internal error: local move bucket overflow.");
  for (int k = 0; k < mr.len && !mr.stopped; ++k) {
    i64 moves = 0;
    for (int r = 0; r < mr.S; ++r)
      moves += h.sweep_movers[k][r];
    mr.st.moves += moves;
    mr.st.last = moves;
    ++mr.st.sweeps;
    if (moves == 0) mr.stopped = true;  // converged (TOP: certified)
  }
  mr.len = 0;
}

inline void finish_local_move(MoveRun& mr, const Control& h)
{
  if (mr.len > 0) move_fold_chunk(mr, h);
}

inline void run_level_init(
  const int* P, const i64* khat, i64 n, i64 id_bound, const MoveBufs& mb, int* err, cudaStream_t s)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(mb.K, 0, id_bound * sizeof(i64), s));
  RAFT_CUDA_TRY(cudaMemsetAsync(mb.csize, 0, id_bound * sizeof(int), s));
  const unsigned g = grid_items(n);
  if (id_bound <= kMoveSmemIds) {
    const std::size_t smem = static_cast<std::size_t>(id_bound) * (sizeof(i64) + sizeof(int));
    move_level_init_kernel<true>
      <<<g, kBlock, smem, s>>>(P, khat, n, id_bound, mb.K, mb.csize, mb.defer_slot, err);
  } else {
    move_level_init_kernel<false>
      <<<g, kBlock, 0, s>>>(P, khat, n, id_bound, mb.K, mb.csize, mb.defer_slot, err);
  }
  RAFT_CHECK_CUDA(s);
}

inline MoveBuckets move_buckets(const MoveBufs& mb,
                                const MoveCounters& ctr,
                                i64 n,
                                const i64 (&class_count)[kNumClasses])
{
  MoveBuckets B;
  B.lists  = mb.buckets;
  B.counts = ctr.buckets();
  B.err    = ctr.err();
  B.n      = n;
  i64 off  = 0;
  for (int c = 0; c < kNumClasses; ++c) {
    B.off[c] = static_cast<int>(off);
    B.cap[c] = static_cast<int>(class_count[c]);
    off += class_count[c];
  }
  CUGRAPH_EXPECTS(off == n, "leiden: internal error: degree class counts must sum to n.");
  return B;
}

// M2 calls for sub-round r, one per class (absent classes disabled).
template <WKind K, bool kPrune>
void move_eval_calls_t(KernelCall* c,
                       const MoveEval<K>& a,
                       const MoveBuckets& B,
                       int r,
                       const i64 (&cc)[kNumClasses],
                       int xl_cap,
                       const MoveAct& t)
{
  c[0].set(cc[kClassLight] > 0,
           move_eval_light_kernel<K, kPrune>,
           dim3(grid_resident(cc[kClassLight], kWarpsPerBlock, kMoveLightBlocksPerSm)),
           dim3(kBlock),
           0,
           a,
           B.list(r, kClassLight),
           B.count(r, kClassLight),
           B.cap[kClassLight],
           t);
  c[1].set(cc[kClassMid] > 0,
           move_eval_mid_kernel<K, kPrune>,
           dim3(grid_occ(
             move_eval_mid_kernel<K, kPrune>, cc[kClassMid], kMoveMidBlock / kWarp, kMoveMidBlock)),
           dim3(kMoveMidBlock),
           0,
           a,
           B.list(r, kClassMid),
           B.count(r, kClassMid),
           B.cap[kClassMid],
           t);
  c[2].set(cc[kClassBlock] > 0,
           move_eval_block_kernel<K, kPrune>,
           dim3(grid_occ(move_eval_block_kernel<K, kPrune>, cc[kClassBlock], 1)),
           dim3(kBlock),
           0,
           a,
           B.list(r, kClassBlock),
           B.count(r, kClassBlock),
           B.cap[kClassBlock],
           xl_cap,
           t);
}

template <WKind K>
void move_eval_calls(KernelCall* c,
                     const MoveEval<K>& a,
                     const MoveBuckets& B,
                     int r,
                     const i64 (&cc)[kNumClasses],
                     int xl_cap,
                     bool prune,
                     const MoveAct& t = MoveAct{})
{
  if (prune) {
    move_eval_calls_t<K, true>(c, a, B, r, cc, xl_cap, t);
  } else {
    move_eval_calls_t<K, false>(c, a, B, r, cc, xl_cap, t);
  }
}

// Pruning pays off where most vertices stay: not in the first iteration's DOWN.
inline bool move_prune(const MoveParams& p) { return p.phase != kPhaseDown || p.it > 0; }

// Calls per sweep: clear, M1, per sub-round the three M2 classes (independent
// branches: disjoint vertices) and M3, the sweep end.
constexpr int kSweepHead     = 2;
constexpr int kSweepPerRound = 4;
inline int move_sweep_calls(int S) { return kSweepHead + kSweepPerRound * S + 1; }

inline GraphDeps move_sweep_deps(int S)
{
  GraphDeps d(static_cast<std::size_t>(move_sweep_calls(S)));
  d[1].assign(1, 0);
  int prev = 1;
  for (int r = 0; r < S; ++r) {
    const int b = kSweepHead + kSweepPerRound * r;
    for (int i = 0; i < 3; ++i)
      d[b + i].assign(1, prev);
    d[b + 3] = {b, b + 1, b + 2};
    prev     = b + 3;
  }
  d[kSweepHead + kSweepPerRound * S].assign(1, prev);  // the sweep end
  return d;
}

// Launches sweep chunks from mr.next: `first` sweeps now, each later chunk
// (with `rest`) after a readback, until a chunk holds a zero-move sweep.
inline void move_launch_chunks(
  MoveRun& mr, MoveLaunch& ml, int first, bool rest, Control* ctl, void* pinned, cudaStream_t s)
{
  const KernelCall* c = ml.calls.data();
  const int ncalls    = static_cast<int>(ml.calls.size());
  const bool loop     = ml.graph && ml.loop;
  for (bool lead = true; mr.next < mr.cap; lead = false) {
    if (!lead) {  // long phases (TOP): one readback between chunks
      if (!rest) return;
      const Control h = read_control(ctl, s, pinned);
      move_fold_chunk(mr, h);
      if (mr.stopped) return;
    }
    const int c0  = mr.next;
    const int len = std::min(lead ? first : kSweepChunk, mr.cap - c0);
    move_chunk_kernel<<<1, kBlock, 0, s>>>(mr.state, c0, c0 + len, ctl);
    RAFT_CHECK_CUDA(s);
    if (loop) {
      ml.graph->launch(c, ncalls, ml.deps, s, true);
    } else {
      for (int k = 0; k < len; ++k) {
        if (ml.graph) {
          ml.graph->launch(c, ncalls, ml.deps, s);
        } else {
          launch_calls(c, ncalls, s);
        }
      }
    }
    mr.len  = len;
    mr.next = c0 + len;
  }
}

// LOCAL_MOVE of a level up to its first sweep: M0, then the sweep's calls in
// ml (M1 buckets the vertices [v0, v1)); stopped if n == 0.
template <WKind K>
MoveRun prepare_local_move(const i64* indptr,
                           const int* indices,
                           EdgeW<K> w,
                           const i64* khat,
                           int* P,
                           i64 n,
                           const i64 (&class_count)[kNumClasses],
                           const MoveParams& prm,
                           const MoveBufs& mb,
                           Control* ctl,
                           MoveLaunch& ml,
                           cudaStream_t s,
                           i64 v0,
                           i64 v1)
{
  const int S = prm.S;
  MoveRun mr;
  mr.S    = S;
  mr.next = mr.cap = prm.cap;  // set again below when sweeps launch
  if (n == 0) {                // every sweep is empty: the first one ends the phase
    mr.st.sweeps = prm.cap > 0 ? 1 : 0;
    mr.stopped   = true;
    return mr;
  }
  const i64 id_bound = prm.id_bound < 0 ? n : prm.id_bound;
  CUGRAPH_EXPECTS(id_bound >= 1 && id_bound <= n, "leiden: internal error: id_bound out of range.");
  const MoveCounters ctr{mb.bucket_count, ctl, S};
  const MoveBuckets B = move_buckets(mb, ctr, n, class_count);
  RAFT_CUDA_TRY(cudaMemsetAsync(ctr.err(), 0, sizeof(int), s));
  run_level_init(P, khat, n, id_bound, mb, ctr.err(), s);  // M0

  const unsigned g_items = grid_items(n);
  MoveEval<K> a{indptr, indices, w, khat, P, mb.K, mb.csize, mb.dest, mb.movers, n, prm.lam};
  // activity bitmaps in the rank buffer: R(s) = bm[s % 2], A(s) = R(s - 1)
  const i64 words = (n + 31) / 32;
  MoveState* st   = reinterpret_cast<MoveState*>(mb.desc);
  MoveAct t;
  t.B              = B;
  t.th             = prm.th;
  t.bm0            = reinterpret_cast<u32*>(mb.rank);
  t.bm1            = reinterpret_cast<u32*>(mb.rank) + words;
  t.ap             = reinterpret_cast<u32*>(mb.rank) + 2 * words;
  t.slot           = mb.defer_slot;
  t.ctl            = ctl;
  t.st             = st;
  t.s64            = seed64(prm.seed);
  t.it             = prm.it;
  t.level          = prm.level;
  t.phase          = prm.phase;
  t.sweep0         = prm.sweep0;
  t.cap            = prm.cap;
  t.S              = S;
  t.top            = prm.phase == kPhaseTop ? 1 : 0;
  const bool prune = move_prune(prm);
  const int ncalls = move_sweep_calls(S);
  ml.calls.resize(static_cast<std::size_t>(ncalls));
  KernelCall* c = ml.calls.data();
  c[0].set(true, move_clear_kernel, dim3(1), dim3(kBlock), 0, ctr.buckets(), S * kNumClasses);
  c[1].set(
    true, move_bucket_kernel, dim3(grid_items(v1 - v0)), dim3(kBlock), 0, indptr, v0, v1, t, P);
  for (int r = 0; r < S; ++r) {
    KernelCall* cr = c + kSweepHead + kSweepPerRound * r;
    t.r            = r;
    move_eval_calls<K>(cr, a, B, r, class_count, prm.xl_cap, prune, t);
    cr[3].set(true,
              move_apply_kernel,
              dim3(g_items),
              dim3(kBlock),
              0,
              khat,
              P,
              mb.K,
              mb.csize,
              mb.dest,
              mb.movers,
              t,
              static_cast<int>(n));
  }
#if LEIDEN_GRAPH_LOOP
  if (ml.graph && ml.loop)
    c[ncalls - 1].set(true,
                      move_sweep_end_loop_kernel,
                      dim3(1),
                      dim3(kWarp),
                      0,
                      st,
                      ctl,
                      S,
                      cudaGraphConditionalHandle{0});
  else
#endif
    c[ncalls - 1].set(true, move_sweep_end_kernel, dim3(1), dim3(kWarp), 0, st);
  if (ml.graph && ml.deps.size() != static_cast<std::size_t>(ncalls)) ml.deps = move_sweep_deps(S);
  t.r      = 0;
  ml.act   = t;
  mr.state = st;
  mr.next  = 0;
  return mr;
}

// LOCAL_MOVE of a level (launch only), P ids in [0, id_bound) -> [0, 2n).
template <WKind K>
MoveRun run_local_move(const i64* indptr,
                       const int* indices,
                       EdgeW<K> w,
                       const i64* khat,
                       int* P,
                       i64 n,
                       const i64 (&class_count)[kNumClasses],
                       const MoveParams& prm,
                       const MoveBufs& mb,
                       Control* ctl,
                       MoveLaunch& ml,
                       void* pinned,
                       cudaStream_t s)
{
  MoveRun mr =
    prepare_local_move(indptr, indices, w, khat, P, n, class_count, prm, mb, ctl, ml, s, 0, n);
  if (mr.stopped) return mr;
  const int first = std::max(1, std::min(prm.first_chunk, static_cast<int>(kSweepChunk)));
  move_launch_chunks(mr, ml, first, !prm.defer_rest, ctl, pinned, s);
  return mr;
}

// The rest of a deferred LOCAL_MOVE after the caller's readback.
inline void continue_local_move(
  MoveRun& mr, MoveLaunch& ml, Control* ctl, void* pinned, cudaStream_t s)
{
  CUGRAPH_EXPECTS(mr.len == 0, "leiden: internal error: move chunk not read back.");
  if (!mr.more_sweeps()) return;
  move_launch_chunks(mr, ml, kSweepChunk, true, ctl, pinned, s);
}

// COMPACT after LOCAL_MOVE (launch only); `gate`: the phase's MoveRun::gate().
inline void launch_compact(int* P,
                           i64 n,
                           const MoveBufs& mb,
                           i64* KS,
                           void* cub,
                           std::size_t cub_bytes,
                           Control* ctl,
                           cudaStream_t s,
                           std::vector<KernelCall>* defer = nullptr,
                           MovePhaseGate gate             = {})
{
  if (n == 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(&ctl->move_C, 0, sizeof(i64), s));
    return;
  }
  const i64 n2 = 2 * n;
  // graph-friendly scan; the driver runs it in PLEIDEN_R's graph (`defer`)
  std::vector<KernelCall> c(4);
  scan_calls(c.data(), ScanUsed{mb.csize, n2}, mb.rank, n2 + 1, cub, cub_bytes);
  c[3].set(true,
           move_compact_kernel,
           dim3(grid_items(n2)),
           dim3(kBlock),
           0,
           mb.rank,
           mb.csize,
           mb.K,
           n,
           P,
           KS,
           ctl,
           gate);
  if (defer) {
    defer->insert(defer->end(), c.begin(), c.end());
  } else {
    launch_calls(c.data(), 4, s);
  }
}

// V1: P_fine[v] <- P_coarse[cmap[v]].
inline void run_project(
  const int* P_coarse, const int* cmap, i64 n_fine, int* P_fine, cudaStream_t s)
{
  if (n_fine == 0) return;
  move_project_kernel<<<grid_items(n_fine), kBlock, 0, s>>>(P_coarse, cmap, n_fine, P_fine);
  RAFT_CHECK_CUDA(s);
}

}  // namespace cugraph::detail::leiden_engine
