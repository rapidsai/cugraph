/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Host driver of the Leiden engine: LeidenDriver. Included by driver_common.cu only (the engine
// translation unit; see driver.hpp).
//
// The typed entry point runs the input path first (I1 ingest_check, layout, workspace, I2 (+ I4)
// quantisation of level 0, see kernels_ingest.cuh); run_engine then runs one resolution:
//
//   per iteration, per level: LOCAL_MOVE (DOWN) + COMPACT              -> SL1
//     (level_done: all-active TOP sweeps certify or re-enter), PLEIDEN_R +
//     AGGREGATE stage 1 -> SL2 (nothing merged: re-enter TOP, then the
//     CC-piece last resort), AGGREGATE stage 2; projection to level 0 with
//     V-cycle UP sweeps, the CC split and, with replicas, the best slice
//   exact Q, size-ordered labels
//
// SL1 and SL2 are the host synchronisations of a level: the sweeps of a LOCAL_MOVE run in launch
// chunks without a readback, and the refinement and AGGREGATE stage 1 are launched before the
// host knows the number of move communities, so one readback delivers both. Every kernel and
// CUDA graph runs on the caller's stream; nothing is allocated on the device (every array is
// carved from the workspace, and the level stack lives in the checked arena).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/graph.cuh"
#include "community/detail/leiden/kernels_aggregate.cuh"
#include "community/detail/leiden/kernels_final.cuh"
#include "community/detail/leiden/kernels_move.cuh"
#include "community/detail/leiden/kernels_refine.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace cugraph::detail::leiden_engine {

// Q gain threshold of n_iterations = -1 (until_stable_done).
constexpr double kDeltaQStop = 1e-6;

// level_done: after local moving at a level with n vertices produced C
// communities, whether the level loop of the current iteration ends here,
// without refining or aggregating this level. Implemented: C == n, Traag et
// al. 2019, Algorithm A.2 line 4 (done <- |P| = |V(G)|: "Terminate when each
// community consists of only one node"), as NetworKit ParallelLeiden.cpp:37
// (numberOfNodes() == result.numberOfSubsets()).
inline bool level_done(i64 C, i64 n) { return C == n; }

// iteration_final: whether iteration `it` (0-based) is the last one, for
// T = n_iterations (>= 1, or -1) and stop = what until_stable_done returned
// after the previous iteration (false before that). Implemented: T = -1:
// stop; else it + 1 >= T (iteration T - 1 is the T-th), the n_iterations of
// leiden_params_t ("Number of Leiden iterations"; "Or -1 to iterate until the
// partition is stable"), each iteration starting from the partition of the
// previous one (Traag et al. 2019, Section II: "using the partition
// identified in one iteration as starting point for the next iteration").
inline bool iteration_final(int T, int it, bool stop) { return T < 0 ? stop : it + 1 >= T; }

// until_stable_done: for n_iterations = -1, after `it` completed iterations,
// whether to stop iterating (the next iteration is then the final one).
// moved = whether the iteration just completed moved any vertex; q / q_prev =
// the modularity after it / after the one before (-inf before the first).
// Implemented: !moved || q - q_prev < kDeltaQStop || it + 1 >= max_it, per
// leiden_params_t::n_iterations ("until an iteration moves no vertex or
// improves the modularity by less than 1e-6, at most 20 iterations"; with the
// final one, at most max_it iterations run). !moved is a stable iteration (Traag et
// al. 2019, Section III-A: "An iteration of the Leiden algorithm in which the
// partition does not change is called a stable iteration"); NetworKit
// ParallelLeiden.cpp:60 iterates "while (changed && numberOfIterations > 0)".
inline bool until_stable_done(bool moved, double q, double q_prev, int it, int max_it)
{
  return !moved || q - q_prev < kDeltaQStop || it + 1 >= max_it;
}

// Knobs of one call (run_engine sets n_iterations, max_levels and graph_mode).
struct DriverParams {
  int n_iterations          = 2;  // >= 1, or -1 (until stable)
  int move_sweeps           = 4;  // DOWN cap
  int subrounds             = kNumSubrounds;
  int vcycle_sweeps         = 4;  // UP cap; 0 = no V-cycle
  bool vcycle_last          = false;
  int top_sweeps            = 32;  // TOP cap
  int max_levels            = kMaxLevels;
  int max_reentry           = 4;  // TOP re-entries of a level whose refinement merged nothing
  int max_iter_until_stable = 20;
  int graph_mode            = 2;  // 0: launches, 1: graph replays, 2: WHILE node (see EngineParams)
  ClassThresholds th{};
  GatherClasses gather{};
  TreeClasses trees{};
};

__global__ void drv_iota_kernel(int* __restrict__ P, i64 n)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride)
    P[v] = static_cast<int>(v);
}

__global__ void drv_replicate_partition_kernel(int* __restrict__ P, i64 n, int R)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int p = P[v];
    for (int r = 1; r < R; ++r)
      P[static_cast<i64>(r) * n + v] = p + static_cast<int>(r * n);
  }
}

// COMPACT of one replica slice: flags of its ids (< C), then their ranks.
__global__ void drv_slice_flags_kernel(
  const int* __restrict__ P, i64 n, i64 C, int* __restrict__ flags, Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = P[v];
    if (c < 0 || c >= C)
      atomicOr(&ctl->flags, kFlagBadLabel);
    else
      flags[c] = 1;  // idempotent store
  }
}

__global__ void drv_slice_relabel_kernel(const int* __restrict__ P_in,
                                         i64 n,
                                         i64 C,
                                         const int* __restrict__ rank,
                                         int* __restrict__ P_out,
                                         Control* ctl)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int c = P_in[v];
    P_out[v]    = (c >= 0 && c < C) ? rank[c] : 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) ctl->n_components = rank[C];
}

// One iteration: whether a vertex moved (stop rule of n_iterations = -1) and its levels.
struct IterationStat {
  bool moved = false;
  int levels = 0;
};

struct ResolutionResult {
  double modularity = 0.0;
  i64 n_clusters    = 0;
  std::vector<IterationStat> iters;
};

struct LevelGraph {
  const i64* indptr  = nullptr;
  const int* indices = nullptr;
  const float* wf    = nullptr;  // fp32: level 0 scaled by 2^s on the fly,
                                 // coarse levels exact integers (scale 1)
  const i64* wq   = nullptr;     // int64 W0q (level 0)
  i64 unit        = 1;           // UNIT kind: the quantised unit weight
  double scale    = 1.0;
  const i64* khat = nullptr;
  i64 n = 0, nnz = 0;
  i64 cls[kNumClasses] = {};       // vertices per degree class
  int* cmap            = nullptr;  // to level l + 1 (arena bottom)

  template <typename Fn>
  void with_w(Fn&& fn) const
  {
    dispatch_weights(wf, wq, scale, unit, std::forward<Fn>(fn));
  }
};

class LeidenDriver {
 public:
  LeidenDriver(Layout& L, const DriverParams& dp, cudaStream_t s) : L_(L), dp_(dp), s_(s) {}

  void set_level0(const QuantizeResult& q, i64 n, i64 nnz)
  {
    q_          = q;
    n_          = n;
    nnz_        = nnz;
    g0_         = LevelGraph{};
    g0_.indptr  = q.indptr;
    g0_.indices = q.indices;
    g0_.wf      = q.wkind == WKind::F32 ? q.wf32 : nullptr;
    g0_.wq      = q.wkind == WKind::I64 ? q.wq : nullptr;
    g0_.unit    = q.wkind == WKind::UNIT ? q.unit_q : 1;
    g0_.scale   = std::ldexp(1.0, q.s);
    g0_.khat    = L_.l0.khat;
    g0_.n       = n;
    g0_.nnz     = nnz;
    for (int c = 0; c < kNumClasses; ++c)
      g0_.cls[c] = q.class_count[c];
  }

  // One resolution from the singleton partition; false if the arena overflowed.
  bool run_resolution(double gamma, u32 seed, int R, int* labels_out, ResolutionResult& out)
  {
    const i64 n     = n_;
    const i64 two_m = q_.two_m_hat;
    if (two_m <= 0) {  // no counted entry: every vertex its own cluster
      out.n_clusters = n;
      drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(labels_out, n);
      RAFT_CHECK_CUDA(s_);
      return true;
    }
    lam_    = gamma / static_cast<double>(two_m);  // one IEEE division
    gamma_  = gamma;
    seed_   = seed;
    s64_    = seed64(seed);
    int* P0 = L_.persist.P0;
    drv_iota_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n);
    RAFT_CHECK_CUDA(s_);
    i64 C         = n;  // id bound of P0
    const int T   = dp_.n_iterations;
    const int V   = dp_.vcycle_sweeps;
    int it        = 0;
    double q_prev = -std::numeric_limits<double>::infinity();
    bool stop     = false;
    while (true) {
      const bool final  = iteration_final(T, it, stop);
      const bool vcycle = V > 0 && (!final || T == 1 || dp_.vcycle_last);
      IterationStat stat;
      if (it == 0 && R > 1) {
        drv_replicate_partition_kernel<<<grid_items(n), kBlock, 0, s_>>>(P0, n, R);
        RAFT_CHECK_CUDA(s_);
        const LevelGraph gu = union_level(R);
        i64 Cb              = 0;
        if (!leiden_iteration(gu, P0, gu.n, it, vcycle, stat, Cb)) return false;
        const i64 Cu = cc_split(gu, P0);  // union ranks [0, Cu)
        // Q_r of each replica with one copy's 2m_hat
        g0_.with_w([&](auto w) {
          run_quality(g0_.indptr,
                      g0_.indices,
                      w,
                      g0_.khat,
                      P0,
                      n,
                      R,
                      Cu,
                      two_m,
                      gamma,
                      L_.final_,
                      L_.ctl,
                      s_);
        });
        const Control c = readback();
        int best        = 0;
        for (int r = 1; r < R; ++r)
          if (c.q[r] > c.q[best]) best = r;  // ties -> lowest r
        C = compact_slice(P0 + static_cast<i64>(best) * n, n, Cu, P0);
      } else {
        i64 Cb = 0;
        if (!leiden_iteration(g0_, P0, C, it, vcycle, stat, Cb)) return false;
        C = cc_split(g0_, P0);
      }
      ++it;
      if (!final && T == -1) {  // is the next iteration final?
        const double q = quality(P0, C).q[0];
        stop           = until_stable_done(stat.moved, q, q_prev, it, dp_.max_iter_until_stable);
        q_prev         = q;
      }
      out.iters.push_back(std::move(stat));
      if (final) break;
    }
    launch_quality(P0, C);
    run_rank_labels(P0, n, C, labels_out, L_.final_, L_.cub, L_.cub_bytes, L_.ctl, s_);
    const Control c = readback();
    out.modularity  = c.q[0];
    out.n_clusters  = C;
    return true;
  }

 private:
  Layout& L_;
  DriverParams dp_;
  cudaStream_t s_;
  QuantizeResult q_;
  LevelGraph g0_;
  i64 n_ = 0, nnz_ = 0;
  double lam_ = 0.0, gamma_ = 0.0;
  u32 seed_ = 0;
  u64 s64_  = 0;
  MoveLaunch ml_;                // sweep calls, reused by every LOCAL_MOVE
  GraphChain graphs_[3];         // one sweep graph per weight kind (WKind)
  GraphChain refine_graphs_[3];  // PLEIDEN_R R1 - R8, per weight kind
  GraphChain stage1_graph_;      // R9 + AGGREGATE stage 1
  GraphChain compact_graph_;     // COMPACT when refinement is not launched

  // Control readback (one sync) with the device invariant checks.
  Control readback()
  {
    const Control c = read_control(L_.ctl, s_, L_.pinned);
    CUGRAPH_EXPECTS(!(c.flags & kFlagForestInvariant),
                    "leiden: internal error: refinement forest invariant violated.");
    CUGRAPH_EXPECTS(!(c.flags & kFlagGatherInvariant),
                    "leiden: internal error: multi-pass gather failed.");
    CUGRAPH_EXPECTS(!(c.flags & kFlagBadLabel),
                    "leiden: internal error: community id out of range.");
    return c;
  }

  int* level_partition(int l) const
  {
    if (l == 0) return L_.persist.P0;
    return (l & 1) ? L_.persist.Pa : L_.persist.Pb;
  }

  LevelGraph union_level(int R) const
  {
    LevelGraph g;
    g.indptr  = L_.uni.indptr;
    g.indices = L_.uni.indices;
    g.wf      = q_.wkind == WKind::F32 ? L_.uni.wf32 : nullptr;
    g.wq      = q_.wkind == WKind::I64 ? L_.uni.wq : nullptr;
    g.unit    = g0_.unit;
    g.scale   = g0_.scale;
    g.khat    = L_.uni.khat;
    g.n       = static_cast<i64>(R) * n_;
    g.nnz     = static_cast<i64>(R) * nnz_;
    for (int c = 0; c < kNumClasses; ++c)
      g.cls[c] = R * g0_.cls[c];
    return g;
  }

  // LOCAL_MOVE + COMPACT on level l, launch only; a TOP phase launches its
  // first chunk and a gated COMPACT (continue_move runs the rest).
  MoveRun launch_move(const LevelGraph& g,
                      int* P,
                      int l,
                      u32 phase,
                      int cap,
                      u32 sweep0,
                      i64 id_bound,
                      u32 it,
                      std::vector<KernelCall>& compact)
  {
    MoveParams prm;
    prm.lam      = lam_;
    prm.seed     = seed_;
    prm.it       = it;
    prm.level    = static_cast<u32>(l);
    prm.phase    = phase;
    prm.sweep0   = sweep0;
    prm.cap      = cap;
    prm.S        = dp_.subrounds;
    prm.th       = dp_.th;
    prm.id_bound = id_bound;
    MoveRun mr;
    g.with_w([&](auto w) {
      constexpr WKind KW = MoveWKindOf<decltype(w)>::value;
      ml_.graph          = dp_.graph_mode > 0 ? &graphs_[static_cast<int>(KW)] : nullptr;
      ml_.loop           = dp_.graph_mode == 2 && graph_loop_supported();
      if (phase == kPhaseTop) {
        // replays would run no-op sweeps after the certifying one
        prm.first_chunk = ml_.graph && ml_.loop ? kSweepChunk : 1;
        prm.defer_rest  = true;
      }
      mr = run_local_move(
        g.indptr, g.indices, w, g.khat, P, g.n, g.cls, prm, L_.move, L_.ctl, ml_, L_.pinned, s_);
    });
    launch_compact(
      P, g.n, L_.move, L_.persist.KS, L_.cub, L_.cub_bytes, L_.ctl, s_, &compact, mr.gate());
    return mr;
  }

  // The rest of a deferred TOP phase: its chunks, COMPACT, the SL1 readback.
  Control continue_move(const LevelGraph& g, int* P, MoveRun& mr)
  {
    continue_local_move(mr, ml_, L_.ctl, L_.pinned, s_);
    std::vector<KernelCall> compact;
    launch_compact(P, g.n, L_.move, L_.persist.KS, L_.cub, L_.cub_bytes, L_.ctl, s_, &compact);
    launch_chain(compact, dp_.graph_mode > 0 ? &compact_graph_ : nullptr, s_);
    const Control c = readback();  // SL1
    finish_local_move(mr, c);
    return c;
  }

  // PLEIDEN_R + AGGREGATE stage 1 of level l (C on the device; SL2 next).
  void launch_refine_begin(const LevelGraph& g,
                           const int* P,
                           int l,
                           u32 it,
                           int* cmap,
                           const AggregateOptions& opt,
                           AggregatePlan& plan,
                           std::vector<KernelCall> pre = {})
  {
    RefineParams rp;
    rp.lam       = lam_;
    rp.ctx_order = ctx(s64_, kTagOrder, it, static_cast<u32>(l), 0, 0);
    rp.trees     = dp_.trees;
    g.with_w([&](auto w) {
      constexpr WKind KW = MoveWKindOf<decltype(w)>::value;
      GraphChain* rg     = dp_.graph_mode > 0 ? &refine_graphs_[static_cast<int>(KW)] : nullptr;
      std::vector<KernelCall> r9;  // runs with AGGREGATE stage 1
      run_refine(g.indptr,
                 g.indices,
                 w,
                 g.khat,
                 P,
                 L_.persist.KS,
                 g.n,
                 g.n,
                 rp,
                 L_.refine,
                 cmap,
                 L_.cub,
                 L_.cub_bytes,
                 L_.ctl,
                 s_,
                 &L_.ctl->move_C,
                 rg,
                 &r9,
                 std::move(pre));
      plan = aggregate_begin(g.indptr,
                             g.indices,
                             w,
                             g.khat,
                             g.n,
                             g.nnz,
                             cmap,
                             opt,
                             L_.aggregate,
                             L_.arena,
                             L_.cub,
                             L_.cub_bytes,
                             L_.ctl,
                             s_,
                             std::move(r9));
    });
  }

  i64 cc_split(const LevelGraph& g, int* P)
  {
    g.with_w([&](auto w) {
      run_cc_split(g.indptr, g.indices, w, P, g.n, L_.final_, L_.cub, L_.cub_bytes, L_.ctl, s_);
    });
    return readback().n_components;
  }

  void launch_quality(const int* P, i64 C)
  {
    g0_.with_w([&](auto w) {
      run_quality(g0_.indptr,
                  g0_.indices,
                  w,
                  g0_.khat,
                  P,
                  n_,
                  1,
                  C,
                  q_.two_m_hat,
                  gamma_,
                  L_.final_,
                  L_.ctl,
                  s_);
    });
  }
  Control quality(const int* P, i64 C)
  {
    launch_quality(P, C);
    return readback();
  }

  // Ranks P_in[0, n) among the ids its slice uses into P_out; returns C.
  i64 compact_slice(const int* P_in, i64 n, i64 C, int* P_out)
  {
    const FinalBufs& fb = L_.final_;
    RAFT_CUDA_TRY(cudaMemsetAsync(fb.flags, 0, (C + 1) * sizeof(int), s_));
    drv_slice_flags_kernel<<<grid_items(n), kBlock, 0, s_>>>(P_in, n, C, fb.flags, L_.ctl);
    RAFT_CHECK_CUDA(s_);
    std::size_t tb = L_.cub_bytes;
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
      L_.cub, tb, fb.flags, fb.rank, cub_items(C + 1, "slice compact"), s_));
    drv_slice_relabel_kernel<<<grid_items(n), kBlock, 0, s_>>>(P_in, n, C, fb.rank, P_out, L_.ctl);
    RAFT_CHECK_CUDA(s_);
    return readback().n_components;
  }

  // One iteration on the level-0 graph g0 with partition P0 (ids < id_bound).
  // False on arena overflow; C_out: communities before the CC split.
  bool leiden_iteration(const LevelGraph& g0,
                        int* P0,
                        i64 id_bound,
                        int it,
                        bool vcycle,
                        IterationStat& stat,
                        i64& C_out)
  {
    LevelArena& arena = L_.arena;
    arena.reset();
    std::vector<LevelGraph> G;
    G.reserve(static_cast<std::size_t>(dp_.max_levels) + 1);
    G.push_back(g0);
    const u32 uit = static_cast<u32>(it);
    bool moved    = false;
    int lvl       = 0;
    u32 phase     = kPhaseDown;
    int reentry   = 0;
    u32 top_done  = 0;  // TOP sweeps run at this level (cumulative)
    i64 idb       = id_bound;
    AggregateOptions opt;
    opt.gather = dp_.gather;
    opt.move   = dp_.th;
    opt.stage1 = dp_.graph_mode > 0 ? &stage1_graph_ : nullptr;
    while (true) {
      LevelGraph& g  = G[lvl];
      int* P         = level_partition(lvl);
      const bool top = phase == kPhaseTop;
      std::vector<KernelCall> compact;
      MoveRun mr = launch_move(g,
                               P,
                               lvl,
                               phase,
                               top ? dp_.top_sweeps : dp_.move_sweeps,
                               top ? top_done : 0,
                               idb,
                               uit,
                               compact);
      // Refinement and AGGREGATE stage 1 launch before the host knows C
      // (one readback delivers C and SL2), except where C == n is likely
      // (TOP, singleton levels); C == n or no merge restores the arena.
      const bool speculate   = !top && !(lvl > 0 && idb == g.n);
      const std::size_t mark = arena.bottom;
      int* cmap = speculate ? arena.push_bottom<int>(static_cast<std::size_t>(g.n)) : nullptr;
      AggregatePlan plan;
      if (cmap) {  // COMPACT runs in refinement's graph
        launch_refine_begin(g, P, lvl, uit, cmap, opt, plan, std::move(compact));
      } else {
        launch_chain(compact, dp_.graph_mode > 0 ? &compact_graph_ : nullptr, s_);
      }
      Control c = readback();  // SL1 (+ SL2)
      finish_local_move(mr, c);
      if (mr.more_sweeps()) {  // TOP only (never speculated)
        CUGRAPH_EXPECTS(cmap == nullptr,
                        "leiden: internal error: refinement speculated on a deferred move.");
        c = continue_move(g, P, mr);
      }
      CUGRAPH_EXPECTS(c.move_C >= 0, "leiden: internal error: COMPACT gated off at SL1.");
      const MoveStats& st = mr.st;
      const i64 C         = c.move_C;
      if (top) top_done += static_cast<u32>(st.sweeps);
      moved = moved || st.moves > 0;
      idb   = C;
      if (level_done(C, g.n)) {
        if (cmap) {  // drop the speculative refinement / level
          arena.release_top();
          arena.bottom = mark;
        }
        if (!top) {  // all-active verification
          phase = kPhaseTop;
          continue;
        }
        ++stat.levels;  // certified gamma-separated top
        break;
      }
      if (!cmap) {  // refinement was not launched: launch it now
        cmap = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
        if (!cmap) return false;  // the arena cannot hold this level
        launch_refine_begin(g, P, lvl, uit, cmap, opt, plan);
        c = readback();  // SL2
      }
      if (c.n_next == g.n) {  // refinement merged nothing (C < n)
        arena.release_top();
        arena.bottom = mark;              // drop the speculative level
        if (reentry < dp_.max_reentry) {  // re-enter TOP sweeps at this level
          ++reentry;
          phase = kPhaseTop;
          continue;
        }
        // last resort (never expected): the connected pieces of P
        cmap = arena.push_bottom<int>(static_cast<std::size_t>(g.n));
        if (!cmap) return false;
        RAFT_CUDA_TRY(cudaMemcpyAsync(cmap, P, g.n * sizeof(int), cudaMemcpyDeviceToDevice, s_));
        const i64 pieces = cc_split(g, cmap);
        if (pieces == g.n) {  // output the pieces (uncertified)
          RAFT_CUDA_TRY(cudaMemcpyAsync(P, cmap, g.n * sizeof(int), cudaMemcpyDeviceToDevice, s_));
          arena.bottom = mark;
          idb          = g.n;
          ++stat.levels;
          break;
        }
        agg_set_n_next_kernel<<<1, 1, 0, s_>>>(L_.ctl, pieces);
        RAFT_CHECK_CUDA(s_);
        g.with_w([&](auto w) {
          plan = aggregate_begin(g.indptr,
                                 g.indices,
                                 w,
                                 g.khat,
                                 g.n,
                                 g.nnz,
                                 cmap,
                                 opt,
                                 L_.aggregate,
                                 arena,
                                 L_.cub,
                                 L_.cub_bytes,
                                 L_.ctl,
                                 s_);
        });
        c = readback();
      }
      const CoarseLevel lv = aggregate_finish(g.n,
                                              cmap,
                                              plan,
                                              c.n_next,
                                              c.nnz_next,
                                              L_.aggregate,
                                              arena,
                                              P,
                                              level_partition(lvl + 1),
                                              s_);
      if (!lv.ok) return false;
      g.cmap = cmap;
      ++stat.levels;
      LevelGraph ng;
      ng.indptr  = lv.indptr;
      ng.indices = lv.indices;
      ng.wf      = lv.weights;
      ng.khat    = lv.khat;
      ng.n       = lv.n;
      ng.nnz     = lv.nnz;
      for (int k = 0; k < kNumClasses; ++k)
        ng.cls[k] = c.coarse_class_count[k];
      G.push_back(ng);  // (invalidates g)
      ++lvl;
      phase    = kPhaseDown;
      reentry  = 0;
      top_done = 0;
      if (lvl == dp_.max_levels) {  // guard; kNN graphs need 5-12
        ++stat.levels;
        break;
      }
    }
    // projection to level 0, with the V-cycle's UP sweeps
    i64 Ccur = idb;
    for (int lp = lvl - 1; lp >= 0; --lp) {
      const LevelGraph& g = G[lp];
      int* P              = level_partition(lp);
      run_project(level_partition(lp + 1), g.cmap, g.n, P, s_);
      if (vcycle) {
        std::vector<KernelCall> compact;
        MoveRun mr = launch_move(g, P, lp, kPhaseUp, dp_.vcycle_sweeps, 0, Ccur, uit, compact);
        launch_chain(compact, dp_.graph_mode > 0 ? &compact_graph_ : nullptr, s_);
        const Control c = readback();  // SL1
        finish_local_move(mr, c);
        Ccur  = c.move_C;
        moved = moved || mr.st.moves > 0;
      }
    }
    stat.moved = moved;
    C_out      = Ccur;
    return true;
  }
};

}  // namespace cugraph::detail::leiden_engine
