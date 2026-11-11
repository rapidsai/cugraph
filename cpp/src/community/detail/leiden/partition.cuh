/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Level 0 split by rows into P partitions (multi-GPU). Partition p owns the
// rows [v0, v1) of the level-0 CSR (v0 a multiple of 32, so each word of an
// activity bitmap has one owner) and stores only them, behind a full-length
// indptr (other rows empty) with global column ids. Every per-vertex and
// per-community array stays full length (replicated): row kernels compute
// the values of the owned rows, which an allgather of the owned slices then
// replicates, and kernels over replicated inputs run on every partition with
// identical results. Local moving exchanges each sub-round's move list and
// the activations of other partitions' vertices; refinement exchanges its
// per-vertex values; the contraction to level 1 exchanges exact int64 partial
// rows, merged by the owner of each coarse row range and gathered on
// partition 0 ("root"), which runs levels >= 1 with LeidenDriver; the CC
// split merges the partitions' forests on the root. Every decision reads
// exact integers of the vertex's own row and replicated state, so labels and
// Q are those of one partition, bit for bit.
//
// All traffic between partitions goes through Exchange, collectives over the
// P partitions, one per rank (RaftExchange over raft::comms, driver_mg.cuh). Host
// decisions use values that are identical everywhere or exchanged. Included by
// driver_common.cu only (the engine translation unit).

#include "community/detail/leiden/driver.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace cugraph::detail::leiden_engine {

// Collectives over the P partitions; every process makes the same calls.
// Device arguments: one pointer per local partition (on its device, ordered
// after the work queued on its stream); counts and bounds: global.
class Exchange {
 public:
  virtual ~Exchange()             = default;
  virtual int parts() const       = 0;
  virtual int n_local() const     = 0;
  virtual int global(int i) const = 0;  // partition of local partition i
  // partition q's slice [bounds[q], bounds[q + 1]) of `elem`-byte elements
  // into the same slice of every partition's array
  virtual void allgather_slices(void* const* bufs, int elem, const i64* bounds) = 0;
  // partition q's counts[q] elements at displ[q] = sum of counts[< q] into
  // every partition's buffer
  virtual void allgatherv(void* const* bufs, int elem, const i64* counts) = 0;
  // k host values per partition: in (n_local x k) -> out (P x k)
  virtual void allgather_host(const i64* in, int k, i64* out) = 0;
  // out = OR of every partition's in (`words` u32, out may be in);
  // scratch: P x words per local partition
  virtual void allreduce_or(u32* const* in, u32* const* out, u32* const* scratch, i64 words) = 0;
  // root's recv <- counts[q] elements of each partition's send, in order
  virtual void gatherv(
    int root, const void* const* send, int elem, const i64* counts, void* recv) = 0;
  // root's `bytes` into every partition's buffer
  virtual void bcast(int root, void* const* bufs, i64 bytes) = 0;
};

// Exchange 1 of a sub-round: (v << 32) | DECIDE code of each local mover.
__global__ void part_pack_kernel(const int* __restrict__ movers,
                                 i64 count,
                                 const i64* __restrict__ dest,
                                 u64* __restrict__ out)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride) {
    const int v = movers[i];
    out[i] =
      (static_cast<u64>(static_cast<u32>(v)) << 32) | static_cast<u64>(static_cast<u32>(dest[v]));
  }
}

// The sub-round's global mover list for M3 on every partition.
__global__ void part_unpack_kernel(const u64* __restrict__ in,
                                   i64 total,
                                   i64* __restrict__ dest,
                                   int* __restrict__ movers,
                                   int* __restrict__ count)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 i = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
    const u64 x = in[i];
    const int v = static_cast<int>(x >> 32);
    dest[v]     = static_cast<i64>(static_cast<u32>(x));  // M3 reads the code
    movers[i]   = v;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *count = static_cast<int>(total);
}

// Exchange 2: the owned vertices of ap_in (OR of every partition's Ap) that
// this partition has not appended yet go to their later sub-round's bucket.
__global__ void __launch_bounds__(kBlock) part_append_kernel(
  const i64* __restrict__ indptr, i64 v0, i64 v1, MoveAct t, const u32* __restrict__ ap_in)
{
  const int k       = t.sweep();
  const MoveFlags F = t.flags(k);
  const u64 cm      = t.ctx_move(k);
  const int lane    = threadIdx.x & (kWarp - 1);
  const i64 stride  = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = v0 + static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < v1;
       base += stride) {
    const i64 v = base + lane;
    u32 in = 0, mine = 0;
    if (lane == 0) {
      in   = ap_in[base >> 5];
      mine = F.Ap[base >> 5];
    }
    in      = __shfl_sync(kFullMask, in, 0);
    mine    = __shfl_sync(kFullMask, mine, 0);
    int key = -1;
    if (v < v1 && (((in & ~mine) >> lane) & 1u)) {
      const int su = move_sub_round(cm, static_cast<int>(v), t.S);
      key          = su * kNumClasses + degree_class(indptr[v + 1] - indptr[v], t.th);
    }
    move_bucket_append(t.B, key, static_cast<int>(v));
    if (lane == 0) F.Ap[base >> 5] = in | mine;
  }
}

// The sweep's deferred activations, folded into R(k) before Exchange 3 (one
// partition: M1 of the next sweep): slot != idle and (multi or
// move_activates(P[v], slot)) sets R; the slot becomes idle.
__global__ void __launch_bounds__(kBlock)
  part_fold_kernel(const int* __restrict__ P, i64 n, MoveAct t)
{
  const MoveFlags F = t.flags(t.sweep());
  const int lane    = threadIdx.x & (kWarp - 1);
  const i64 stride  = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 base = static_cast<i64>(blockIdx.x) * blockDim.x + (threadIdx.x & ~(kWarp - 1));
       base < n;
       base += stride) {
    const i64 v = base + lane;
    bool act    = false;
    if (v < n) {
      const int sl = t.slot[v];
      if (sl != kDeferIdle) {
        t.slot[v] = kDeferIdle;
        act       = sl == kDeferMulti || move_activates(P[v], sl);
      }
    }
    const unsigned m = __ballot_sync(kFullMask, act);
    if (lane == 0 && m) F.R[base >> 5] |= m;  // this warp owns the word
  }
}

// out[w] = in[w] | the OR of `others` pulled copies (scratch[j * words + w]).
__global__ void part_or_kernel(
  const u32* in, u32* out, const u32* __restrict__ scratch, int others, i64 words)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 w = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) {
    u32 x = in[w];
    for (int j = 0; j < others; ++j)
      x |= scratch[j * words + w];
    out[w] = x;
  }
}

// A2: coarse ranges [C_q, C_q+1) balanced by member count (member_start, the
// same on every partition; n1 = ctl->n_next) and this partition's partial
// offsets there: out[q] = C_q, out[P + 1 + q] = indptr_next[C_q].
__global__ void part_ranges_kernel(const i64* __restrict__ mstart,
                                   const Control* ctl,
                                   i64 n0,
                                   int P,
                                   const i64* __restrict__ indptr_next,
                                   i64* __restrict__ out)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const i64 n1 = ctl->n_next;
  for (int q = 0; q <= P; ++q) {
    i64 c = n1;
    if (q < P) {  // the first c with mstart[c] >= ceil(q n0 / P)
      const i64 target = (static_cast<i64>(q) * n0 + P - 1) / P;
      i64 lo = 0, hi = n1;
      while (lo < hi) {
        const i64 mid = (lo + hi) / 2;
        if (mstart[mid] < target)
          lo = mid + 1;
        else
          hi = mid;
      }
      c = lo;
    }
    out[q]         = c;
    out[P + 1 + q] = indptr_next[c];
  }
}

// A4: the lengths of the merged rows over [0, n1] (rows outside [c0, c0 +
// len) empty): sums of the P received partial rows (offsets off[p][0..len]).
__global__ void part_merge_len_kernel(
  const i64* __restrict__ off, int P, i64 len, i64 c0, i64 n1, i64* __restrict__ lens)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 c = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; c <= n1; c += stride) {
    i64 L = 0;
    if (c >= c0 && c < c0 + len) {
      const i64 i = c - c0;
      for (int p = 0; p < P; ++p) {
        const i64* o = off + static_cast<i64>(p) * (len + 1);
        L += o[i + 1] - o[i];
      }
    }
    lens[c] = L;
  }
}

// A4: the received partial rows of each coarse row, concatenated in partition
// order (warp per row).
__global__ void part_merge_copy_kernel(const i64* __restrict__ off,
                                       int P,
                                       i64 len,
                                       i64 c0,
                                       const i64* __restrict__ aux_indptr,
                                       const int* __restrict__ ridx,
                                       const i64* __restrict__ rw,
                                       int* __restrict__ aidx,
                                       i64* __restrict__ aw)
{
  const int lane   = threadIdx.x & (kWarp - 1);
  const i64 warp0  = static_cast<i64>(blockIdx.x) * kWarpsPerBlock + threadIdx.x / kWarp;
  const i64 nwarps = static_cast<i64>(gridDim.x) * kWarpsPerBlock;
  for (i64 i = warp0; i < len; i += nwarps) {
    i64 dst = aux_indptr[c0 + i], blk = 0;
    for (int p = 0; p < P; ++p) {
      const i64* o = off + static_cast<i64>(p) * (len + 1);
      const i64 b = blk + o[i] - o[0], m = o[i + 1] - o[i];
      for (i64 t = lane; t < m; t += kWarp) {
        aidx[dst + t] = ridx[b + t];
        aw[dst + t]   = rw[b + t];
      }
      dst += m;
      blk += o[len] - o[0];
    }
  }
}

// The forest of another partition's rows (link: its flattened parents).
__global__ void cc_merge_kernel(const int* __restrict__ link, i64 n, int* parent)
{
  const i64 stride = static_cast<i64>(gridDim.x) * blockDim.x;
  for (i64 v = static_cast<i64>(blockIdx.x) * blockDim.x + threadIdx.x; v < n; v += stride) {
    const int l = link[v];
    if (l != v) cc_union(static_cast<int>(v), l, parent);
  }
}

__global__ void drv_set_i64_kernel(i64* p, i64 v) { *p = v; }

// One partition: its rows, workspace, stream and level-0 state.
struct Part {
  int global     = 0;  // partition index (0: root)
  int device     = 0;
  cudaStream_t s = nullptr;
  void* pinned   = nullptr;  // kPinnedBytes for readbacks
  Layout L;
  i64 v0 = 0, v1 = 0;  // owned rows
  i64 nnz = 0;         // stored entries of the owned rows
  QuantizeResult q;
  LevelGraph g0;  // partition CSR: full-length indptr, global ids
  MoveLaunch ml;
  MoveRun mr;
  std::vector<KernelCall> calls;  // refinement R1 - R8
  std::size_t ends[3] = {};
  int* cmap0          = nullptr;
  std::size_t mark    = 0;  // arena bottom below cmap0, and above it
  std::size_t base    = 0;
  // contraction: dense partial rows, received partials, merged rows
  i64 e = 0, f = 0;
  i64 ranges[2 * (kMaxParts + 1)] = {};
  int* pidx                       = nullptr;
  i64* pw                         = nullptr;
  i64* roff                       = nullptr;
  int* ridx                       = nullptr;
  i64* rw                         = nullptr;
  int* fidx                       = nullptr;
  float* fw                       = nullptr;
};

// The input of a partitioned run: the ingest check of the whole graph, one
// seed per resolution, the labels ([resolutions][n], written on the root).
struct DriverInput {
  IngestInfo info;
  std::vector<double> gammas;
  std::vector<u32> seeds;
  int* labels_out = nullptr;
};

struct DriverResult {
  std::vector<ResolutionResult> res;
};

// The partitioned level 0 of every resolution; levels >= 1 run on the root.
class PartitionedDriver {
 public:
  using Quantize = std::function<QuantizeResult(Part&, double)>;

  PartitionedDriver(
    std::vector<Part>& parts, Exchange& ex, const DriverParams& dp, i64 n, std::vector<i64> bounds)
    : parts_(parts), ex_(ex), dp_(dp), n_(n), bounds_(std::move(bounds))
  {
    for (Part& p : parts_)
      if (p.global == 0) root_ = &p;
    if (root_) {
      use(*root_);
      root_drv_ = std::make_unique<LeidenDriver>(root_->L, dp_, root_->s);
    }
  }

  // Every resolution; false (on every partition) if an arena block was not allocated (failed in
  // that partition's arena).
  bool run(const DriverInput& in, const Quantize& quantize, DriverResult& out)
  {
    int s_prev = INT_MIN;
    for (std::size_t i = 0; i < in.gammas.size(); ++i) {
      const double gamma = in.gammas[i];
      int* labels        = in.labels_out ? in.labels_out + static_cast<i64>(i) * n_ : nullptr;
      ResolutionResult r;
      const int sg = scale_for_gamma(scale_s0(in.info.n_counted, in.info.wmax), gamma);
      if (in.info.n_counted > 0 && sg != s_prev) {
        set_level0(gamma, quantize);
        s_prev = sg;
      }
      if (in.info.n_counted == 0 || parts_[0].q.two_m_hat <= 0) {
        r.n_clusters = n_;  // no counted entry: singletons
        if (root_ && labels) {
          use(*root_);
          drv_iota_kernel<<<grid_items(n_), kBlock, 0, root_->s>>>(labels, n_);
          RAFT_CHECK_CUDA(root_->s);
        }
      } else if (!run_resolution(gamma, in.seeds[i], labels, r)) {
        return false;
      }
      out.res.push_back(std::move(r));
    }
    return true;
  }

 private:
  struct PhaseResult {
    MoveStats st;
    i64 C = 0;
  };

  std::vector<Part>& parts_;
  Exchange& ex_;
  DriverParams dp_;
  i64 n_ = 0;
  std::vector<i64> bounds_;  // owned rows of each partition
  Part* root_ = nullptr;     // the local root partition, if any
  std::unique_ptr<LeidenDriver> root_drv_;
  double lam_ = 0.0, gamma_ = 0.0;
  u32 seed_ = 0;
  u64 s64_  = 0;

  static void use(const Part& p)
  {
    int cur = -1;
    RAFT_CUDA_TRY(cudaGetDevice(&cur));
    if (cur != p.device) RAFT_CUDA_TRY(cudaSetDevice(p.device));
  }
  template <typename F>
  void each(F&& f)
  {
    for (Part& p : parts_) {
      use(p);
      f(p);
    }
  }
  template <typename F>
  std::vector<void*> ptrs(F&& f)
  {
    std::vector<void*> v;
    for (Part& p : parts_)
      v.push_back(static_cast<void*>(f(p)));
    return v;
  }
  Part* local(int global)
  {
    for (Part& p : parts_)
      if (p.global == global) return &p;
    return nullptr;
  }
  // one value per local partition -> the values of all P partitions
  template <typename F>
  std::vector<i64> all_values(F&& f)
  {
    std::vector<i64> in, out(static_cast<std::size_t>(ex_.parts()));
    for (Part& p : parts_)
      in.push_back(f(p));
    ex_.allgather_host(in.data(), 1, out.data());
    return out;
  }
  template <typename F>
  i64 sum_values(F&& f)
  {
    i64 t = 0;
    for (i64 x : all_values(f))
      t += x;
    return t;
  }
  // true on every partition iff ok holds on every partition
  template <typename F>
  bool all_ok(F&& ok)
  {
    return sum_values([&](Part& p) { return ok(p) ? i64{0} : i64{1}; }) == 0;
  }
  // the root's host value on every partition
  i64 from_root(i64 v)
  {
    return all_values([&](Part& p) { return p.global == 0 ? v : 0; })[0];
  }

  Control read(Part& p)
  {
    use(p);
    const Control c = read_control(p.L.ctl, p.s, p.pinned);
    check_device_flags(c.flags);
    return c;
  }
  // k i64 device values of every local partition (through the pinned
  // scratch); one sync per partition
  template <typename F>
  void read_values(int k, F&& src, std::vector<std::vector<i64>>& out)
  {
    out.assign(parts_.size(), std::vector<i64>(static_cast<std::size_t>(k)));
    each([&](Part& p) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(p.pinned) + kScratchOffset,
                                    src(p),
                                    k * sizeof(i64),
                                    cudaMemcpyDeviceToHost,
                                    p.s));
    });
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p = parts_[i];
      use(p);
      RAFT_CUDA_TRY(cudaStreamSynchronize(p.s));
      std::memcpy(out[i].data(), static_cast<char*>(p.pinned) + kScratchOffset, k * sizeof(i64));
    }
  }

  // I2 for the scale of gamma, then the k_hat allgather and the 2m_hat sum.
  void set_level0(double gamma, const Quantize& quantize)
  {
    each([&](Part& p) { p.q = quantize(p, gamma); });
    std::vector<void*> kh = ptrs([](Part& p) { return p.L.l0.khat; });
    ex_.allgather_slices(kh.data(), sizeof(i64), bounds_.data());
    const i64 two_m = sum_values([](Part& p) { return p.q.two_m_hat; });
    for (Part& p : parts_) {
      p.q.two_m_hat = two_m;
      LevelGraph& g = p.g0;
      g             = LevelGraph{};
      g.indptr      = p.q.indptr;
      g.indices     = p.q.indices;
      g.wf          = p.q.wkind == WKind::F32 ? p.q.wf32 : nullptr;
      g.wq          = p.q.wkind == WKind::I64 ? p.q.wq : nullptr;
      g.unit        = p.q.wkind == WKind::UNIT ? p.q.unit_q : 1;
      g.scale       = std::ldexp(1.0, p.q.s);
      g.khat        = p.L.l0.khat;
      g.n           = n_;
      g.nnz         = p.nnz;
      for (int c = 0; c < kNumClasses; ++c)
        g.cls[c] = p.q.class_count[c];
    }
    if (root_) root_drv_->set_level0(root_->q, n_, root_->nnz);
  }

  bool run_resolution(double gamma, u32 seed, int* labels_out, ResolutionResult& out)
  {
    lam_   = gamma / static_cast<double>(parts_[0].q.two_m_hat);
    gamma_ = gamma;
    seed_  = seed;
    s64_   = seed64(seed);
    if (root_) {
      use(*root_);
      root_drv_->set_resolution(gamma, seed);
    }
    each([&](Part& p) {
      drv_iota_kernel<<<grid_items(n_), kBlock, 0, p.s>>>(p.L.persist.P0, n_);
      RAFT_CHECK_CUDA(p.s);
    });
    i64 C         = n_;  // id bound of P0
    const int T   = dp_.n_iterations;
    const int V   = dp_.vcycle_sweeps;
    int it        = 0;
    double q_prev = -std::numeric_limits<double>::infinity();
    bool stop     = false;
    while (true) {
      const bool final  = iteration_final(T, it, stop);
      const bool vcycle = V > 0 && (!final || T == 1 || dp_.vcycle_last);
      IterationStat stat;
      i64 Cb = 0;
      if (!iteration(C, it, vcycle, stat, Cb)) return false;
      C = cc_split([](Part& p) { return p.L.persist.P0; });
      ++it;
      if (!final && T == -1) {  // is the next iteration final?
        const double q = quality(C);
        stop           = until_stable_done(stat.moved, q, q_prev, it, dp_.max_iter_until_stable);
        q_prev         = q;
      }
      out.iters.push_back(std::move(stat));
      if (final) break;
    }
    out.modularity = quality(C);
    if (root_ && labels_out) {
      use(*root_);
      const Layout& L = root_->L;
      run_rank_labels(
        L.persist.P0, n_, C, labels_out, L.final_, L.cub, L.cub_bytes, L.ctl, root_->s);
      read(*root_);
    }
    out.n_clusters = C;
    return true;
  }

  // One iteration (its level 0 partitioned, levels >= 1 on the root); false
  // if an arena block was not allocated. C_out: communities before the CC
  // split.
  bool iteration(i64 id_bound, int it, bool vcycle, IterationStat& stat, i64& C_out)
  {
    each([](Part& p) { p.L.arena.reset(); });
    const u32 uit = static_cast<u32>(it);
    bool moved    = false;
    u32 phase     = kPhaseDown;
    int reentry   = 0;
    u32 top_done  = 0;  // TOP sweeps run at level 0 (cumulative)
    i64 idb       = id_bound;
    while (true) {
      const bool top = phase == kPhaseTop;
      const PhaseResult mv =
        local_move(phase, top ? dp_.top_sweeps : dp_.move_sweeps, top ? top_done : 0, idb, uit);
      const i64 C = mv.C;
      if (top) top_done += static_cast<u32>(mv.st.sweeps);
      moved          = moved || mv.st.moves > 0;
      idb            = C;
      LevelStep step = level_step(phase, C, n_, -1, -1, reentry, 0, dp_);
      if (step == LevelStep::kVerifyTop) {
        phase = kPhaseTop;
        continue;
      }
      if (step == LevelStep::kCertified) {
        ++stat.levels;
        break;
      }
      if (!push_cmap0()) return false;  // refinement (C is known)
      const i64 n_next = refine(C, uit);
      i64 n1           = n_next;
      step             = level_step(phase, C, n_, n_next, -1, reentry, 0, dp_);
      if (step == LevelStep::kReenterTop || step == LevelStep::kPieces) {
        each([](Part& p) { p.L.arena.restore(p.mark); });
        if (step == LevelStep::kReenterTop) {
          ++reentry;
          phase = kPhaseTop;
          continue;
        }
        if (!push_cmap0()) return false;
        each([&](Part& p) {
          RAFT_CUDA_TRY(cudaMemcpyAsync(
            p.cmap0, p.L.persist.P0, n_ * sizeof(int), cudaMemcpyDeviceToDevice, p.s));
        });
        const i64 pieces = cc_split([](Part& p) { return p.cmap0; });
        step             = level_step(phase, C, n_, n_next, pieces, reentry, 0, dp_);
        if (step == LevelStep::kOutputPieces) {  // uncertified
          each([&](Part& p) {
            RAFT_CUDA_TRY(cudaMemcpyAsync(
              p.L.persist.P0, p.cmap0, n_ * sizeof(int), cudaMemcpyDeviceToDevice, p.s));
            p.L.arena.restore(p.mark);
          });
          idb = n_;
          ++stat.levels;
          break;
        }
        each([&](Part& p) {
          agg_set_n_next_kernel<<<1, 1, 0, p.s>>>(p.L.ctl, pieces);
          RAFT_CHECK_CUDA(p.s);
        });
        n1 = pieces;
      }
      LevelGraph g1;
      if (!aggregate(n1, g1)) return false;
      ++stat.levels;
      i64 Ccur      = idb;
      bool up_moved = false;
      int ok        = 1;
      if (step == LevelStep::kAggregateStop) {
        ++stat.levels;
      } else if (root_) {  // levels >= 1, projected down to level 1
        use(*root_);
        LevelGraph g0 = root_->g0;
        g0.cmap       = root_->cmap0;
        ok = root_drv_->upper_levels(g0, g1, idb, it, vcycle, stat, Ccur, up_moved) ? 1 : 0;
      }
      if (!from_root(ok)) return false;
      Ccur  = from_root(Ccur);
      moved = from_root(up_moved ? 1 : 0) != 0 || moved;
      // projection to level 0 and its V-cycle UP sweeps
      std::vector<void*> P1 = ptrs([](Part& p) { return p.L.persist.Pa; });
      ex_.bcast(0, P1.data(), n1 * static_cast<i64>(sizeof(int)));
      each([&](Part& p) { run_project(p.L.persist.Pa, p.cmap0, n_, p.L.persist.P0, p.s); });
      if (vcycle) {
        const PhaseResult up = local_move(kPhaseUp, dp_.vcycle_sweeps, 0, Ccur, uit);
        Ccur                 = up.C;
        moved                = moved || up.st.moves > 0;
      }
      stat.moved = moved;
      C_out      = Ccur;
      return true;
    }
    stat.moved = moved;  // the level loop ended at level 0
    C_out      = idb;
    return true;
  }

  bool push_cmap0()
  {
    each([&](Part& p) {
      p.mark  = p.L.arena.bottom;
      p.cmap0 = p.L.arena.push_bottom<int>(static_cast<std::size_t>(n_));
    });
    return all_ok([](Part& p) { return p.cmap0 != nullptr; });
  }

  // LOCAL_MOVE + COMPACT of level 0: one host loop over the sweeps and
  // sub-rounds, with the exchanges between the launches.
  PhaseResult local_move(u32 phase, int cap, u32 sweep0, i64 id_bound, u32 it)
  {
    const bool top = phase == kPhaseTop;
    const int S    = dp_.subrounds;
    MoveParams prm;
    prm.lam      = lam_;
    prm.seed     = seed_;
    prm.it       = it;
    prm.level    = 0;
    prm.phase    = phase;
    prm.sweep0   = sweep0;
    prm.cap      = cap;
    prm.S        = S;
    prm.th       = dp_.th;
    prm.id_bound = id_bound;
    each([&](Part& p) {
      p.ml.graph = nullptr;
      p.ml.loop  = false;
      p.g0.with_w([&](auto w) {
        p.mr = prepare_local_move(p.g0.indptr,
                                  p.g0.indices,
                                  w,
                                  p.g0.khat,
                                  p.L.persist.P0,
                                  n_,
                                  p.g0.cls,
                                  prm,
                                  p.L.move,
                                  p.L.ctl,
                                  p.ml,
                                  p.s,
                                  p.v0,
                                  p.v1);
      });
    });
    const i64 words = (n_ + 31) / 32;
    auto launch     = [](Part& p, int i) { p.ml.calls[static_cast<std::size_t>(i)].launch(p.s); };
    PhaseResult res;
    for (int k = 0; k < cap; ++k) {
      each([&](Part& p) {
        const MoveAct& t = p.ml.act;
        // M1 clears the owned words of R(k) and Ap, this the others
        RAFT_CUDA_TRY(cudaMemsetAsync((k & 1) ? t.bm1 : t.bm0, 0, words * sizeof(u32), p.s));
        RAFT_CUDA_TRY(cudaMemsetAsync(t.ap, 0, words * sizeof(u32), p.s));
        move_chunk_kernel<<<1, kBlock, 0, p.s>>>(p.mr.state, k, k + 1, p.L.ctl);
        RAFT_CHECK_CUDA(p.s);
        launch(p, 0);  // bucket counts
        launch(p, 1);  // M1 of the owned rows
      });
      i64 moves = 0;
      for (int r = 0; r < S; ++r) {
        const int c0 = kSweepHead + kSweepPerRound * r;
        each([&](Part& p) {
          for (int i = 0; i < 3; ++i)
            launch(p, c0 + i);  // M2
        });
        const std::vector<i64> counts = mover_counts(r);
        i64 total                     = 0;
        for (i64 x : counts)
          total += x;
        if (total == 0) continue;
        exchange_movers(counts, total, r);
        if (!top && k > 0 && r + 1 < S) exchange_activations(words);
        each([&](Part& p) { launch(p, c0 + 3); });  // M3, global list
        moves += total;
      }
      res.st.moves += moves;
      res.st.last = moves;
      ++res.st.sweeps;
      if (moves == 0) break;      // converged (TOP: certified)
      if (!top && k + 1 < cap) {  // deferred activations; A(k+1) = R(k)
        each([&](Part& p) {
          part_fold_kernel<<<grid_items(n_), kBlock, 0, p.s>>>(p.L.persist.P0, n_, p.ml.act);
          RAFT_CHECK_CUDA(p.s);
        });
        std::vector<u32*> R, X;
        for (Part& p : parts_) {
          R.push_back((k & 1) ? p.ml.act.bm1 : p.ml.act.bm0);
          X.push_back(reinterpret_cast<u32*>(p.L.move.xbuf));
        }
        ex_.allreduce_or(R.data(), R.data(), X.data(), words);
      }
    }
    each([&](Part& p) {
      launch_compact(
        p.L.persist.P0, n_, p.L.move, p.L.persist.KS, p.L.cub, p.L.cub_bytes, p.L.ctl, p.s);
    });
    int err = 0;
    for (Part& p : parts_) {  // SL1: C is the same everywhere
      const Control c = read(p);
      err |= c.move_err;
      res.C = c.move_C;
    }
    for (i64 e : all_values([&](Part&) { return static_cast<i64>(err); }))
      err |= static_cast<int>(e);
    if (err) {  // every partition throws
      MoveRun none;
      Control h{};
      h.move_err = err;
      move_fold_chunk(none, h);
    }
    return res;
  }

  // The movers of sub-round r on every partition (one sync each).
  std::vector<i64> mover_counts(int r)
  {
    each([&](Part& p) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(static_cast<char*>(p.pinned) + kScratchOffset,
                                    &p.L.ctl->sweep_movers[0][r],
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    p.s));
    });
    return all_values([](Part& p) {
      use(p);
      RAFT_CUDA_TRY(cudaStreamSynchronize(p.s));
      int c = 0;
      std::memcpy(&c, static_cast<char*>(p.pinned) + kScratchOffset, sizeof(int));
      return static_cast<i64>(c);
    });
  }

  // Exchange 1: every partition's movers (vertex, DECIDE code) to all.
  void exchange_movers(const std::vector<i64>& counts, i64 total, int r)
  {
    std::vector<i64> displ(counts.size(), 0);
    for (std::size_t q = 1; q < counts.size(); ++q)
      displ[q] = displ[q - 1] + counts[q - 1];
    each([&](Part& p) {
      const i64 c = counts[static_cast<std::size_t>(p.global)];
      if (c == 0) return;
      part_pack_kernel<<<grid_items(c), kBlock, 0, p.s>>>(
        p.L.move.movers,
        c,
        p.L.move.dest,
        p.L.move.xbuf + displ[static_cast<std::size_t>(p.global)]);
      RAFT_CHECK_CUDA(p.s);
    });
    std::vector<void*> X = ptrs([](Part& p) { return p.L.move.xbuf; });
    ex_.allgatherv(X.data(), sizeof(u64), counts.data());
    each([&](Part& p) {
      part_unpack_kernel<<<grid_items(total), kBlock, 0, p.s>>>(
        p.L.move.xbuf, total, p.L.move.dest, p.L.move.movers, &p.L.ctl->sweep_movers[0][r]);
      RAFT_CHECK_CUDA(p.s);
    });
  }

  // Exchange 2: activations of vertices for later sub-rounds by movers of
  // other partitions (the OR of every Ap); each owner appends the new ones.
  void exchange_activations(i64 words)
  {
    std::vector<u32*> Ap, ap_in, X;  // xbuf: ap_in, then the OR scratch
    for (Part& p : parts_) {
      u32* x = reinterpret_cast<u32*>(p.L.move.xbuf);
      Ap.push_back(p.ml.act.ap);
      ap_in.push_back(x);
      X.push_back(x + words);
    }
    ex_.allreduce_or(Ap.data(), ap_in.data(), X.data(), words);
    each([&](Part& p) {
      if (p.v1 <= p.v0) return;
      part_append_kernel<<<grid_items(p.v1 - p.v0), kBlock, 0, p.s>>>(
        p.g0.indptr, p.v0, p.v1, p.ml.act, reinterpret_cast<const u32*>(p.L.move.xbuf));
      RAFT_CHECK_CUDA(p.s);
    });
  }

  // PLEIDEN_R of level 0: the row kernels on every partition, their values
  // allgathered, the rest replicated; returns n_next (cmap0 written).
  i64 refine(i64 C, u32 it)
  {
    RefineParams rp;
    rp.lam       = lam_;
    rp.ctx_order = ctx(s64_, kTagOrder, it, 0, 0, 0);
    rp.trees     = dp_.trees;
    auto run     = [&](std::size_t lo, std::size_t hi) {
      each([&](Part& p) { launch_calls(p.calls.data() + lo, static_cast<int>(hi - lo), p.s); });
    };
    each([&](Part& p) {
      p.g0.with_w([&](auto w) {
        p.calls = refine_calls(p.g0.indptr,
                               p.g0.indices,
                               w,
                               p.g0.khat,
                               p.L.persist.P0,
                               p.L.persist.KS,
                               n_,
                               C,
                               rp,
                               p.L.refine,
                               p.L.cub,
                               p.L.cub_bytes,
                               p.L.ctl,
                               nullptr,
                               {},
                               p.ends);
      });
    });
    const std::size_t* ends = parts_[0].ends;  // the same everywhere
    const i64* b            = bounds_.data();
    run(0, ends[0]);  // R1: ext, U and SU (in cid)
    std::vector<void*> su  = ptrs([](Part& p) { return p.L.refine.cid; });
    std::vector<void*> ext = ptrs([](Part& p) { return p.L.refine.ext; });
    ex_.allgather_slices(su.data(), sizeof(int), b);
    ex_.allgather_slices(ext.data(), sizeof(i64), b);
    run(ends[0], ends[1]);  // R2: hn, root
    std::vector<void*> hn = ptrs([](Part& p) { return p.L.refine.hn; });
    std::vector<void*> rt = ptrs([](Part& p) { return p.L.refine.root; });
    ex_.allgather_slices(hn.data(), sizeof(int), b);
    ex_.allgather_slices(rt.data(), sizeof(int), b);
    run(ends[1], ends[2]);  // R3 - R7
    std::vector<void*> in = ptrs([](Part& p) { return p.L.refine.inner; });
    ex_.allgather_slices(in.data(), sizeof(i64), b);
    each([&](Part& p) {  // R8, R9 (replicated)
      launch_calls(p.calls.data() + ends[2], static_cast<int>(p.calls.size() - ends[2]), p.s);
      refine_large(p.g0.khat,
                   p.L.persist.P0,
                   p.L.persist.KS,
                   n_,
                   rp,
                   p.L.refine,
                   p.L.cub,
                   p.L.cub_bytes,
                   p.L.ctl,
                   p.s);
      const std::vector<KernelCall> r9 =
        refine_r9_calls(n_, p.L.refine, p.cmap0, p.L.cub, p.L.cub_bytes, p.L.ctl);
      launch_calls(r9.data(), static_cast<int>(r9.size()), p.s);
    });
    i64 n_next = 0;
    for (Part& p : parts_)
      n_next = read(p).n_next;  // the same everywhere
    return n_next;
  }

  // Contraction of level 0 by cmap0 into level 1 (n1 vertices) on the root:
  // A1 partial rows (exact int64 sums) of every partition's rows, A2 coarse
  // row ranges, A3 the partials of each range to its owner, A4 the owner's
  // merge (the contraction of the interleaved partials by the identity: one
  // fp32 rounding of each exact sum), A5 the merged rows to the root. False
  // (on every partition) if an arena block was not allocated.
  bool aggregate(i64 n1, LevelGraph& g1)
  {
    const int P          = ex_.parts();
    const std::size_t P1 = static_cast<std::size_t>(P) + 1;
    AggregateOptions opt;
    opt.gather = dp_.gather;
    opt.move   = dp_.th;
    each([&](Part& p) {  // A1, A2
      p.base                  = p.L.arena.bottom;
      const AggregateBufs& ab = p.L.aggregate;
      p.g0.with_w([&](auto w) {
        constexpr WKind KW      = MoveWKindOf<decltype(w)>::value;
        AggregatePlan<i64> plan = aggregate_begin<KW, i64>(p.g0.indptr,
                                                           p.g0.indices,
                                                           w,
                                                           p.g0.khat,
                                                           n_,
                                                           p.nnz,
                                                           p.cmap0,
                                                           opt,
                                                           ab,
                                                           p.L.arena,
                                                           p.L.cub,
                                                           p.L.cub_bytes,
                                                           p.L.ctl,
                                                           p.s);
        p.pw                    = plan.holey ? plan.h_w : nullptr;
        p.pidx                  = plan.holey ? plan.h_idx : nullptr;
      });
      part_ranges_kernel<<<1, 1, 0, p.s>>>(
        ab.member_start, p.L.ctl, n_, P, ab.indptr_next, ctl_scratch(p.L.ctl));
      RAFT_CHECK_CUDA(p.s);
    });
    std::vector<std::vector<i64>> rv;
    read_values(2 * P + 2, [](Part& p) { return ctl_scratch(p.L.ctl); }, rv);
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p = parts_[i];
      p.e     = read(p).nnz_next;
      std::copy(rv[i].begin(), rv[i].end(), p.ranges);
    }
    if (!all_ok([](Part& p) { return p.pidx != nullptr; })) return false;
    std::vector<i64> C(parts_[0].ranges, parts_[0].ranges + P1);
    // the holey partial rows, dense at the arena bottom
    each([&](Part& p) {
      const int* h_idx    = p.pidx;
      const i64* h_w      = p.pw;
      const std::size_t e = static_cast<std::size_t>(p.e);
      p.L.arena.reserve_bottom(align_up(4 * e) + align_up(8 * e));
      p.pidx = p.L.arena.push_bottom<int>(e);
      p.pw   = p.L.arena.push_bottom<i64>(e);
      p.L.arena.reserve_bottom(0);
      if (!p.pidx || !p.pw) return;
      const AggregateBufs& ab = p.L.aggregate;
      if (n1 > 0) {
        agg_compact_kernel<i64><<<grid_rows(n1), kBlock, 0, p.s>>>(
          ab.window_offset, ab.cdeg, ab.indptr_next, n1, h_idx, h_w, p.pidx, p.pw);
        RAFT_CHECK_CUDA(p.s);
      }
      p.L.arena.release_top();
    });
    if (!all_ok([](Part& p) { return p.pidx && p.pw; })) return false;
    // A3: E[p P + q] = entries of partition p's rows in range q
    std::vector<i64> mine, E(static_cast<std::size_t>(P) * P);
    for (Part& p : parts_)
      for (int q = 0; q < P; ++q)
        mine.push_back(p.ranges[P1 + q + 1] - p.ranges[P1 + q]);
    ex_.allgather_host(mine.data(), P, E.data());
    auto got = [&](int q) {  // entries received by owner q
      i64 t = 0;
      for (int p = 0; p < P; ++p)
        t += E[static_cast<std::size_t>(p) * P + q];
      return t;
    };
    each([&](Part& p) {
      const int q   = p.global;
      const i64 len = C[q + 1] - C[q], tot = got(q);
      const std::size_t no = static_cast<std::size_t>(P) * (len + 1),
                        nt = static_cast<std::size_t>(tot);
      p.L.arena.reserve_top(align_up(8 * no) + align_up(4 * nt) + align_up(8 * nt));
      p.roff = p.L.arena.push_top<i64>(no);
      p.ridx = p.L.arena.push_top<int>(nt);
      p.rw   = p.L.arena.push_top<i64>(nt);
      p.L.arena.reserve_top(0);
    });
    if (!all_ok([](Part& p) { return p.roff && p.ridx && p.rw; })) return false;
    for (int q = 0; q < P; ++q) {
      const i64 len = C[q + 1] - C[q];
      std::vector<i64> n_off(static_cast<std::size_t>(P), len + 1), n_e;
      for (int p = 0; p < P; ++p)
        n_e.push_back(E[static_cast<std::size_t>(p) * P + q]);
      std::vector<const void*> s_off, s_idx, s_w;
      for (Part& p : parts_) {
        const i64 o = p.ranges[P1 + q];
        s_off.push_back(p.L.aggregate.indptr_next + C[q]);
        s_idx.push_back(p.pidx + o);
        s_w.push_back(p.pw + o);
      }
      Part* owner = local(q);
      ex_.gatherv(q, s_off.data(), sizeof(i64), n_off.data(), owner ? owner->roff : nullptr);
      ex_.gatherv(q, s_idx.data(), sizeof(int), n_e.data(), owner ? owner->ridx : nullptr);
      ex_.gatherv(q, s_w.data(), sizeof(i64), n_e.data(), owner ? owner->rw : nullptr);
    }
    // A4: the owner's aux rows (interleaved partials) at the arena bottom
    // (the sent partials are free), contracted by the identity
    std::vector<i64*> aux_ip(parts_.size()), aux_kh(parts_.size());
    std::vector<int*> aux_idx(parts_.size()), iota(parts_.size());
    std::vector<i64*> aux_w(parts_.size());
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p               = parts_[i];
      const std::size_t tot = static_cast<std::size_t>(got(p.global));
      const std::size_t m   = static_cast<std::size_t>(n1);
      LevelArena& a         = p.L.arena;
      a.restore(p.base);
      a.reserve_bottom(align_up(8 * (m + 1)) + align_up(4 * tot) + align_up(8 * tot) +
                       align_up(4 * m) + align_up(8 * m));
      aux_ip[i]  = a.push_bottom<i64>(m + 1);
      aux_idx[i] = a.push_bottom<int>(tot);
      aux_w[i]   = a.push_bottom<i64>(tot);
      iota[i]    = a.push_bottom<int>(m);
      aux_kh[i]  = a.push_bottom<i64>(m);
      a.reserve_bottom(0);
      p.fidx = (aux_ip[i] && aux_idx[i] && aux_w[i] && iota[i] && aux_kh[i]) ? aux_idx[i] : nullptr;
    }
    if (!all_ok([](Part& p) { return p.fidx != nullptr; })) return false;
    std::vector<AggregatePlan<float>> plans(parts_.size());
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p = parts_[i];
      use(p);
      const int q   = p.global;
      const i64 len = C[q + 1] - C[q], tot = got(q);
      const AggregateBufs& ab = p.L.aggregate;
      part_merge_len_kernel<<<grid_items(n1 + 1), kBlock, 0, p.s>>>(
        p.roff, P, len, C[q], n1, ab.window);
      RAFT_CHECK_CUDA(p.s);
      KernelCall sc[3];
      scan_calls(sc, ScanArray<i64, i64>{ab.window}, aux_ip[i], n1 + 1, p.L.cub, p.L.cub_bytes);
      launch_calls(sc, 3, p.s);
      if (len > 0) {
        part_merge_copy_kernel<<<grid_rows(len), kBlock, 0, p.s>>>(
          p.roff, P, len, C[q], aux_ip[i], p.ridx, p.rw, aux_idx[i], aux_w[i]);
        RAFT_CHECK_CUDA(p.s);
      }
      p.L.arena.release_top();  // the received partials
      drv_iota_kernel<<<grid_items(n1), kBlock, 0, p.s>>>(iota[i], n1);
      RAFT_CHECK_CUDA(p.s);
      agg_set_n_next_kernel<<<1, 1, 0, p.s>>>(p.L.ctl, n1);
      RAFT_CHECK_CUDA(p.s);
      // the identity's k_hat is khat_next (A1, complete, kept for
      // level 1); its own coarse k_hat (the same) goes to aux_kh
      AggregateBufs ib = ab;
      ib.khat_next     = aux_kh[i];
      plans[i]         = aggregate_begin<WKind::I64, float>(aux_ip[i],
                                                    aux_idx[i],
                                                    EdgeW<WKind::I64>{aux_w[i]},
                                                    ab.khat_next,
                                                    n1,
                                                    tot,
                                                    iota[i],
                                                    opt,
                                                    ib,
                                                    p.L.arena,
                                                    p.L.cub,
                                                    p.L.cub_bytes,
                                                    p.L.ctl,
                                                    p.s);
    }
    for (std::size_t i = 0; i < parts_.size(); ++i)
      parts_[i].f = plans[i].holey ? read(parts_[i]).nnz_next : -1;
    if (!all_ok([](Part& p) { return p.f >= 0; })) return false;
    // A5: the merged rows to the root, straight into level 1 there
    const std::vector<i64> F = all_values([](Part& p) { return p.f; });
    i64 nnz1                 = 0;
    for (i64 x : F)
      nnz1 += x;
    CoarseLevel lv;
    lv.n   = n1;
    lv.nnz = nnz1;
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p             = parts_[i];
      LevelArena& a       = p.L.arena;
      const std::size_t m = static_cast<std::size_t>(n1);
      a.restore(p.base);    // the aux rows are merged
      if (p.global == 0) {  // level 1, with room for its cmap
        const std::size_t e = static_cast<std::size_t>(nnz1);
        a.reserve_bottom(align_up(8 * (m + 1)) + align_up(8 * m) + 2 * align_up(4 * e) +
                         align_up(4 * m) + kAlign);
        lv.indptr     = a.push_bottom<i64>(m + 1);
        lv.khat       = a.push_bottom<i64>(m);
        lv.indices    = a.push_bottom<int>(e);
        lv.weights    = a.push_bottom<float>(e);
        const bool ok = lv.indptr && lv.khat && lv.indices && lv.weights;
        p.fidx        = ok ? lv.indices : nullptr;
        p.fw          = ok ? lv.weights : nullptr;
      } else {
        const std::size_t f = static_cast<std::size_t>(p.f);
        a.reserve_bottom(2 * align_up(4 * f));
        p.fidx = a.push_bottom<int>(f);
        p.fw   = a.push_bottom<float>(f);
      }
      a.reserve_bottom(0);
    }
    if (!all_ok([](Part& p) { return p.fidx && p.fw; })) return false;
    for (std::size_t i = 0; i < parts_.size(); ++i) {
      Part& p = parts_[i];
      use(p);
      const AggregateBufs& ab = p.L.aggregate;
      if (n1 > 0) {
        agg_compact_kernel<float><<<grid_rows(n1), kBlock, 0, p.s>>>(ab.window_offset,
                                                                     ab.cdeg,
                                                                     ab.indptr_next,
                                                                     n1,
                                                                     plans[i].h_idx,
                                                                     plans[i].h_w,
                                                                     p.fidx,
                                                                     p.fw);
        RAFT_CHECK_CUDA(p.s);
      }
      p.L.arena.release_top();
    }
    std::vector<i64> lens;
    for (int q = 0; q < P; ++q)
      lens.push_back(C[q + 1] - C[q]);
    std::vector<const void*> s_deg, s_idx, s_w;
    for (Part& p : parts_) {
      s_deg.push_back(p.L.aggregate.cdeg + C[p.global]);
      s_idx.push_back(p.fidx);
      s_w.push_back(p.fw);
    }
    ex_.gatherv(
      0, s_deg.data(), sizeof(i64), lens.data(), root_ ? root_->L.aggregate.cdeg : nullptr);
    ex_.gatherv(0, s_idx.data(), sizeof(int), F.data(), lv.indices);
    ex_.gatherv(0, s_w.data(), sizeof(float), F.data(), lv.weights);
    for (Part& p : parts_)
      if (p.global != 0) p.L.arena.restore(p.base);
    if (!root_) return true;
    // root: indptr_1, k_hat_1 (A1's khat_next), the classes, P_1
    Part& r = *root_;
    use(r);
    const AggregateBufs& ab = r.L.aggregate;
    KernelCall sc[3];
    scan_calls(sc, ScanArray<i64, i64>{ab.cdeg}, ab.indptr_next, n1 + 1, r.L.cub, r.L.cub_bytes);
    launch_calls(sc, 3, r.s);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(r.L.ctl->coarse_class_count, 0, sizeof(r.L.ctl->coarse_class_count), r.s));
    agg_stats_kernel<<<grid_items(n1), kBlock, 0, r.s>>>(ab.cdeg, ab.indptr_next, dp_.th, r.L.ctl);
    RAFT_CHECK_CUDA(r.s);
    agg_finish_kernel<<<grid_items((n_ > n1 ? n_ : n1) + 1), kBlock, 0, r.s>>>(n_,
                                                                               n1,
                                                                               r.cmap0,
                                                                               r.L.persist.P0,
                                                                               r.L.persist.Pa,
                                                                               ab.indptr_next,
                                                                               ab.khat_next,
                                                                               lv.indptr,
                                                                               lv.khat);
    RAFT_CHECK_CUDA(r.s);
    const Control c = read(r);
    CUGRAPH_EXPECTS(c.nnz_next == nnz1 && c.n_next == n1,
                    "leiden: internal error: partitioned contraction mismatch.");
    g1         = LevelGraph{};
    g1.indptr  = lv.indptr;
    g1.indices = lv.indices;
    g1.wf      = lv.weights;
    g1.khat    = lv.khat;
    g1.n       = n1;
    g1.nnz     = nnz1;
    for (int k = 0; k < kNumClasses; ++k)
      g1.cls[k] = c.coarse_class_count[k];
    return true;
  }

  // CC_SPLIT of the partition arr(p) (replicated): each partition's forest of
  // its rows, merged on the root, relabelled there and broadcast; returns
  // the number of pieces.
  template <typename Sel>
  i64 cc_split(Sel arr)
  {
    each([&](Part& p) {
      p.g0.with_w(
        [&](auto w) { cc_forest(p.g0.indptr, p.g0.indices, w, arr(p), n_, p.L.final_, p.s); });
    });
    const int P = ex_.parts();
    std::vector<const void*> forest;
    for (Part& p : parts_)
      forest.push_back(p.L.final_.parent);
    for (int q = 1; q < P; ++q) {  // one forest per round
      std::vector<i64> counts(static_cast<std::size_t>(P), 0);
      counts[static_cast<std::size_t>(q)] = n_;
      ex_.gatherv(
        0, forest.data(), sizeof(int), counts.data(), root_ ? root_->L.final_.keys_in : nullptr);
      if (root_) {
        use(*root_);
        cc_merge_kernel<<<grid_items(n_), kBlock, 0, root_->s>>>(
          reinterpret_cast<const int*>(root_->L.final_.keys_in), n_, root_->L.final_.parent);
        RAFT_CHECK_CUDA(root_->s);
      }
    }
    i64 pieces = 0;
    if (root_) {
      Part& r = *root_;
      use(r);
      const FinalBufs& fb = r.L.final_;
      cc_flatten_kernel<<<grid_items(n_), kBlock, 0, r.s>>>(n_, fb.parent, fb.flags);
      RAFT_CHECK_CUDA(r.s);
      cc_relabel(arr(r), n_, fb, r.L.cub, r.L.cub_bytes, r.L.ctl, r.s);
      pieces = read(r).n_components;
    }
    std::vector<void*> dst = ptrs([&](Part& p) { return arr(p); });
    ex_.bcast(0, dst.data(), n_ * static_cast<i64>(sizeof(int)));
    return from_root(pieces);
  }

  // Q of P0 (C communities): L_hat summed over the partitions' rows, the
  // volumes complete on each (all vertices), the same fp64 tree everywhere.
  double quality(i64 C)
  {
    each([&](Part& p) {
      p.g0.with_w([&](auto w) {
        quality_rows(p.g0.indptr,
                     p.g0.indices,
                     w,
                     p.g0.khat,
                     p.L.persist.P0,
                     n_,
                     1,
                     C,
                     p.L.final_,
                     p.L.ctl,
                     p.s);
      });
    });
    std::vector<std::vector<i64>> lh;
    read_values(1, [](Part& p) { return &p.L.ctl->l_hat[0]; }, lh);
    std::size_t i = 0;
    const i64 L   = sum_values([&](Part&) { return lh[i++][0]; });
    each([&](Part& p) {
      drv_set_i64_kernel<<<1, 1, 0, p.s>>>(&p.L.ctl->l_hat[0], L);
      RAFT_CHECK_CUDA(p.s);
      quality_tree(1, C, p.q.two_m_hat, gamma_, p.L.final_, p.L.ctl, p.s);
    });
    double q = 0.0;
    for (Part& p : parts_)
      q = read(p).q[0];  // the same everywhere
    return q;
  }
};

}  // namespace cugraph::detail::leiden_engine
