/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Interface between the typed raw-CSR entry point (community/detail/leiden/leiden_csr_impl.cuh:
// input check, canonicalization, layout, workspace, level-0 quantisation) and the
// type-independent engine (community/detail/leiden/driver_common.cu: local moving, refinement,
// aggregation, finalize), on one GPU (run_engine) or with level 0 split by rows over the ranks of
// a communicator (run_engine_mg, partition.cuh).
//
// After the level-0 quantisation (kernels_ingest.cuh, I2) every array the engine reads has an
// internal type (int64 offsets, int32 indices, fp32 / int64 / unit weights), independent of the
// (vertex_t, edge_t, weight_t) of the cuGraph graph. The engine kernels are therefore compiled
// once, in driver_common.cu, instead of once per instantiation of cugraph::leiden.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/numerics.cuh"

#include <raft/core/comms.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <vector>

namespace cugraph::detail::leiden_engine {

// Result of the input check I1 (kernels_ingest.cuh, run_ingest_check).
struct IngestInfo {
  u32 flags     = 0;
  u64 wmax_bits = 0;
  double wmax   = 0.0;  // max counted value (1.0 for unit weights)
  i64 n_counted = 0;    // nnz_c: stored entries with u != v and w > 0
  u64 h_fwd = 0, h_rev = 0;
};

// Level-0 weight kind for one scale exponent s.
inline WKind level0_kind(u32 flags, bool weighted, bool data_is_f32, int s)
{
  if (!weighted) return (flags & kFlagUnitZero) ? WKind::I64 : WKind::UNIT;
  if (data_is_f32 && s >= kF32ScaleMin && s <= kF32ScaleMax) return WKind::F32;
  return WKind::I64;
}

// Replicas of the first iteration: R = min(max_replicas, max(1, min(budget / nnz_c, (2^30 - 2) /
// n))). Small graphs run R disjoint copies with different hash contexts and keep the best one; the
// cap by n keeps the R * n vertex ids and CUB item counts in range (R * n <= 2^30 - 2).
constexpr i64 kReplicaNnzBudget = i64{1} << 19;

inline int n_replicas(i64 nnz_counted,
                      i64 n,
                      int max_replicas = kMaxReplicas,
                      i64 budget       = kReplicaNnzBudget)
{
  if (nnz_counted <= 0) return 1;
  i64 const r = std::min(budget / nnz_counted, (kMaxVertices - 2) / std::max<i64>(n, 1));
  return static_cast<int>(std::min<i64>(max_replicas, std::max<i64>(1, r)));
}

// Partitioned runs (partition.cuh; here because the typed entry point sizes their layouts):
// MoveBufs::xbuf holds ap_in and P OR words per bitmap word for P < 64; the
// control scratch holds the coarse ranges and offsets of A2.
constexpr int kMaxParts = 32;
static_assert(2 * (kMaxParts + 1) <= kScratchValues, "the A2 ranges must fit the control scratch");

// Level 0 after I2 (+ I4), as produced by run_quantize (kernels_ingest.cuh).
struct QuantizeResult {
  int s                        = 0;  // fixed-point scale exponent s(gamma)
  i64 unit_q                   = 0;  // quantised unit weight (WKind::UNIT)
  i64 two_m_hat                = 0;  // 2m_hat (exact int64)
  i64 class_count[kNumClasses] = {};
  WKind wkind                  = WKind::F32;
  // level-0 arrays (device pointers into the workspace or the input graph)
  i64 const* indptr  = nullptr;
  int const* indices = nullptr;
  float const* wf32  = nullptr;
  i64 const* wq      = nullptr;
};

// Knobs of one engine call. Everything else uses the engine defaults (4 DOWN sweeps of 4
// sub-rounds, 4 V-cycle sweeps in every iteration but the last (and in a single iteration), 32 TOP
// sweeps, 4 TOP re-entries per level, at most 20 iterations for n_iterations = -1).
struct EngineParams {
  int n_iterations = 2;           // >= 1 (< 2^16), or -1: until stable
  int max_levels   = kMaxLevels;  // levels per iteration, [1, 64]
  // 0: plain stream launches, 1: CUDA graph replays, 2: replays with the sweeps of a chunk in a
  // conditional WHILE node (falls back to 1 where unsupported). Launch layout only.
  int graph_mode = 2;
};

struct EngineResult {
  double modularity  = 0.0;  // exact Q(gamma) of the returned clustering
  i64 n_clusters     = 0;
  i64 num_iterations = 0;
  i64 num_levels     = 0;  // levels of the last iteration
};

/**
 * Runs the Leiden iterations and FINALIZE (exact Q, size-ordered labels) for one resolution on
 * the level-0 graph `q` carved in `L` (run_quantize must have run on `L`). Writes the int32 labels
 * of the n level-0 vertices to the device array `labels_out`. Every kernel and CUDA graph runs
 * on `stream`. The levels above the first live in arena blocks of exactly their size from the
 * current RMM memory resource, stream-ordered on `stream` (dropped levels free theirs, `L` the
 * rest); a block that cannot be allocated ends the call with its allocation error.
 */
void run_engine(Layout& L,
                QuantizeResult const& q,
                EngineParams const& params,
                double gamma,
                u32 seed,
                int* labels_out,
                cudaStream_t stream,
                EngineResult& out);

// I2 (run_quantize) of this rank's rows on its layout for one resolution.
using QuantizeFn = std::function<QuantizeResult(Layout&, double)>;

/**
 * run_engine with level 0 split by rows over the ranks of `comm` (partition.cuh; the collectives
 * are driver_mg.cuh's RaftExchange). Rank r owns the rows [bounds[r], bounds[r + 1]) (bounds:
 * comm.get_size() + 1 entries from 0 to n, non-decreasing, multiples of 32 but the last; at most
 * kMaxParts ranks) and holds them, behind full-length offsets with global column ids, in `L`: its
 * own L.p.nnz entries, one replica and a non-null L.pinned. `info` is the input check of the whole
 * graph (with at least one counted entry) and `quantize` runs I2 on L. Rank 0 runs the levels
 * >= 1. Every rank receives the labels of all n vertices in `labels_out` and the same `out`, the
 * result of run_engine on the whole graph (with one replica), bitwise.
 *
 * The arena blocks are allocated as in run_engine. A block that cannot be allocated ends the call
 * on every rank (none is left in a collective): that rank throws its allocation error, the others
 * a cugraph::logic_error.
 */
void run_engine_mg(raft::comms::comms_t const& comm,
                   Layout L,
                   std::vector<i64> const& bounds,
                   IngestInfo const& info,
                   QuantizeFn const& quantize,
                   EngineParams const& params,
                   double gamma,
                   u32 seed,
                   int* labels_out,
                   cudaStream_t stream,
                   EngineResult& out);

}  // namespace cugraph::detail::leiden_engine
