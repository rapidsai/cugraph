/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The type-independent engine of cugraph::leiden (see driver.hpp): local moving, refinement,
// aggregation and FINALIZE on the internal level-0 form, on one GPU (driver.cuh) or with level 0
// split by rows over the ranks (partition.cuh, driver_mg.cuh). Compiled once into
// libcugraph_common (not per vertex_t / edge_t / weight_t instantiation, and shared by the
// single-GPU and the multi-GPU library) and with --fmad=false (cpp/CMakeLists.txt).

#include "community/detail/leiden/driver.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/driver_mg.cuh"
#include "community/detail/leiden/partition.cuh"

#include <cugraph/utilities/error.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

#include <vector>

namespace cugraph::detail::leiden_engine {

namespace {

DriverParams driver_params(EngineParams const& params, Layout const& L)
{
  CUGRAPH_EXPECTS(
    params.n_iterations == -1 || (params.n_iterations >= 1 && params.n_iterations < (1 << 16)),
    "leiden: n_iterations must be in [1, 2^16) or -1.");
  CUGRAPH_EXPECTS(params.max_levels >= 1 && params.max_levels <= kMaxLevels,
                  "leiden: max_levels must be in [1, 64].");
  CUGRAPH_EXPECTS(params.graph_mode >= 0 && params.graph_mode <= 2,
                  "leiden: graph_mode must be 0, 1 or 2.");
  DriverParams dp;
  dp.n_iterations = params.n_iterations;
  dp.max_levels   = params.max_levels;
  dp.graph_mode   = params.graph_mode;
  dp.subrounds    = L.p.subrounds;
  return dp;
}

}  // namespace

bool run_engine(Layout& L,
                QuantizeResult const& q,
                EngineParams const& params,
                double gamma,
                u32 seed,
                int* labels_out,
                cudaStream_t stream,
                EngineResult& out)
{
  DriverParams const dp = driver_params(params, L);

  return run_synced(stream, [&] {
    LeidenDriver drv(L, dp, stream);
    drv.set_level0(q, L.p.n, L.p.nnz);
    ResolutionResult r;
    if (!drv.run_resolution(gamma, seed, L.p.replicas, labels_out, r)) return false;
    out.modularity     = r.modularity;
    out.n_clusters     = r.n_clusters;
    out.num_iterations = static_cast<i64>(r.iters.size());
    out.num_levels     = r.iters.empty() ? 0 : static_cast<i64>(r.iters.back().levels);
    return true;
  });
}

bool run_engine_mg(raft::comms::comms_t const& comm,
                   Layout& L,
                   std::vector<i64> const& bounds,
                   IngestInfo const& info,
                   QuantizeFn const& quantize,
                   EngineParams const& params,
                   double gamma,
                   u32 seed,
                   int* labels_out,
                   cudaStream_t stream,
                   EngineResult& out)
{
  DriverParams const dp = driver_params(params, L);
  int const P           = comm.get_size();
  int const rank        = comm.get_rank();
  CUGRAPH_EXPECTS(P <= kMaxParts && bounds.size() == static_cast<std::size_t>(P) + 1 &&
                    bounds[0] == 0 && bounds[P] == L.p.n && L.p.replicas == 1 &&
                    L.pinned != nullptr && info.n_counted > 0,
                  "leiden: internal error: invalid partitioned run.");

  std::vector<Part> parts(1);  // this rank's partition
  Part& part  = parts[0];
  part.global = rank;
  RAFT_CUDA_TRY(cudaGetDevice(&part.device));
  part.s      = stream;
  part.pinned = L.pinned;
  part.L      = L;
  part.v0     = bounds[rank];
  part.v1     = bounds[rank + 1];
  part.nnz    = L.p.nnz;
  RaftExchange ex(comm, stream);
  DriverInput in;
  in.info       = info;
  in.gammas     = {gamma};
  in.seeds      = {seed};
  in.labels_out = labels_out;
  DriverResult res;
  bool const ok = run_synced(stream, [&] {
    PartitionedDriver drv(parts, ex, dp, L.p.n, bounds);
    return drv.run(in, [&](Part& p, double g) { return quantize(p.L, g); }, res);
  });
  // the demand of an arena that overflowed, for the caller's rerun
  L.arena = part.L.arena;
  if (!ok) return false;

  // The labels (written on the root) and the levels of the last iteration (the root ran those
  // above level 0) to every rank.
  void* labels[] = {labels_out};
  ex.bcast(0, labels, L.p.n * static_cast<i64>(sizeof(int)));
  ResolutionResult const& r = res.res.back();
  i64 const levels          = r.iters.empty() ? 0 : static_cast<i64>(r.iters.back().levels);
  std::vector<i64> all_levels(static_cast<std::size_t>(P));
  ex.allgather_host(&levels, 1, all_levels.data());
  out.modularity     = r.modularity;
  out.n_clusters     = r.n_clusters;
  out.num_iterations = static_cast<i64>(r.iters.size());
  out.num_levels     = all_levels[0];
  return true;
}

}  // namespace cugraph::detail::leiden_engine
