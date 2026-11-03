/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The type-independent engine of cugraph::leiden (see driver.hpp): local moving, refinement,
// aggregation and FINALIZE on the internal level-0 form. Compiled once into libcugraph_common
// (not per vertex_t / edge_t / weight_t instantiation, and shared by the single-GPU and the
// multi-GPU library) and with --fmad=false (cpp/CMakeLists.txt).

#include "community/detail/leiden/driver.cuh"
#include "community/detail/leiden/driver.hpp"

#include <cugraph/utilities/error.hpp>

namespace cugraph::detail::leiden_engine {

bool run_engine(Layout& L,
                QuantizeResult const& q,
                EngineParams const& params,
                double gamma,
                u32 seed,
                int* labels_out,
                cudaStream_t stream,
                EngineResult& out)
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

}  // namespace cugraph::detail::leiden_engine
