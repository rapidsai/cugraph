/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// Core of cugraph::leiden (community/leiden_impl.cuh) on raw arrays. The single-GPU path passes
// the graph view's own CSR to leiden_csr() (zero copy); the multi-GPU path passes the edges of each
// rank's rows to leiden_coo_mg(). Compiled once into libcugraph_common
// (leiden_csr_common_v*_e*.cu), so libcugraph (SG) and libcugraph_mg share one copy of the engine.

#include <cugraph/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace cugraph {
namespace detail {

// Input bounds under which every int64 fixed-point sum and penalty of the engine is exact (no
// overflow): vertex ids and the R * n replica ids fit in 30 bits (and the 2n + 2 items of the CUB
// scans in 31), and the quantised weights are scaled so that 2m <= 2^58 with headroom for
// resolutions up to 2^20.
constexpr int64_t leiden_max_vertices   = (int64_t{1} << 30) - 1;
constexpr double leiden_max_resolution  = 1048576.0;  // 2^20
constexpr size_t leiden_max_level_cap   = 64;
constexpr int32_t leiden_max_iterations = int32_t{1} << 16;  // exclusive

/**
 * @brief Layout options of the engine. They never change the result (the kernels, their
 * arguments and every decision are the same); tests use them to cover every launch and memory
 * path.
 */
struct leiden_layout_options_t {
  /// How the engine launches its kernels: 0 plain stream launches, 1 CUDA graph replays, 2 CUDA
  /// graph replays with the local-moving sweeps in a conditional WHILE node (CUDA >= 12.4; falls
  /// back to 1 where unsupported).
  int graph_mode{2};
  /// Initial size of the level arena, in units of 8 bytes per stored entry. A level that does not
  /// fit makes the engine rerun on a larger arena.
  double arena_factor{1.9};
};

/**
 * @brief Throws cugraph::logic_error if a field of @p params is outside its documented range.
 */
inline void check_leiden_params(leiden_params_t const& params)
{
  CUGRAPH_EXPECTS(params.resolution >= 0.0 && params.resolution <= leiden_max_resolution,
                  "Invalid input argument: resolution must be finite and in [0, 2^20].");
  CUGRAPH_EXPECTS(params.n_iterations == -1 ||
                    (params.n_iterations >= 1 && params.n_iterations < leiden_max_iterations),
                  "Invalid input argument: n_iterations must be in [1, 2^16), or -1.");
  CUGRAPH_EXPECTS(std::isfinite(params.beta) && params.beta >= 0.0,
                  "Invalid input argument: beta must be finite and >= 0.");
  CUGRAPH_EXPECTS(params.beta == 0.0,
                  "Invalid input argument: beta > 0 (randomized refinement) is not implemented "
                  "yet; use beta = 0.");
  CUGRAPH_EXPECTS(params.max_level >= 1 && params.max_level <= leiden_max_level_cap,
                  "Invalid input argument: max_level must be in [1, 64].");
}

/**
 * @brief Leiden on a raw CSR (the core of cugraph::leiden).
 *
 * Runs the input check, canonicalizes rows that are not strictly increasing by column (parallel
 * edges are summed), and runs the engine on the CSR (@p offsets, @p indices, @p weights) of
 * n = offsets.size() - 1 vertices. Same semantics and guarantees as cugraph::leiden; the result
 * depends only on the multiset of stored (row, column, weight) entries, @p seed and @p params.
 *
 * @param handle   RAFT handle; every kernel runs on handle.get_stream().
 * @param seed     32-bit seed of the call.
 * @param offsets  CSR offsets, n + 1 entries.
 * @param indices  CSR column indices.
 * @param weights  CSR weights aligned with @p indices (std::nullopt: every entry has weight 1).
 * @param labels   Output, n entries: the community of every vertex, in [0, num_clusters),
 *                 ordered by decreasing community size (ties by the smallest vertex id).
 * @param params   Algorithm parameters (checked with check_leiden_params()).
 * @param options  Layout options (see leiden_layout_options_t; the defaults are the production
 *                 configuration).
 * @return Result summary.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_csr(raft::handle_t const& handle,
                           uint32_t seed,
                           raft::device_span<edge_t const> offsets,
                           raft::device_span<vertex_t const> indices,
                           std::optional<raft::device_span<weight_t const>> weights,
                           raft::device_span<vertex_t> labels,
                           leiden_params_t const& params,
                           leiden_layout_options_t const& options = leiden_layout_options_t{});

/**
 * @brief Multi-GPU Leiden on the edges of this rank's rows (the multi-GPU path of
 * cugraph::leiden), with level 0 split by rows over the ranks of handle.get_comms().
 *
 * Rank r holds every edge whose source lies in [bounds[r], bounds[r + 1]) (in any order). Every
 * rank assembles the CSR of its rows (sorted by column, as the single-GPU CSR), the input check is
 * combined over the ranks, and the engine runs partitioned: every rank stores only its rows, rank 0
 * also runs the levels above the first (detail/leiden/partition.cuh). One rank, more than 32
 * ranks, or a graph the single-GPU engine runs as replicas (at most 2^18 counted edges and fewer
 * than 2^29 vertices) run the single-GPU engine on the whole graph on every rank instead. The
 * weights must be passed on every rank or on none (checked). Either way, the result and
 * the labels are those of leiden_csr() on the CSR of all edges with the same seed, bitwise, on
 * every rank. Consumes @p srcs, @p dsts and @p weights.
 *
 * @param handle        RAFT handle with an initialized communicator; every kernel and collective
 *                      runs on handle.get_stream().
 * @param seed          32-bit seed of the call (the same on every rank).
 * @param num_vertices  Number of vertices n (< 2^30 - 1); vertex ids are in [0, n).
 * @param bounds        Row ranges of the ranks: comm_size + 1 entries from 0 to n, non-decreasing,
 *                      multiples of 32 but the last (the same on every rank).
 * @param srcs          Sources of this rank's edges (in [bounds[rank], bounds[rank + 1])).
 * @param dsts          Destinations of this rank's edges.
 * @param weights       Weights of this rank's edges (std::nullopt on every rank: unit weights).
 * @param labels        Output, n entries on every rank (see leiden_csr()).
 * @param params        Algorithm parameters.
 * @param options       Layout options (see leiden_csr()); rank 0 sizes its arena by arena_factor.
 * @return Result summary (the same on every rank).
 */
template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_coo_mg(raft::handle_t const& handle,
                              uint32_t seed,
                              vertex_t num_vertices,
                              std::vector<int64_t> const& bounds,
                              rmm::device_uvector<vertex_t>&& srcs,
                              rmm::device_uvector<vertex_t>&& dsts,
                              std::optional<rmm::device_uvector<weight_t>>&& weights,
                              raft::device_span<vertex_t> labels,
                              leiden_params_t const& params,
                              leiden_layout_options_t const& options = leiden_layout_options_t{});

}  // namespace detail
}  // namespace cugraph
