/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// detail::leiden_csr (leiden_csr.hpp): input check (I1), canonicalization of non-canonical rows
// (I1c), workspace sizing and allocation, level-0 quantisation (I2, I4) and the engine call.
// Only I1, I1c and I2 depend on (vertex_t, edge_t, weight_t); the engine itself is compiled once
// (driver_common.cu). detail::leiden_coo assembles the CSR of an edge list in any order.

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/canonicalize.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/kernels_ingest.cuh"
#include "community/detail/leiden/leiden_csr.hpp"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/host_staging_buffer_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

namespace cugraph {
namespace detail {

// Workspace sizing: the level arena holds arena_factor * 8 B per input entry
// (leiden_layout_options_t::arena_factor, 1.9 by default, which fits every level of the k-NN
// graphs we measured). A level that does not fit ends the engine call, which is then repeated on a
// larger workspace (at most leiden_max_reruns times; the layout never reaches a decision, so every
// rerun gives the bitwise identical result).
constexpr double leiden_workspace_growth = 2.0;
constexpr double leiden_required_slack   = 1.25;
constexpr int leiden_max_reruns          = 6;

// Layout, workspace (with the regrow retries), I2 (+ I4) and the engine on a CSR that passed I1
// (`info`) as canonical and symmetric. <IP, IX, WI> are the types of the arrays the engine reads
// at level 0 (the input CSR, or its canonicalized copy).
template <typename IP, typename IX, typename WI, typename vertex_t>
leiden_result_t leiden_level0(raft::handle_t const& handle,
                              uint32_t seed,
                              IP const* indptr,
                              IX const* indices,
                              WI const* data,  // nullptr: unit weights
                              int64_t n,
                              int64_t nnz,
                              leiden_engine::IngestInfo const& info,
                              void* pinned,
                              raft::device_span<vertex_t> labels,
                              leiden_params_t const& params,
                              leiden_layout_options_t const& options)
{
  namespace le              = leiden_engine;
  cudaStream_t const stream = handle.get_stream().get();
  bool const weighted       = data != nullptr;

  le::LayoutParams p{};
  p.n            = n;
  p.nnz          = nnz;
  int const s    = le::scale_for_gamma(le::scale_s0(info.n_counted, info.wmax), params.resolution);
  p.wkind        = le::level0_kind(info.flags, weighted, std::is_same_v<WI, float>, s);
  p.idx64_input  = !std::is_same_v<IX, int>;
  p.replicas     = le::n_replicas(info.n_counted);
  p.arena_factor = options.arena_factor;
  p.subrounds    = le::kNumSubrounds;

  le::EngineParams ep{};
  ep.n_iterations = params.n_iterations;
  ep.max_levels   = static_cast<int>(params.max_level);
  ep.graph_mode   = options.graph_mode;

  // int32 labels of the engine: the output itself for int32 vertex ids, else a separate buffer.
  std::optional<rmm::device_uvector<int>> labels32_buffer{std::nullopt};
  if constexpr (!std::is_same_v<vertex_t, int32_t>) {
    labels32_buffer.emplace(n, handle.get_stream());
  }

  for (int run = 0;; ++run) {
    rmm::device_buffer workspace(le::workspace_bytes(p), handle.get_stream());
    le::Layout L = le::make_layout(p, workspace.data());
    L.pinned     = pinned;

    // I2 (+ I4): level 0 in the engine's internal form, quantised for this resolution.
    auto const q = le::run_synced(stream, [&] {
      return le::run_quantize<IP, IX, WI>(indptr,
                                          indices,
                                          data,
                                          weighted,
                                          n,
                                          nnz,
                                          true,
                                          params.resolution,
                                          info,
                                          L,
                                          le::ClassThresholds{},
                                          stream);
    });

    int* labels32{nullptr};
    if constexpr (std::is_same_v<vertex_t, int32_t>) {
      labels32 = labels.data();
    } else {
      labels32 = labels32_buffer->data();
    }
    le::EngineResult r{};
    if (le::run_engine(L, q, ep, params.resolution, seed, labels32, stream, r)) {
      if constexpr (!std::is_same_v<vertex_t, int32_t>) {
        thrust::copy(handle.get_thrust_policy(), labels32, labels32 + n, labels.begin());
      }
      return leiden_result_t{static_cast<size_t>(r.n_clusters),
                             r.modularity,
                             static_cast<size_t>(r.num_iterations),
                             static_cast<size_t>(r.num_levels)};
    }
    // A level did not fit into the level arena (rare): grow the arena (the workspace is released
    // at the end of this iteration, before the next one allocates).
    CUGRAPH_EXPECTS(run < leiden_max_reruns,
                    "leiden: the level arena overflowed %d times in a row.",
                    leiden_max_reruns + 1);
    double const required = le::arena_required_factor(p, L.arena.required);
    p.arena_factor =
      std::max(leiden_workspace_growth * p.arena_factor, leiden_required_slack * required);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_csr(raft::handle_t const& handle,
                           uint32_t seed,
                           raft::device_span<edge_t const> offsets,
                           raft::device_span<vertex_t const> indices,
                           std::optional<raft::device_span<weight_t const>> weights,
                           raft::device_span<vertex_t> labels,
                           leiden_params_t const& params,
                           leiden_layout_options_t const& options)
{
  namespace le = leiden_engine;
  static_assert(std::is_same_v<vertex_t, int32_t> || std::is_same_v<vertex_t, int64_t>);
  static_assert(std::is_same_v<edge_t, int32_t> || std::is_same_v<edge_t, int64_t>);
  static_assert(std::is_same_v<weight_t, float> || std::is_same_v<weight_t, double>);

  check_leiden_params(params);
  CUGRAPH_EXPECTS(offsets.size() >= 1,
                  "Invalid input argument: the CSR offsets must have n + 1 entries.");
  auto const n   = static_cast<int64_t>(offsets.size()) - 1;
  auto const nnz = static_cast<int64_t>(indices.size());
  CUGRAPH_EXPECTS(n < leiden_max_vertices,
                  "Invalid input argument: the number of vertices must be < 2^30.");
  CUGRAPH_EXPECTS(static_cast<int64_t>(labels.size()) == n,
                  "Invalid input argument: labels must have one entry per vertex.");
  CUGRAPH_EXPECTS(!weights || static_cast<int64_t>(weights->size()) == nnz,
                  "Invalid input argument: the edge weights do not match the edges.");
  CUGRAPH_EXPECTS(options.graph_mode >= 0 && options.graph_mode <= 2,
                  "Invalid input argument: graph_mode must be 0, 1 or 2.");
  CUGRAPH_EXPECTS(options.arena_factor > 0.0 && std::isfinite(options.arena_factor),
                  "Invalid input argument: arena_factor must be finite and > 0.");
  if (n == 0) return leiden_result_t{};

  cudaStream_t const stream = handle.get_stream().get();
  weight_t const* data      = weights ? weights->data() : nullptr;

  // Readback mailbox of the control block: pinned when cuGraph's staging pool is set up,
  // pageable stack memory otherwise (correct, a few microseconds slower per sync).
  std::optional<rmm::device_uvector<char>> staging{std::nullopt};
  void* pinned{nullptr};
  if (host_staging_buffer_manager::initialized()) {
    staging.emplace(host_staging_buffer_manager::allocate_staging_buffer<char>(
      le::kPinnedBytes, handle.get_stream()));
    pinned = staging->data();
  }

  auto ingest = [&](auto const* ip, auto const* ix, auto const* w, int64_t entries) {
    using IP = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using IX = std::remove_cv_t<std::remove_pointer_t<decltype(ix)>>;
    using WI = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    rmm::device_uvector<char> control(le::kControlBytes, handle.get_stream());
    auto* ctl = reinterpret_cast<le::Control*>(control.data());
    auto info = le::run_synced(stream, [&] {
      return le::run_ingest_check<IP, IX, WI>(
        ip, ix, w, w != nullptr, n, entries, true, ctl, pinned, stream);
    });
    CUGRAPH_EXPECTS(!(info.flags & (le::kFlagNegative | le::kFlagNonFinite)),
                    "Invalid input argument: edge weights must be finite and non-negative.");
    return info;
  };
  auto check_symmetric = [](le::IngestInfo const& info) {
    CUGRAPH_EXPECTS(!(info.flags & le::kFlagAsymmetric),
                    "Invalid input argument: the graph is not symmetric (an edge, or the sum of "
                    "its parallel edges, differs from its reverse).");
  };

  // I1: structure, weights, row order, symmetry fingerprints, counted entries, w_max.
  le::IngestInfo info = ingest(offsets.data(), indices.data(), data, nnz);

  if (info.flags & le::kFlagNonCanonical) {
    // I1c: sort the rows and sum parallel edges (fp64) into a canonical copy, then check again.
    int64_t const max_row = le::canon_max_row_nnz(offsets.data(), n, stream);
    rmm::device_uvector<le::i64> c_offsets(n + 1, handle.get_stream());
    rmm::device_uvector<int> c_indices(nnz, handle.get_stream());
    rmm::device_uvector<double> c_weights(nnz, handle.get_stream());
    int64_t c_nnz{0};
    {
      rmm::device_buffer scratch(le::canonicalize_scratch_bytes(n, nnz, max_row),
                                 handle.get_stream());
      c_nnz = le::run_synced(stream, [&] {
        return le::run_canonicalize<edge_t, vertex_t, weight_t>(offsets.data(),
                                                                indices.data(),
                                                                data,
                                                                n,
                                                                nnz,
                                                                max_row,
                                                                c_offsets.data(),
                                                                c_indices.data(),
                                                                c_weights.data(),
                                                                scratch.data(),
                                                                scratch.size(),
                                                                stream);
      });
    }
    c_indices.resize(c_nnz, handle.get_stream());
    c_weights.resize(c_nnz, handle.get_stream());
    info = ingest(c_offsets.data(), c_indices.data(), c_weights.data(), c_nnz);
    CUGRAPH_EXPECTS(!(info.flags & le::kFlagNonCanonical),
                    "leiden: internal error: the canonicalized graph is not canonical.");
    check_symmetric(info);
    if (info.n_counted == 0) {  // no counted edge: every vertex is its own community, Q = 0
      thrust::sequence(handle.get_thrust_policy(), labels.begin(), labels.end(), vertex_t{0});
      return leiden_result_t{static_cast<size_t>(n), 0.0, 0, 0};
    }
    return leiden_level0<le::i64, int, double, vertex_t>(handle,
                                                         seed,
                                                         c_offsets.data(),
                                                         c_indices.data(),
                                                         c_weights.data(),
                                                         n,
                                                         c_nnz,
                                                         info,
                                                         pinned,
                                                         labels,
                                                         params,
                                                         options);
  }

  check_symmetric(info);
  if (info.n_counted == 0) {  // no counted edge: every vertex is its own community, Q = 0
    thrust::sequence(handle.get_thrust_policy(), labels.begin(), labels.end(), vertex_t{0});
    return leiden_result_t{static_cast<size_t>(n), 0.0, 0, 0};
  }
  return leiden_level0<edge_t, vertex_t, weight_t, vertex_t>(handle,
                                                             seed,
                                                             offsets.data(),
                                                             indices.data(),
                                                             data,
                                                             n,
                                                             nnz,
                                                             info,
                                                             pinned,
                                                             labels,
                                                             params,
                                                             options);
}

// leiden_coo packs an edge (src, dst) into one 64-bit sort key; n < 2^30.
constexpr int leiden_key_bits = 30;

template <typename vertex_t>
struct leiden_pack_edge_t {
  vertex_t n;
  __device__ uint64_t operator()(vertex_t src, vertex_t dst) const
  {
    // an id outside [0, n) gets a key past every row (reported after the sort)
    if (src < 0 || src >= n || dst < 0 || dst >= n) return ~uint64_t{0};
    return (static_cast<uint64_t>(src) << leiden_key_bits) | static_cast<uint64_t>(dst);
  }
};

template <typename vertex_t>
struct leiden_key_dst_t {
  __device__ vertex_t operator()(uint64_t key) const
  {
    return static_cast<vertex_t>(key & ((uint64_t{1} << leiden_key_bits) - 1));
  }
};

template <typename vertex_t>
struct leiden_row_key_t {
  __device__ uint64_t operator()(vertex_t v) const
  {
    return static_cast<uint64_t>(v) << leiden_key_bits;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_coo(raft::handle_t const& handle,
                           uint32_t seed,
                           vertex_t num_vertices,
                           rmm::device_uvector<vertex_t>&& srcs,
                           rmm::device_uvector<vertex_t>&& dsts,
                           std::optional<rmm::device_uvector<weight_t>>&& weights,
                           raft::device_span<vertex_t> labels,
                           leiden_params_t const& params,
                           leiden_layout_options_t const& options)
{
  check_leiden_params(params);
  auto const n   = static_cast<int64_t>(num_vertices);
  auto const nnz = srcs.size();
  CUGRAPH_EXPECTS(n >= 0 && n < leiden_max_vertices,
                  "Invalid input argument: the number of vertices must be < 2^30.");
  CUGRAPH_EXPECTS(dsts.size() == nnz && (!weights || weights->size() == nnz),
                  "Invalid input argument: sources, destinations and weights differ in size.");
  CUGRAPH_EXPECTS(nnz <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                  "Invalid input argument: too many edges for edge_t.");
  auto const stream = handle.get_stream();

  rmm::device_uvector<uint64_t> keys(nnz, stream);
  thrust::transform(handle.get_thrust_policy(),
                    srcs.begin(),
                    srcs.end(),
                    dsts.begin(),
                    keys.begin(),
                    leiden_pack_edge_t<vertex_t>{num_vertices});
  srcs.resize(0, stream);
  srcs.shrink_to_fit(stream);
  dsts.resize(0, stream);
  dsts.shrink_to_fit(stream);

  // Rows sorted by column: the CSR depends only on the multiset of edges, not on their order.
  // (Parallel edges end up adjacent in an order that depends on the input order; leiden_csr sums
  // them in a canonical order of their weights.)
  if (weights) {
    thrust::sort_by_key(handle.get_thrust_policy(), keys.begin(), keys.end(), weights->begin());
  } else {
    thrust::sort(handle.get_thrust_policy(), keys.begin(), keys.end());
  }
  rmm::device_uvector<edge_t> offsets(n + 1, stream);
  auto row_first = thrust::make_transform_iterator(thrust::make_counting_iterator(vertex_t{0}),
                                                   leiden_row_key_t<vertex_t>{});
  thrust::lower_bound(handle.get_thrust_policy(),
                      keys.begin(),
                      keys.end(),
                      row_first,
                      row_first + (n + 1),
                      offsets.begin());
  CUGRAPH_EXPECTS(nnz == 0 || offsets.element(n, stream) == static_cast<edge_t>(nnz),
                  "Invalid input argument: an edge endpoint is outside [0, num_vertices).");
  rmm::device_uvector<vertex_t> indices(nnz, stream);
  thrust::transform(handle.get_thrust_policy(),
                    keys.begin(),
                    keys.end(),
                    indices.begin(),
                    leiden_key_dst_t<vertex_t>{});
  keys.resize(0, stream);
  keys.shrink_to_fit(stream);

  return leiden_csr<vertex_t, edge_t, weight_t>(
    handle,
    seed,
    raft::device_span<edge_t const>(offsets.data(), offsets.size()),
    raft::device_span<vertex_t const>(indices.data(), indices.size()),
    weights
      ? std::make_optional(raft::device_span<weight_t const>(weights->data(), weights->size()))
      : std::nullopt,
    labels,
    params,
    options);
}

}  // namespace detail
}  // namespace cugraph
