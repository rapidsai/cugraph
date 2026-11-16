/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// detail::leiden_csr (leiden_csr.hpp): input check (I1), canonicalization of non-canonical rows
// (I1c), workspace sizing and allocation, level-0 quantisation (I2, I4) and the engine call.
// Only I1, I1c and I2 depend on (vertex_t, edge_t, weight_t); the engine itself is compiled once
// (driver_common.cu). detail::leiden_coo_mg runs the same steps on the rows of one rank, combining
// the input check over the ranks, and the engine with level 0 split by rows (run_engine_mg).

#include "community/detail/leiden/arena.cuh"
#include "community/detail/leiden/canonicalize.cuh"
#include "community/detail/leiden/driver.hpp"
#include "community/detail/leiden/kernels_ingest.cuh"
#include "community/detail/leiden/leiden_csr.hpp"
#include "community/detail/leiden/numerics.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/host_staging_buffer_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/comms.hpp>
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
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph {
namespace detail {

// Layout, workspace, I2 (+ I4) and the engine on a CSR that passed I1 (`info`) as canonical and
// symmetric. <IP, IX, WI> are the types of the arrays the engine reads at level 0 (the input CSR,
// or its canonicalized copy). The levels above the first take arena blocks of exactly their size
// from the current RMM memory resource while the engine runs (run_engine).
template <typename IP, typename IX, typename WI, typename vertex_t>
leiden_result_t leiden_level0(raft::handle_t const& handle,
                              uint32_t seed,
                              IP const* indptr,
                              IX const* indices,
                              WI const* data,
                              bool weighted,  // false: unit weights (data is not read)
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

  le::LayoutParams p{};
  p.n           = n;
  p.nnz         = nnz;
  int const s   = le::scale_for_gamma(le::scale_s0(info.n_counted, info.wmax), params.resolution);
  p.wkind       = le::level0_kind(info.flags, weighted, std::is_same_v<WI, float>, s);
  p.idx64_input = !std::is_same_v<IX, int>;
  p.replicas    = le::n_replicas(info.n_counted, n);
  p.subrounds   = le::kNumSubrounds;

  le::EngineParams ep{};
  ep.n_iterations = params.n_iterations;
  ep.max_levels   = static_cast<int>(params.max_level);
  ep.graph_mode   = options.graph_mode;

  // int32 labels of the engine: the output itself for int32 vertex ids, else a separate buffer.
  std::optional<rmm::device_uvector<int>> labels32_buffer{std::nullopt};
  int* labels32{nullptr};
  if constexpr (std::is_same_v<vertex_t, int32_t>) {
    labels32 = labels.data();
  } else {
    labels32_buffer.emplace(n, handle.get_stream());
    labels32 = labels32_buffer->data();
  }

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

  le::EngineResult r{};
  le::run_engine(L, q, ep, params.resolution, seed, labels32, stream, r);
  if constexpr (!std::is_same_v<vertex_t, int32_t>) {
    thrust::copy(handle.get_thrust_policy(), labels32, labels32 + n, labels.begin());
  }
  return leiden_result_t{static_cast<size_t>(r.n_clusters),
                         r.modularity,
                         static_cast<size_t>(r.num_iterations),
                         static_cast<size_t>(r.num_levels)};
}

// Steps shared by the single-GPU and the multi-GPU paths: I1 (`ingest`), I1c (a canonical copy if
// a row is not canonical, checked again), the input errors and `level0` on the canonical rows.
// ingest(offsets, indices, weights, weighted, nnz) returns the IngestInfo of the whole graph (every
// rank receives the same one); level0(offsets, indices, weights, weighted, nnz, info) runs the
// engine. `weighted` (whether the graph has weights; its canonical copy always has: the fp64 sums,
// or multiplicities) is passed explicitly, the same on every rank: the weights of a rank without
// entries may be a null pointer.
template <typename vertex_t, typename edge_t, typename weight_t, typename Ingest, typename Level0>
leiden_result_t leiden_checked(raft::handle_t const& handle,
                               edge_t const* offsets,
                               vertex_t const* indices,
                               weight_t const* data,
                               bool weighted,
                               int64_t n,
                               int64_t nnz,
                               raft::device_span<vertex_t> labels,
                               Ingest&& ingest,
                               Level0&& level0)
{
  namespace le              = leiden_engine;
  cudaStream_t const stream = handle.get_stream().get();

  auto check_weights = [](le::IngestInfo const& info) {
    CUGRAPH_EXPECTS(!(info.flags & (le::kFlagNegative | le::kFlagNonFinite)),
                    "Invalid input argument: edge weights must be finite and non-negative.");
  };
  auto check_symmetric = [](le::IngestInfo const& info) {
    CUGRAPH_EXPECTS(!(info.flags & le::kFlagAsymmetric),
                    "Invalid input argument: the graph is not symmetric (an edge, or the sum of "
                    "its parallel edges, differs from its reverse).");
  };
  auto singletons = [&] {  // no counted edge: every vertex is its own community, Q = 0
    thrust::sequence(handle.get_thrust_policy(), labels.begin(), labels.end(), vertex_t{0});
    return leiden_result_t{static_cast<size_t>(n), 0.0, 0, 0};
  };

  // I1: structure, weights, row order, symmetry fingerprints, counted entries, w_max.
  le::IngestInfo info = ingest(offsets, indices, data, weighted, nnz);
  check_weights(info);

  if (info.flags & le::kFlagNonCanonical) {
    // I1c: sort the rows and sum parallel edges (fp64) into a canonical copy, then check again.
    int64_t const max_row = le::canon_max_row_nnz(offsets, n, stream);
    rmm::device_uvector<le::i64> c_offsets(n + 1, handle.get_stream());
    rmm::device_uvector<int> c_indices(nnz, handle.get_stream());
    rmm::device_uvector<double> c_weights(nnz, handle.get_stream());
    int64_t c_nnz{0};
    {
      rmm::device_buffer scratch(le::canonicalize_scratch_bytes(n, nnz, max_row),
                                 handle.get_stream());
      c_nnz = le::run_synced(stream, [&] {
        return le::run_canonicalize<edge_t, vertex_t, weight_t>(offsets,
                                                                indices,
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
    info = ingest(c_offsets.data(), c_indices.data(), c_weights.data(), true, c_nnz);
    check_weights(info);
    CUGRAPH_EXPECTS(!(info.flags & le::kFlagNonCanonical),
                    "leiden: internal error: the canonicalized graph is not canonical.");
    check_symmetric(info);
    if (info.n_counted == 0) return singletons();
    return level0(c_offsets.data(), c_indices.data(), c_weights.data(), true, c_nnz, info);
  }

  check_symmetric(info);
  if (info.n_counted == 0) return singletons();
  return level0(offsets, indices, data, weighted, nnz, info);
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
                  "Invalid input argument: the number of vertices must be < 2^30 - 1.");
  CUGRAPH_EXPECTS(static_cast<int64_t>(labels.size()) == n,
                  "Invalid input argument: labels must have one entry per vertex.");
  CUGRAPH_EXPECTS(!weights || static_cast<int64_t>(weights->size()) == nnz,
                  "Invalid input argument: the edge weights do not match the edges.");
  CUGRAPH_EXPECTS(options.graph_mode >= 0 && options.graph_mode <= 2,
                  "Invalid input argument: graph_mode must be 0, 1 or 2.");
  if (n == 0) return leiden_result_t{};

  cudaStream_t const stream = handle.get_stream().get();

  // Readback mailbox of the control block: pinned when cuGraph's staging pool is set up,
  // pageable stack memory otherwise (correct, a few microseconds slower per sync).
  std::optional<rmm::device_uvector<char>> staging{std::nullopt};
  void* pinned{nullptr};
  if (host_staging_buffer_manager::initialized()) {
    staging.emplace(host_staging_buffer_manager::allocate_staging_buffer<char>(
      le::kPinnedBytes, handle.get_stream()));
    pinned = staging->data();
  }

  auto ingest = [&](auto const* ip, auto const* ix, auto const* w, bool weighted, int64_t entries) {
    using IP = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using IX = std::remove_cv_t<std::remove_pointer_t<decltype(ix)>>;
    using WI = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    rmm::device_uvector<char> control(le::kControlBytes, handle.get_stream());
    auto* ctl = reinterpret_cast<le::Control*>(control.data());
    return le::run_synced(stream, [&] {
      return le::run_ingest_check<IP, IX, WI>(
        ip, ix, w, weighted, n, entries, true, ctl, pinned, stream);
    });
  };
  auto level0 = [&](auto const* ip,
                    auto const* ix,
                    auto const* w,
                    bool weighted,
                    int64_t entries,
                    le::IngestInfo const& info) {
    using IP = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using IX = std::remove_cv_t<std::remove_pointer_t<decltype(ix)>>;
    using WI = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    return leiden_level0<IP, IX, WI, vertex_t>(
      handle, seed, ip, ix, w, weighted, n, entries, info, pinned, labels, params, options);
  };
  return leiden_checked(handle,
                        offsets.data(),
                        indices.data(),
                        weights ? weights->data() : nullptr,
                        weights.has_value(),
                        n,
                        nnz,
                        labels,
                        ingest,
                        level0);
}

// ---------------------------------------------------------------------------
// Multi-GPU: level 0 split by rows over the ranks (leiden_coo_mg)
// ---------------------------------------------------------------------------

// I1 of this rank's rows, combined over the ranks into the I1 of the whole graph (flags or-ed,
// w_max the maximum, counted entries and symmetry fingerprints summed; the fingerprints of one rank
// do not match, as an edge and its reverse are stored on the ranks of their sources). Every rank
// receives the same result and throws alike: `weighted`, which also selects how w_max is decoded,
// is the same on every rank (`data` may be null on a rank without entries).
template <typename IP, typename IX, typename WI>
leiden_engine::IngestInfo ingest_check_mg(raft::comms::comms_t const& comm,
                                          IP const* indptr,
                                          IX const* indices,
                                          WI const* data,
                                          bool weighted,  // false: unit weights
                                          int64_t n,
                                          int64_t nnz,
                                          leiden_engine::Control* ctl,
                                          void* pinned,
                                          cudaStream_t stream)
{
  namespace le = leiden_engine;
  le::EntryValue<WI> const val{data, weighted};
  le::clear_control(ctl, stream);
  if (n > 0) {
    le::ingest_scan_kernel<IP, IX, WI>
      <<<le::grid_occ(le::ingest_scan_kernel<IP, IX, WI>, n, le::kWarpsPerBlock),
         le::kBlock,
         0,
         stream>>>(indptr, indices, val, n, nnz, ctl);
    RAFT_CHECK_CUDA(stream);
  }
  le::Control const h = le::read_control(ctl, stream, pinned);
  auto const all      = host_scalar_allgather(
    comm, thrust::make_tuple(h.flags, h.wmax_bits, h.n_counted, h.h_fwd, h.h_rev), stream);
  le::IngestInfo info;
  for (auto const& x : all) {
    info.flags |= thrust::get<0>(x);
    info.wmax_bits = std::max(info.wmax_bits, thrust::get<1>(x));
    info.n_counted += thrust::get<2>(x);
    info.h_fwd += thrust::get<3>(x);
    info.h_rev += thrust::get<4>(x);
  }
  le::check_structure_flags(info.flags);
  CUGRAPH_EXPECTS(info.n_counted < le::kMaxCountedEntries,
                  "leiden: graphs with >= 2^40 counted edges are not supported.");
  if (info.h_fwd != info.h_rev) info.flags |= le::kFlagAsymmetric;
  info.wmax = le::value_from_bits(info.wmax_bits, !val.weighted || std::is_same_v<WI, float>);
  return info;
}

// leiden_level0 for the rows [bounds[r], bounds[r + 1]) of rank r, held behind full-length
// offsets: the partitioned engine (run_engine_mg), or the single-GPU engine on the whole graph,
// gathered on every rank, where the partitioned one does not apply: one rank, more ranks than
// partitions (kMaxParts), or a graph small enough for replicas. Every rank receives the labels of
// all n vertices. `weighted` is the same on every rank; `data` may be null on a rank without
// entries (no kernel reads it there, and its collectives send 0 elements).
template <typename IP, typename IX, typename WI, typename vertex_t>
leiden_result_t leiden_level0_mg(raft::handle_t const& handle,
                                 uint32_t seed,
                                 std::vector<int64_t> const& bounds,
                                 IP const* indptr,
                                 IX const* indices,
                                 WI const* data,
                                 bool weighted,  // false: unit weights
                                 int64_t n,
                                 int64_t nnz,
                                 leiden_engine::IngestInfo const& info,
                                 void* pinned,
                                 raft::device_span<vertex_t> labels,
                                 leiden_params_t const& params,
                                 leiden_layout_options_t const& options)
{
  namespace le              = leiden_engine;
  auto& comm                = handle.get_comms();
  int const P               = comm.get_size();
  cudaStream_t const stream = handle.get_stream().get();
  auto const counts         = host_scalar_allgather(comm, static_cast<size_t>(nnz), stream);
  std::vector<size_t> displs(P, 0);
  std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), size_t{0});
  auto const nnz_all = static_cast<int64_t>(displs.back() + counts.back());

  if (P == 1 || P > le::kMaxParts || le::n_replicas(info.n_counted, n) > 1) {
    // The whole graph on every rank: the offsets are the sum of the ranks' (each covers its own
    // rows), the entries those of the ranks in rank order (= row order).
    rmm::device_uvector<int64_t> g_offsets(n + 1, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), indptr, indptr + n + 1, g_offsets.begin());
    comm.allreduce(g_offsets.data(), g_offsets.data(), n + 1, raft::comms::op_t::SUM, stream);
    rmm::device_uvector<IX> g_indices(nnz_all, handle.get_stream());
    comm.allgatherv(indices, g_indices.data(), counts.data(), displs.data(), stream);
    std::optional<rmm::device_uvector<WI>> g_weights{std::nullopt};
    if (weighted) {
      g_weights.emplace(nnz_all, handle.get_stream());
      comm.allgatherv(data, g_weights->data(), counts.data(), displs.data(), stream);
    }
    return leiden_level0<int64_t, IX, WI, vertex_t>(handle,
                                                    seed,
                                                    g_offsets.data(),
                                                    g_indices.data(),
                                                    g_weights ? g_weights->data() : nullptr,
                                                    weighted,
                                                    n,
                                                    nnz_all,
                                                    info,
                                                    pinned,
                                                    labels,
                                                    params,
                                                    options);
  }

  le::LayoutParams p{};
  p.n           = n;
  p.nnz         = nnz;
  int const s   = le::scale_for_gamma(le::scale_s0(info.n_counted, info.wmax), params.resolution);
  p.wkind       = le::level0_kind(info.flags, weighted, std::is_same_v<WI, float>, s);
  p.idx64_input = !std::is_same_v<IX, int>;
  p.replicas    = 1;
  p.subrounds   = le::kNumSubrounds;

  le::EngineParams ep{};
  ep.n_iterations = params.n_iterations;
  ep.max_levels   = static_cast<int>(params.max_level);
  ep.graph_mode   = options.graph_mode;

  std::vector<le::i64> const part_bounds(bounds.begin(), bounds.end());
  auto const quantize = [&](le::Layout& L, double gamma) {
    return le::run_quantize<IP, IX, WI>(
      indptr, indices, data, weighted, n, nnz, true, gamma, info, L, le::ClassThresholds{}, stream);
  };

  // int32 labels of the engine: the output itself for int32 vertex ids, else a separate buffer.
  std::optional<rmm::device_uvector<int>> labels32_buffer{std::nullopt};
  if constexpr (!std::is_same_v<vertex_t, int32_t>) {
    labels32_buffer.emplace(n, handle.get_stream());
  }
  int* labels32{nullptr};
  if constexpr (std::is_same_v<vertex_t, int32_t>) {
    labels32 = labels.data();
  } else {
    labels32 = labels32_buffer->data();
  }

  // A workspace allocation that fails fails on every rank (none may be left in a collective; the
  // arena blocks fail alike inside run_engine_mg).
  std::optional<rmm::device_buffer> workspace{std::nullopt};
  std::exception_ptr failed{nullptr};
  try {
    workspace.emplace(le::workspace_bytes(p), handle.get_stream());
  } catch (...) {
    failed = std::current_exception();
  }
  auto const failures =
    host_scalar_allreduce(comm, failed ? int{1} : int{0}, raft::comms::op_t::SUM, stream);
  if (failed) std::rethrow_exception(failed);
  CUGRAPH_EXPECTS(failures == 0, "leiden: the workspace allocation failed on another rank.");
  le::Layout L = le::make_layout(p, workspace->data());
  L.pinned     = pinned;

  le::EngineResult r{};
  le::run_engine_mg(comm,
                    std::move(L),
                    part_bounds,
                    info,
                    quantize,
                    ep,
                    params.resolution,
                    seed,
                    labels32,
                    stream,
                    r);
  if constexpr (!std::is_same_v<vertex_t, int32_t>) {
    thrust::copy(handle.get_thrust_policy(), labels32, labels32 + n, labels.begin());
  }
  return leiden_result_t{static_cast<size_t>(r.n_clusters),
                         r.modularity,
                         static_cast<size_t>(r.num_iterations),
                         static_cast<size_t>(r.num_levels)};
}

// leiden_coo_mg packs an edge (src, dst) into one 64-bit sort key; n < 2^30.
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
leiden_result_t leiden_coo_mg(raft::handle_t const& handle,
                              uint32_t seed,
                              vertex_t num_vertices,
                              std::vector<int64_t> const& bounds,
                              rmm::device_uvector<vertex_t>&& srcs,
                              rmm::device_uvector<vertex_t>&& dsts,
                              std::optional<rmm::device_uvector<weight_t>>&& weights,
                              raft::device_span<vertex_t> labels,
                              leiden_params_t const& params,
                              leiden_layout_options_t const& options)
{
  namespace le = leiden_engine;
  static_assert(std::is_same_v<vertex_t, int32_t> || std::is_same_v<vertex_t, int64_t>);
  static_assert(std::is_same_v<edge_t, int32_t> || std::is_same_v<edge_t, int64_t>);
  static_assert(std::is_same_v<weight_t, float> || std::is_same_v<weight_t, double>);

  // Checks of replicated arguments, which hold or fail on every rank alike.
  auto& comm     = handle.get_comms();
  int const P    = comm.get_size();
  int const rank = comm.get_rank();
  check_leiden_params(params);
  auto const n = static_cast<int64_t>(num_vertices);
  CUGRAPH_EXPECTS(n >= 0 && n < leiden_max_vertices,
                  "Invalid input argument: the number of vertices must be < 2^30 - 1.");
  CUGRAPH_EXPECTS(static_cast<int64_t>(labels.size()) == n,
                  "Invalid input argument: labels must have one entry per vertex.");
  CUGRAPH_EXPECTS(bounds.size() == static_cast<size_t>(P) + 1 && bounds[0] == 0 && bounds[P] == n,
                  "Invalid input argument: bounds must run from 0 to n in comm_size + 1 entries.");
  for (int q = 1; q <= P; ++q)
    CUGRAPH_EXPECTS(bounds[q - 1] <= bounds[q] && (q == P || bounds[q] % 32 == 0),
                    "Invalid input argument: bounds must be non-decreasing multiples of 32 (but "
                    "the last).");
  CUGRAPH_EXPECTS(options.graph_mode >= 0 && options.graph_mode <= 2,
                  "Invalid input argument: graph_mode must be 0, 1 or 2.");
  if (n == 0) return leiden_result_t{};

  auto const nnz = srcs.size();
  CUGRAPH_EXPECTS(dsts.size() == nnz && (!weights || weights->size() == nnz),
                  "Invalid input argument: sources, destinations and weights differ in size.");
  CUGRAPH_EXPECTS(nnz <= static_cast<size_t>(std::numeric_limits<edge_t>::max()),
                  "Invalid input argument: too many edges for edge_t.");
  cudaStream_t const stream = handle.get_stream().get();

  // This rank's rows as a CSR over all n rows (the other rows empty), each sorted by column, as
  // the single-GPU CSR. (Parallel edges end up adjacent in an order that depends on the input
  // order; I1c sums them in a canonical order of their weights.)
  rmm::device_uvector<uint64_t> keys(nnz, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    srcs.begin(),
                    srcs.end(),
                    dsts.begin(),
                    keys.begin(),
                    leiden_pack_edge_t<vertex_t>{num_vertices});
  srcs.resize(0, handle.get_stream());
  srcs.shrink_to_fit(handle.get_stream());
  dsts.resize(0, handle.get_stream());
  dsts.shrink_to_fit(handle.get_stream());
  if (weights) {
    thrust::sort_by_key(handle.get_thrust_policy(), keys.begin(), keys.end(), weights->begin());
  } else {
    thrust::sort(handle.get_thrust_policy(), keys.begin(), keys.end());
  }
  rmm::device_uvector<edge_t> offsets(n + 1, handle.get_stream());
  auto row_first = thrust::make_transform_iterator(thrust::make_counting_iterator(vertex_t{0}),
                                                   leiden_row_key_t<vertex_t>{});
  thrust::lower_bound(handle.get_thrust_policy(),
                      keys.begin(),
                      keys.end(),
                      row_first,
                      row_first + (n + 1),
                      offsets.begin());
  // Every edge on the rank of its source, both ends in [0, n) (reported on every rank alike).
  bool const placed =
    nnz == 0 ||
    (offsets.element(n, handle.get_stream()) == static_cast<edge_t>(nnz) &&
     offsets.element(bounds[rank], handle.get_stream()) == 0 &&
     offsets.element(bounds[rank + 1], handle.get_stream()) == static_cast<edge_t>(nnz));
  CUGRAPH_EXPECTS(
    host_scalar_allreduce(comm, placed ? int{0} : int{1}, raft::comms::op_t::SUM, stream) == 0,
    "Invalid input argument: an edge endpoint is outside [0, num_vertices), or an edge is not on "
    "the rank of its source.");
  // The weights decide which collectives run: passed on some ranks only, they would diverge.
  auto const weighted_ranks =
    host_scalar_allreduce(comm, weights ? int{1} : int{0}, raft::comms::op_t::SUM, stream);
  CUGRAPH_EXPECTS(weighted_ranks == 0 || weighted_ranks == P,
                  "Invalid input argument: weights must be passed on every rank or on none.");
  rmm::device_uvector<vertex_t> indices(nnz, handle.get_stream());
  thrust::transform(handle.get_thrust_policy(),
                    keys.begin(),
                    keys.end(),
                    indices.begin(),
                    leiden_key_dst_t<vertex_t>{});
  keys.resize(0, handle.get_stream());
  keys.shrink_to_fit(handle.get_stream());

  // Readback mailbox: pinned when cuGraph's staging pool is set up, else pageable (the partitioned
  // engine needs one).
  std::optional<rmm::device_uvector<char>> staging{std::nullopt};
  std::vector<char> pageable{};
  void* pinned{nullptr};
  if (host_staging_buffer_manager::initialized()) {
    staging.emplace(host_staging_buffer_manager::allocate_staging_buffer<char>(
      le::kPinnedBytes, handle.get_stream()));
    pinned = staging->data();
  } else {
    pageable.resize(le::kPinnedBytes);
    pinned = pageable.data();
  }

  rmm::device_uvector<char> control(le::kControlBytes, handle.get_stream());
  auto* ctl   = reinterpret_cast<le::Control*>(control.data());
  auto ingest = [&](auto const* ip, auto const* ix, auto const* w, bool weighted, int64_t entries) {
    using IP = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using IX = std::remove_cv_t<std::remove_pointer_t<decltype(ix)>>;
    using WI = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    return le::run_synced(stream, [&] {
      return ingest_check_mg<IP, IX, WI>(
        comm, ip, ix, w, weighted, n, entries, ctl, pinned, stream);
    });
  };
  auto level0 = [&](auto const* ip,
                    auto const* ix,
                    auto const* w,
                    bool weighted,
                    int64_t entries,
                    le::IngestInfo const& info) {
    using IP = std::remove_cv_t<std::remove_pointer_t<decltype(ip)>>;
    using IX = std::remove_cv_t<std::remove_pointer_t<decltype(ix)>>;
    using WI = std::remove_cv_t<std::remove_pointer_t<decltype(w)>>;
    return leiden_level0_mg<IP, IX, WI, vertex_t>(
      handle, seed, bounds, ip, ix, w, weighted, n, entries, info, pinned, labels, params, options);
  };
  // Weighted iff weights were passed (on every rank or on none), not by this rank's pointer: the
  // weights of a rank whose rows hold no edges are an empty vector, whose data() is null.
  return leiden_checked(handle,
                        offsets.data(),
                        indices.data(),
                        weights ? weights->data() : nullptr,
                        weights.has_value(),
                        n,
                        static_cast<int64_t>(nnz),
                        labels,
                        ingest,
                        level0);
}

}  // namespace detail
}  // namespace cugraph
