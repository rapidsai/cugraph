/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

// cugraph::leiden (algorithms.hpp). Both paths end in the raw-array core of
// community/detail/leiden/ (leiden_csr.hpp), which holds the engine:
//
//   single-GPU  the graph view's own CSR (offsets, indices, weights) is passed to
//               detail::leiden_csr as is (zero copy).
//   multi-GPU   level 0 is split by rows over the ranks: rank r owns the rows of its vertex
//               partition range with both ends rounded down to a multiple of 32 (the last end is
//               n), the local edges (internal vertex ids) are shuffled to the owners of their
//               sources, and detail::leiden_coo_mg runs the engine partitioned (every rank stores
//               only its rows; rank 0 also runs the levels above the first, see
//               detail/leiden/partition.cuh) with the seed of rank 0. Every rank receives the
//               labels of all vertices and keeps those of its local vertex partition range. The
//               result equals single-GPU Leiden on the same (internal) vertex ids, bitwise, for any
//               number of GPUs.

#include "community/detail/leiden/leiden_csr.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace cugraph {

namespace detail {

// 32-bit seed of one call: the low 32 bits of rng_state.seed, XOR a hash of the base subsequence
// (zero for a fresh RngState, so RngState{s} gives the seed uint32_t(s)). Advances the state by one
// subsequence, so successive calls with the same state use different seeds.
inline uint32_t leiden_seed(raft::random::RngState& rng_state)
{
  uint32_t h = 0;
  if (rng_state.base_subsequence != 0) {
    uint64_t x = rng_state.base_subsequence + 0x9E3779B97F4A7C15ull;  // splitmix64 finalizer
    x          = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x          = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    h = static_cast<uint32_t>(x >> 32);
  }
  rng_state.advance(1);
  return static_cast<uint32_t>(rng_state.seed) ^ h;
}

template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_sg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t> clustering,
  leiden_params_t const& params)
{
  CUGRAPH_EXPECTS(
    static_cast<size_t>(clustering.size()) == static_cast<size_t>(graph_view.number_of_vertices()),
    "Invalid input argument: clustering must have graph_view.number_of_vertices() entries.");
  auto const seed = leiden_seed(rng_state);

  // A single-GPU graph has one edge partition: a CSR whose rows are sorted by column, with the
  // weights aligned with the indices. The core reads it in place.
  auto edge_partition = graph_view.local_edge_partition_view();
  auto offsets        = edge_partition.offsets();
  auto indices        = edge_partition.indices();
  std::optional<raft::device_span<weight_t const>> weights{std::nullopt};
  if (edge_weight_view) {
    CUGRAPH_EXPECTS(edge_weight_view->value_firsts().size() == 1 &&
                      static_cast<size_t>(edge_weight_view->edge_counts()[0]) == indices.size(),
                    "Invalid input argument: the edge weights do not match the graph.");
    weights =
      raft::device_span<weight_t const>(edge_weight_view->value_firsts()[0], indices.size());
  }
  return leiden_csr<vertex_t, edge_t, weight_t>(
    handle,
    seed,
    raft::device_span<edge_t const>(offsets.data(), offsets.size()),
    raft::device_span<vertex_t const>(indices.data(), indices.size()),
    weights,
    clustering,
    params);
}

template <typename vertex_t, typename edge_t, typename weight_t>
leiden_result_t leiden_mg(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, true> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t> clustering,
  leiden_params_t const& params,
  bool do_expensive_check)
{
  auto& comm        = handle.get_comms();
  auto const stream = handle.get_stream();

  bool const local_size_ok = static_cast<size_t>(clustering.size()) ==
                             static_cast<size_t>(graph_view.local_vertex_partition_range_size());
  if (do_expensive_check) {
    auto const bad_ranks = host_scalar_allreduce(
      comm, local_size_ok ? int32_t{0} : int32_t{1}, raft::comms::op_t::SUM, stream);
    CUGRAPH_EXPECTS(bad_ranks == 0,
                    "Invalid input argument: clustering must have "
                    "graph_view.local_vertex_partition_range_size() entries (on every rank).");
  }
  CUGRAPH_EXPECTS(local_size_ok,
                  "Invalid input argument: clustering must have "
                  "graph_view.local_vertex_partition_range_size() entries.");

  // Every rank uses the seed of rank 0 (and advances its own state, as in single-GPU).
  auto const seed = host_scalar_bcast(comm, leiden_seed(rng_state), int{0}, stream);

  // Engine rows of every rank: its vertex partition range with both ends rounded down to a
  // multiple of 32 (each word of an activity bitmap has one owner), the last end n.
  auto const comm_size = comm.get_size();
  auto const n         = graph_view.number_of_vertices();
  auto const lasts     = graph_view.vertex_partition_range_lasts();
  std::vector<vertex_t> row_lasts(comm_size);
  std::vector<int64_t> bounds(comm_size + 1, 0);
  for (int r = 0; r < comm_size; ++r) {
    row_lasts[r]  = r + 1 < comm_size ? lasts[r] / 32 * 32 : n;
    bounds[r + 1] = row_lasts[r];
  }

  // The local edges with internal (global) vertex ids, shuffled to the ranks owning their sources.
  rmm::device_uvector<vertex_t> srcs(0, stream);
  rmm::device_uvector<vertex_t> dsts(0, stream);
  std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
  {
    auto [local_srcs, local_dsts, local_weights, ids, types] =
      decompress_to_edgelist<vertex_t, edge_t, weight_t, int32_t, false, true>(
        handle,
        graph_view,
        edge_weight_view,
        std::optional<edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
        std::optional<edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
        std::optional<raft::device_span<vertex_t const>>{std::nullopt});
    std::vector<arithmetic_device_uvector_t> properties{};
    properties.push_back(std::move(local_dsts));
    if (local_weights) { properties.push_back(std::move(*local_weights)); }
    std::tie(srcs, properties) =
      shuffle_int_vertices(handle,
                           std::move(local_srcs),
                           std::move(properties),
                           raft::host_span<vertex_t const>(row_lasts.data(), row_lasts.size()));
    dsts = std::move(std::get<rmm::device_uvector<vertex_t>>(properties[0]));
    if (properties.size() > 1) {
      weights = std::move(std::get<rmm::device_uvector<weight_t>>(properties[1]));
    }
  }

  // The partitioned engine (labels of all vertices on every rank), then the local slice.
  rmm::device_uvector<vertex_t> labels(n, stream);
  auto const result = leiden_coo_mg<vertex_t, edge_t, weight_t>(
    handle,
    seed,
    n,
    bounds,
    std::move(srcs),
    std::move(dsts),
    std::move(weights),
    raft::device_span<vertex_t>(labels.data(), labels.size()),
    params);
  thrust::copy(handle.get_thrust_policy(),
               labels.begin() + graph_view.local_vertex_partition_range_first(),
               labels.begin() + graph_view.local_vertex_partition_range_last(),
               clustering.begin());
  return result;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
leiden_result_t leiden(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  raft::device_span<vertex_t> clustering,
  leiden_params_t const& params,
  bool do_expensive_check)
{
  static_assert(std::is_same_v<vertex_t, int32_t> || std::is_same_v<vertex_t, int64_t>);
  static_assert(std::is_same_v<weight_t, float> || std::is_same_v<weight_t, double>);

  // Checks that hold or fail identically on every rank (global graph properties, parameters).
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");
  CUGRAPH_EXPECTS(
    static_cast<int64_t>(graph_view.number_of_vertices()) < detail::leiden_max_vertices,
    "Invalid input argument: the number of vertices must be < 2^30 - 1.");
  detail::check_leiden_params(params);

  if constexpr (multi_gpu) {
    return detail::leiden_mg<vertex_t, edge_t, weight_t>(
      handle, rng_state, graph_view, edge_weight_view, clustering, params, do_expensive_check);
  } else {
    // The input checks of the core (CSR structure, finite non-negative weights, symmetry) cost one
    // pass over the edges and always run; there is nothing more expensive to check.
    return detail::leiden_sg<vertex_t, edge_t, weight_t>(
      handle, rng_state, graph_view, edge_weight_view, clustering, params);
  }
}

}  // namespace cugraph
