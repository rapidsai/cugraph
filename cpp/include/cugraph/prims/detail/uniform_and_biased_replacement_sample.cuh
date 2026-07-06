/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_span.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/**
 * @brief Uniform sampling index selection without replacement for homogeneous graphs.
 *
 * Template dependence is limited to @p edge_t.
 */
template <typename edge_t>
rmm::device_uvector<edge_t> compute_homogeneous_uniform_sampling_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  raft::random::RngState& rng_state,
  size_t K);

/**
 * @brief Biased sampling with replacement.
 *
 * Template dependence is limited to @p edge_t, @p edge_type_t, @p bias_t, and @p multi_gpu.
 */
template <typename edge_t, typename edge_type_t, typename bias_t, bool multi_gpu>
std::tuple<rmm::device_uvector<edge_t> /* local_nbr_indices */,
           std::optional<rmm::device_uvector<size_t>> /* key_indices */,
           std::vector<size_t> /* local_frontier_sample_offsets */>
biased_sample_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
