/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/**
 * @brief Reservoir sampling (algorithm R) without replacement for homogeneous frontiers.
 *
 * Template dependence is limited to @p edge_t and @p bias_t (random number type).
 */
template <typename edge_t, typename bias_t>
void sample_nbr_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  std::optional<raft::device_span<size_t const>> frontier_indices,
  raft::device_span<edge_t> nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  size_t K,
  bool algo_r = true);

/**
 * @brief Reservoir sampling (algorithm R) without replacement for heterogeneous frontiers.
 *
 * Template dependence is limited to @p edge_t, @p edge_type_t, and @p bias_t.
 */
template <typename edge_t, typename edge_type_t, typename bias_t>
void sample_nbr_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  std::optional<std::tuple<raft::device_span<size_t const>, raft::device_span<edge_type_t const>>>
    frontier_index_type_pairs,
  raft::device_span<edge_t> per_type_nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum,
  bool algo_r = true);

/**
 * @brief Uniform sampling index selection without replacement for heterogeneous graphs.
 *
 * Template dependence is limited to @p edge_t and @p edge_type_t.
 */
template <typename edge_t, typename edge_type_t>
rmm::device_uvector<edge_t> compute_heterogeneous_uniform_sampling_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
