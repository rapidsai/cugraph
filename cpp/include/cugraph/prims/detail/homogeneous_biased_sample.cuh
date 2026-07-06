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
 * @brief Algorithm A-Res biased sampling index selection for homogeneous graphs.
 *
 * Template dependence is limited to @p edge_t and @p bias_t.
 */
template <typename edge_t, typename bias_t>
void compute_homogeneous_biased_sampling_index_without_replacement(
  raft::handle_t const& handle,
  std::optional<raft::device_span<size_t const>>
    input_frontier_indices,  // input_degree_offsets & input_biases
                             // are already packed if std::nullopt
  raft::device_span<size_t const> input_degree_offsets,
  raft::device_span<bias_t const> input_biases,  // bias 0 edges can't be selected
  std::optional<raft::device_span<size_t const>>
    output_frontier_indices,  // output_nbr_indices is already packed if std::nullopt
  raft::device_span<edge_t> output_nbr_indices,
  std::optional<raft::device_span<bias_t>> output_keys,
  raft::random::RngState& rng_state,
  size_t K,
  bool jump);

/**
 * @brief Biased sampling without replacement for homogeneous graphs.
 *
 * Template dependence is limited to @p edge_t, @p bias_t, and @p multi_gpu.
 */
template <typename edge_t, typename bias_t, bool multi_gpu>
std::tuple<rmm::device_uvector<edge_t> /* local_nbr_indices */,
           std::optional<rmm::device_uvector<size_t>> /* key_indices */,
           std::vector<size_t> /* local_frontier_sample_offsets */>
homogeneous_biased_sample_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  raft::random::RngState& rng_state,
  size_t K);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
