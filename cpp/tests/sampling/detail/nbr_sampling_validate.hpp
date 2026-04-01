/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/sampling_functions.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>

// utilities for testing / verification of Nbr Sampling functionality:
//
namespace cugraph {
namespace test {

template <typename vertex_t, typename weight_t>
bool validate_extracted_graph_is_subgraph(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> src,
  raft::device_span<vertex_t const> dst,
  std::optional<raft::device_span<weight_t const>> wgt,
  raft::device_span<vertex_t const> subgraph_src,
  raft::device_span<vertex_t const> subgraph_dst,
  std::optional<raft::device_span<weight_t const>> subgraph_wgt);

template <typename vertex_t>
bool validate_sampling_depth(raft::handle_t const& handle,
                             rmm::device_uvector<vertex_t>&& d_src,
                             rmm::device_uvector<vertex_t>&& d_dst,
                             rmm::device_uvector<vertex_t>&& d_source_vertices,
                             int max_depth);

template <typename vertex_t, typename time_stamp_t>
bool validate_temporal_integrity(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> srcs,
  raft::device_span<vertex_t const> dsts,
  raft::device_span<time_stamp_t const> edge_times,
  raft::device_span<vertex_t const> source_vertices,
  cugraph::temporal_sampling_comparison_t temporal_sampling_comparison);

/**
 * @brief Validate disjoint sampling constraints.
 *
 * For disjoint sampling, batches (labels) of sources should expand without overlapping destinations
 * across batches for the same hop.
 */
template <typename vertex_t>
bool validate_disjoint_sampling(raft::handle_t const& handle,
                                raft::device_span<vertex_t const> srcs,
                                raft::device_span<vertex_t const> dsts,
                                raft::device_span<vertex_t const> starting_vertices,
                                std::optional<raft::device_span<size_t const>> label_offsets,
                                std::optional<raft::device_span<int32_t const>> batch_numbers);

}  // namespace test
}  // namespace cugraph
