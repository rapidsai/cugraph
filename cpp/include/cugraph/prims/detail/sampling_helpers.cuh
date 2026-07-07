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

#include <thrust/copy.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <vector>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

// this functor output will later be used to convert global value (neighbor index, random number) to
// (local value, minor_comm_rank) pairs.
template <typename value_t>
struct compute_local_value_displacements_and_global_value_t {
  raft::device_span<value_t const> gathered_local_values{};
  raft::device_span<value_t>
    partitioned_local_value_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<value_t> global_values{};
  int minor_comm_size{};
  size_t num_values_per_key{};

  __device__ void operator()(size_t i) const
  {
    auto key_idx              = i / num_values_per_key;
    auto value_idx            = i % num_values_per_key;
    constexpr int buffer_size = 8;  // tuning parameter
    value_t displacements[buffer_size];
    value_t sum{0};
    for (int round = 0; round < (minor_comm_size + buffer_size - 1) / buffer_size; ++round) {
      auto loop_count = std::min(buffer_size, minor_comm_size - round * buffer_size);
      for (int j = 0; j < loop_count; ++j) {
        displacements[j] = sum;
        sum += gathered_local_values[i + (round * buffer_size + j) * global_values.size()];
      }
      thrust::copy(thrust::seq,
                   displacements,
                   displacements + loop_count,
                   partitioned_local_value_displacements.begin() +
                     key_idx * num_values_per_key * minor_comm_size + value_idx * minor_comm_size +
                     round * buffer_size);
    }
    global_values[i] = sum;
  }
};

template <typename value_t>
std::tuple<rmm::device_uvector<value_t>, rmm::device_uvector<value_t>>
compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
  raft::handle_t const& handle,
  raft::device_span<value_t const> aggregate_local_frontier_local_value_sums,
  raft::host_span<size_t const> local_frontier_offsets,
  size_t num_values_per_key);

template <typename edge_t, typename bias_t>
void sample_nbr_index_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  std::optional<raft::device_span<size_t const>> frontier_indices,
  raft::device_span<edge_t> nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  size_t K);

template <typename edge_t, typename edge_type_t, typename bias_t>
void sample_nbr_index_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  std::optional<std::tuple<raft::device_span<size_t const>, raft::device_span<edge_type_t const>>>
    frontier_index_type_pairs,
  raft::device_span<edge_t> per_type_nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum);

template <typename edge_t, typename edge_type_t>
rmm::device_uvector<edge_t> compute_local_nbr_indices_from_per_type_local_nbr_indices(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  std::optional<std::tuple<raft::device_span<edge_type_t const>, raft::device_span<size_t const>>>
    edge_type_key_idx_pairs,
  rmm::device_uvector<edge_t>&& per_type_local_nbr_indices,
  raft::host_span<size_t const> local_frontier_sample_offsets,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum);

template <typename edge_t>
rmm::device_uvector<edge_t> remap_local_nbr_indices(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<edge_t const> aggregate_local_frontier_unique_key_org_indices,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  rmm::device_uvector<edge_t>&& local_nbr_indices,
  std::optional<raft::device_span<size_t const>> key_indices,
  raft::host_span<size_t const> local_frontier_sample_offsets,
  size_t K);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
