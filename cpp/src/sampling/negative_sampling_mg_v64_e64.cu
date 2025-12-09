/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "negative_sampling_impl.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/sampling_functions.hpp>

namespace cugraph {

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<raft::device_span<float const>> src_bias,
  std::optional<raft::device_span<float const>> dst_bias,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check);

template std::tuple<rmm::device_uvector<int64_t>, rmm::device_uvector<int64_t>> negative_sampling(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<int64_t, int64_t, false, true> const& graph_view,
  std::optional<raft::device_span<double const>> src_bias,
  std::optional<raft::device_span<double const>> dst_bias,
  size_t num_samples,
  bool remove_duplicates,
  bool remove_existing_edges,
  bool exact_number_of_samples,
  bool do_expensive_check);

}  // namespace cugraph
