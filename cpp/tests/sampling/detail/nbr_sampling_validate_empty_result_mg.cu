/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Multi-GPU instantiations live here (linked only by cugraphmgtestutil) so that SG test
// executables are not forced to resolve multi_gpu graph_view symbols from libcugraph_mg.

#include "nbr_sampling_validate_empty_result.cuh"

namespace cugraph {
namespace test {

template bool validate_sampling_empty_result(
  raft::handle_t const&,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const&,
  raft::device_span<int32_t const>,
  size_t,
  bool);

template bool validate_sampling_empty_result(
  raft::handle_t const&,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const&,
  raft::device_span<int64_t const>,
  size_t,
  bool);

template bool validate_sampling_empty_result(
  raft::handle_t const&,
  cugraph::graph_view_t<int32_t, int32_t, false, true> const&,
  cugraph::edge_property_view_t<int32_t, int32_t const*>,
  raft::device_span<int32_t const>,
  std::optional<raft::device_span<int32_t const>>,
  std::optional<raft::device_span<int32_t const>>,
  size_t,
  cugraph::temporal_sampling_comparison_t);

template bool validate_sampling_empty_result(
  raft::handle_t const&,
  cugraph::graph_view_t<int64_t, int64_t, false, true> const&,
  cugraph::edge_property_view_t<int64_t, int32_t const*>,
  raft::device_span<int64_t const>,
  std::optional<raft::device_span<int32_t const>>,
  std::optional<raft::device_span<int32_t const>>,
  size_t,
  cugraph::temporal_sampling_comparison_t);

}  // namespace test
}  // namespace cugraph
