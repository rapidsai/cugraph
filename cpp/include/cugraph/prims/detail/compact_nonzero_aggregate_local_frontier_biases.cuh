/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/export.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <tuple>

namespace CUGRAPH_EXPORT cugraph {
namespace detail {

/**
 * @brief Exclude zero-bias neighbors and compact aggregate-local-frontier bias data.
 *
 * Post-processes the output of @p transform_v_frontier_e bias collection. Template dependence
 * is limited to @p edge_t and @p bias_t.
 */
template <typename edge_t, typename bias_t>
std::tuple<rmm::device_uvector<bias_t>, rmm::device_uvector<edge_t>, rmm::device_uvector<size_t>>
compact_nonzero_aggregate_local_frontier_biases(
  raft::handle_t const& handle,
  rmm::device_uvector<bias_t>&& aggregate_local_frontier_biases,
  rmm::device_uvector<size_t>&& aggregate_local_frontier_local_degree_offsets,
  size_t local_frontier_size,
  bool do_expensive_check,
  bool multi_gpu);

}  // namespace detail
}  // namespace CUGRAPH_EXPORT cugraph
