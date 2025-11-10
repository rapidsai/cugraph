/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/vertex_partition_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

namespace cugraph::test {
template <typename vertex_t, bool multi_gpu>
size_t count_invalid_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  cugraph::vertex_partition_view_t<vertex_t, multi_gpu> const& vertex_partition);

template <typename vertex_t>
size_t count_duplicate_vertex_pairs_sorted(raft::handle_t const& handle,
                                           raft::device_span<vertex_t const> src,
                                           raft::device_span<vertex_t const> dst);

template <typename vertex_t>
void sort(raft::handle_t const& handle,
          raft::device_span<vertex_t> srcs,
          raft::device_span<vertex_t> dsts);

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename time_stamp_t>
void sort(raft::handle_t const& handle,
          raft::device_span<vertex_t> srcs,
          raft::device_span<vertex_t> dsts,
          std::optional<raft::device_span<weight_t>> wgts,
          std::optional<raft::device_span<edge_t>> ids,
          std::optional<raft::device_span<edge_type_t>> types,
          std::optional<raft::device_span<time_stamp_t>> start_times,
          std::optional<raft::device_span<time_stamp_t>> end_times);

template <typename vertex_t, typename edge_t, typename weight_t, typename edge_type_t>
size_t count_intersection(raft::handle_t const& handle,
                          raft::device_span<vertex_t const> srcs1,
                          raft::device_span<vertex_t const> dsts1,
                          std::optional<raft::device_span<weight_t const>> wgts1,
                          std::optional<raft::device_span<edge_t const>> edge_ids1,
                          std::optional<raft::device_span<edge_type_t const>> edge_types1,
                          raft::device_span<vertex_t const> srcs2,
                          raft::device_span<vertex_t const> dsts2,
                          std::optional<raft::device_span<weight_t const>> wgts2,
                          std::optional<raft::device_span<edge_t const>> edge_ids2,
                          std::optional<raft::device_span<edge_type_t const>> edge_types2);

template <typename vertex_t>
size_t count_edges_on_wrong_int_gpu(raft::handle_t const& handle,
                                    raft::device_span<vertex_t const> srcs,
                                    raft::device_span<vertex_t const> dsts,
                                    raft::device_span<vertex_t const> vertex_partition_range_lasts);

}  // namespace cugraph::test
