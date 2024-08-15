/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

namespace cugraph::test {
template <typename vertex_t, bool multi_gpu>
size_t count_invalid_vertices(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> vertices,
  cugraph::vertex_partition_device_view_t<vertex_t, multi_gpu> const& vertex_partition);

template <typename vertex_t, bool multi_gpu>
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
          bool multi_gpu>
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

}  // namespace cugraph::test
