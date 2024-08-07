/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <optional>

template <typename index_t>
bool check_offsets(raft::handle_t const& handle,
                   raft::device_span<index_t const> offsets,
                   index_t num_segments,
                   index_t num_elements);

template <typename vertex_t>
bool check_edgelist_is_sorted(raft::handle_t const& handle,
                              raft::device_span<vertex_t const> edgelist_majors,
                              raft::device_span<vertex_t const> edgelist_minors);

// unrenumber the renumbered edge list and check whether the original & unrenumbered edge lists are
// identical
template <typename vertex_t, typename weight_t>
bool compare_edgelist(raft::handle_t const& handle,
                      raft::device_span<vertex_t const> org_edgelist_srcs,
                      raft::device_span<vertex_t const> org_edgelist_dsts,
                      std::optional<raft::device_span<weight_t const>> org_edgelist_weights,
                      raft::device_span<vertex_t const> renumbered_edgelist_srcs,
                      raft::device_span<vertex_t const> renumbered_edgelist_dsts,
                      std::optional<raft::device_span<weight_t const>> renumbered_edgelist_weights,
                      std::optional<raft::device_span<vertex_t const>> renumber_map);

// unrenumber the renumbered edge list and check whether the original & unrenumbered edge lists are
// identical
template <typename vertex_t, typename weight_t, typename edge_id_t, typename edge_type_t>
bool compare_heterogeneous_edgelist(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> org_edgelist_srcs,
  raft::device_span<vertex_t const> org_edgelist_dsts,
  std::optional<raft::device_span<weight_t const>> org_edgelist_weights,
  std::optional<raft::device_span<edge_id_t const>> org_edgelist_edge_ids,
  std::optional<raft::device_span<edge_type_t const>> org_edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  std::optional<raft::device_span<size_t const>> org_edgelist_label_offsets,
  raft::device_span<vertex_t const> renumbered_edgelist_srcs,
  raft::device_span<vertex_t const> renumbered_edgelist_dsts,
  std::optional<raft::device_span<weight_t const>> renumbered_edgelist_weights,
  std::optional<raft::device_span<edge_id_t const>> renumbered_edgelist_edge_ids,
  std::optional<raft::device_span<size_t const>> renumbered_edgelist_label_edge_type_hop_offsets,
  raft::device_span<vertex_t const> vertex_renumber_map,
  raft::device_span<size_t const> vertex_renumber_map_label_type_offsets,
  std::optional<raft::device_span<edge_id_t const>> edge_id_renumber_map,
  std::optional<raft::device_span<size_t const>> edge_id_renumber_map_label_type_offsets,
  raft::device_span<vertex_t const> vertex_type_offsets,
  size_t num_labels,
  size_t num_vertex_types,
  size_t num_edge_types,
  size_t num_hops);

template <typename vertex_t>
bool check_vertex_renumber_map_invariants(
  raft::handle_t const& handle,
  std::optional<raft::device_span<vertex_t const>> starting_vertices,
  raft::device_span<vertex_t const> org_edgelist_srcs,
  raft::device_span<vertex_t const> org_edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> org_edgelist_hops,
  raft::device_span<vertex_t const> renumber_map,
  bool src_is_major);
