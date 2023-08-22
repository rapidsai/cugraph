/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {

/**
 * @brief renumber sampling output
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) outputs satisfying the
 * following requirements.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, flag=src, hop) triplets for each
 * vertex ID in @p edgelist_srcs and (vertex ID, flag=dst, hop) triplets for each vertex ID in @p
 * edgelist_dsts. From these triplets, we can find the minimum (hop, flag) pairs for every unique
 * vertex ID (hop is the primary key and flag is the secondary key, flag=src is considered smaller
 * than flag=dst if hop numbers are same). Vertex IDs with smaller (hop, flag) pairs precede vertex
 * IDs with larger (hop, flag) pairs in renumbering. Ordering can be arbitrary among the vertices
 * with the same (hop, flag) pairs.
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in @p edgelist_srcs precede vertex IDs that
 * appear only in @p edgelist_dsts.
 * 3. If label_offsets.has_value() is ture, edge lists for different labels will be renumbered
 * separately.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam label_t Type of labels. Needs to be an integral type.
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing original edgelist source vertices.
 * @param edgelist_dsts A vector storing original edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_hops An optional pointer to the array storing hops for each edge list (source,
 * destination) pairs (size = @p edgelist_srcs.size() if valid).
 * @param label_offsets An optional tuple of unique labels and the input edge list (@p
 * edgelist_srcs, @p edgelist_hops, and @p edgelist_dsts) offsets for the labels (size = # unique
 * labels + 1).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing renumbered edge sources (size = @p edgelist_srcs.size()) ,
 * renumbered edge destinations (size = @p edgelist_dsts.size()), renumber_map to query original
 * verties (size = # unique vertices or aggregate # unique vertices for every label), and
 * renumber_map offsets (size = std::get<0>(*label_offsets).size() + 1, valid only if @p
 * label_offsets.has_value() is true).
 */
template <typename vertex_t, typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<size_t>>>
renumber_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<size_t const>>>
    label_offsets,
  bool do_expensive_check = false);

/*
 * @brief compress edges in the CSR|CSC format.
 *
 * This function assumes that source/destination IDs are renumbered (using the
 * cugraph::renumber_sampled_edgelist function). If @p compress_src is true, compress in the CSR
 * format, If @p compress_src is false, compress in the CSC format. If edgelist_hops.has_value() or
 * label_offsets.has_value() is true, edges lists for different hops/labels will be compressed
 * separately.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @tparam label_t Type of labels. Needs to be an integral type.
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param label_offsets An optional tuple of unique labels and the input edge list (@p
 * edgelist_srcs, @p edgelist_hops, and @p edgelist_dsts) offsets for the labels (size = # unique
 * labels + 1).
 * @param compress_src A flag determine whether to compress source (results in the CSR format) or
 * destination (results in the CSC format).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing an offset array (or a collection of offset arrays for each label
 * if @p label_offsets.has_value() is true), edge destinations (if @p compress_src is true) or
 * sources (if @p compress_dst is true), optional edge weights (valid if @p
 * edgelist_weights.has_value() is true), optional edge IDs (valid if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid if @p
 * edgelist_edge_types.has_value() is true), and label offsets for the collection of offset arrays
 * (valid if @p label_offsets.has_value() is true, size = thrust::get<0>(*label_offsets).size() +
 * 1). Size of the offset array (or each array in the collection of offset arrays) is the maximum
 * vertex ID in the edge list (for each label if @p label_offsets.has_value() is true) + 1.
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          typename label_t>
std::tuple<rmm::device_uvector<size_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_id_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>,
           std::optional<rmm::device_uvector<size_t>>>
compress_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<size_t const>>>
    label_offsets,
  bool compress_src,
  bool do_expensive_check = false);

/*
 * @brief sort edges by (src, dst) pairs.
 *
 * If @p src_is_primary_key is true, edges are sorted by (primary key: src, secondary key: dst). If
 * @p src_is_primary_key is false, edges are sorted by (primary key: dst, secondary key: src).
 * Orders among the edges with the same (src, dst) are arbitrary. If edgelist_hops.has_value() or
 * label_offsets.has_value() is true, edge lists for different hops|labels will be sorted
 * separately.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
 * @tparam label_t Type of labels. Needs to be an integral type.
 * @param  handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs A vector storing edgelist source vertices.
 * @param edgelist_dsts A vector storing edgelist destination vertices (size = @p
 * edgelist_srcs.size()).
 * @param edgelist_weights An optional vector storing edgelist weights (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_ids An optional vector storing edgelist edge IDs (size = @p
 * edgelist_srcs.size() if valid).
 * @param edgelist_edge_types An optional vector storing edgelist edge types (size = @p
 * edgelist_srcs.size() if valid).
 * @param label_offsets An optional tuple of unique labels and the input edge list (@p
 * edgelist_srcs, @p edgelist_hops, and @p edgelist_dsts) offsets for the labels (size = # unique
 * labels + 1).
 * @param src_is_primary_key A flag to determine whether to use the source or destination as the
 * primary key in sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid if
 * @p edgelist_weights.has_value() is true), optional edge IDs (valid if @p
 * edgelist_edge_ids.has_value() is true), and optional edge types (valid if @p
 * edgelist_edge_types.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t,
          typename label_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_id_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<std::tuple<raft::device_span<label_t const>, raft::device_span<size_t const>>>
    label_offsets,
  bool src_is_primary_key,
  bool do_expensive_check = false);

}  // namespace cugraph
