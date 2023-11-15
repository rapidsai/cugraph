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

/*
 * @brief renumber sampled edge list and compress to the (D)CSR|(D)CSC format.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edges fulfilling
 * the following requirements. Assume major = source if @p src_is_major is true, major = destination
 * if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pairs for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs.
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors.
 * 3. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are compressed based on the following requirements.
 *
 * 1. If @p compress_per_hop is true, edges are compressed separately for each hop. If @p
 * compress_per_hop is false, edges with different hop numbers are compressed altogether.
 * 2. Edges are compressed independently for different labels.
 * 3. If @p doubly_compress is false, edges are compressed to CSR (if @p src_is_major is true) or
 * CSC (if @p src_is_major is false). If @p doubly_compress is true, edges are compressed to DCSR
 * (if @p src_is_major is true) or DCSC (if @p src_is_major is false). If @p doubly_compress is
 * false, the CSR/CSC offset array size is the number of vertices (which is the maximum vertex ID +
 * 1) + 1. Here, the maximum vertex ID is the maximum major vertex ID in the edges to compress if @p
 * compress_per_hop is false or for hop 0. If @p compress_per_hop is true and hop number is 1 or
 * larger, the maximum vertex ID is the larger of the maximum major vertex ID for this hop and the
 * maximum vertex ID for the edges in the previous hops.
 *
 * If both @p compress_per_hop is false and @p edgelist_hops.has_value() is true, majors should be
 * non-decreasing within each label after renumbering and sorting by (hop, major, minor). Also,
 * majors in hop N should not appear in any of the previous hops. This condition is satisfied if
 * majors in hop N + 1 does not have any vertices from the previous hops excluding the minors from
 * hop N.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
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
 * @param edgelist_hops An optional tuple having a vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid) and the number of hops.
 * @param edgelist_label_offsets An optional tuple storing a pointer to the array storing label
 * offsets to the input edges (size = std::get<1>(*edgelist_label_offsets) + 1) and the number of
 * labels.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and compression.
 * @param compress_per_hop A flag to determine whether to compress edges with different hop numbers
 * separately (if true) or altogether (if false). If @p compress_per_hop is true, @p
 * edgelist_hops.has_value() should be true and @p doubly_compress should be false.
 * @param doubly_compress A flag to determine whether to compress to the CSR/CSC format (if false)
 * or the DCSR/DCSC format (if true).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing optional DCSR/DCSC major vertex IDs with one or more neighbors,
 * (D)CSR|(D)CSC offset values, edge minor vertex IDs, optional edge weights (valid only if @p
 * edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the
 * (D)CSR|(D)CSC offset array (size = # labels * # hops + 1, where # labels =
 * std::get<1>(*edgelist_label_offsets) if @p edgelist_label_offsets.has_value() is true and 1
 * otherwise and # hops = std::get<1>(*edgelist_hops) if edgelist_hops.has_value() is true and 1
 * otherwise, valid only if at least one of @p edgelist_label_offsets.has_value() or @p
 * edgelist_hops.has_value() is true), renumber_map to query original vertices (size = # unique
 * vertices or aggregate # unique vertices for every label), and label offsets to the renumber_map
 * (size = std::get<1>(*edgelist_label_offsets) + 1, valid only if @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<std::optional<rmm::device_uvector<vertex_t>>,     // dcsr/dcsc major vertices
           rmm::device_uvector<size_t>,                      // (d)csr/(d)csc offset values
           rmm::device_uvector<vertex_t>,                    // minor vertices
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,  // (label, hop) offsets to the (d)csr/(d)csc
                                                        // offset array
           rmm::device_uvector<vertex_t>,               // renumber map
           std::optional<rmm::device_uvector<size_t>>>  // label offsets to the renumber map
renumber_and_compress_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major       = true,
  bool compress_per_hop   = false,
  bool doubly_compress    = false,
  bool do_expensive_check = false);

/*
 * @brief renumber sampled edge list and sort the renumbered edges.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edges fulfilling
 * the following requirements. Assume major = source if @p src_is_major is true, major = destination
 * if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pairs for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs.
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors.
 * 3. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((hop), src, dst) as the key in sorting. If @p src_is_major is
 * false, use ((hop), dst, src) instead. hop is used only if @p edgelist_hops.has_value() is true.
 * 2. Edges in each label are sorted independently if @p edgelist_label_offsets.has_value() is true.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
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
 * @param edgelist_hops An optional tuple having a vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid) and the number of hops. The hop vector values should be
 * non-decreasing within each label.
 * @param edgelist_label_offsets An optional tuple storing a pointer to the array storing label
 * offsets to the input edges (size = std::get<1>(*edgelist_label_offsets) + 1) and the number of
 * labels.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the renumbered
 * and sorted edges (size = # labels * # hops + 1, where # labels =
 * std::get<1>(*edgelist_label_offsets) if @p edgelist_label_offsets.has_value() is true and 1
 * otherwise and # hops = std::get<1>(*edgelist_hops) if edgelist_hops.has_value() is true and 1
 * otherwise, valid only if at least one of @p edgelist_label_offsets.has_value() or @p
 * edgelist_hops.has_value() is true), renumber_map to query original vertices (size = # unique
 * vertices or aggregate # unique vertices for every label), and label offsets to the renumber_map
 * (size = std::get<1>(*edgelist_label_offsets) + 1, valid only if @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>,       // (label, hop) offsets to the edges
           rmm::device_uvector<vertex_t>,                    // renumber map
           std::optional<rmm::device_uvector<size_t>>>       // label offsets to the renumber map
renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major       = true,
  bool do_expensive_check = false);

/*
 * @brief sort sampled edge list.
 *
 * Sampled edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((hop), src, dst) as the key in sorting. If @p src_is_major is
 * false, use ((hop), dst, src) instead. hop is used only if @p edgelist_hops.has_value() is true.
 * 2. Edges in each label are sorted independently if @p edgelist_label_offsets.has_value() is true.
 *
 * This function is single-GPU only (we are not aware of any practical multi-GPU use cases).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weight.  Needs to be floating point type
 * @tparam edge_id_t Type of edge id.  Needs to be an integral type
 * @tparam edge_type_t Type of edge type.  Needs to be an integral type, currently only int32_t is
 * supported
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
 * @param edgelist_hops An optional tuple having a vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid) and the number of hops. The hop vector values should be
 * non-decreasing within each label.
 * @param edgelist_label_offsets An optional tuple storing a pointer to the array storing label
 * offsets to the input edges (size = std::get<1>(*edgelist_label_offsets) + 1) and the number of
 * labels.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), and optional (label, hop) offset values to the
 * renumbered and sorted edges (size = # labels * # hops + 1, where # labels =
 * std::get<1>(*edgelist_label_offsets) if @p edgelist_label_offsets.has_value() is true and 1
 * otherwise and # hops = std::get<1>(*edgelist_hops) if edgelist_hops.has_value() is true and 1
 * otherwise, valid only if at least one of @p edgelist_label_offsets.has_value() or @p
 * edgelist_hops.has_value() is true)
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<rmm::device_uvector<vertex_t>,                    // srcs
           rmm::device_uvector<vertex_t>,                    // dsts
           std::optional<rmm::device_uvector<weight_t>>,     // weights
           std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
           std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
           std::optional<rmm::device_uvector<size_t>>>       // (label, hop) offsets to the edges
sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<std::tuple<rmm::device_uvector<int32_t>, size_t>>&& edgelist_hops,
  std::optional<std::tuple<raft::device_span<size_t const>, size_t>> edgelist_label_offsets,
  bool src_is_major       = true,
  bool do_expensive_check = false);

}  // namespace cugraph
