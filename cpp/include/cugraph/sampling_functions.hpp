/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs. If
 * @p seed_vertices.has_value() is true, we assume (hop=0, flag=major) for every vertex in @p
 * *seed_vertices in renumbering (this is relevant when there are seed vertices with no neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
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
 * compress_per_hop is false or for hop 0 (@p seed_vertices should be included if valid). If @p
 * compress_per_hop is true and hop number is 1 or larger, the maximum vertex ID is the larger of
 * the maximum major vertex ID for this hop and the maximum vertex ID for the edges in the previous
 * hops.
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
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
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
 * (D)CSR|(D)CSC offset array (size = @p num_labels * @p num_hops + 1, valid only when @p
 * edgelist_hops.has_value() or @p edgelist_label_offsets.has_value() is true), renumber_map to
 * query original vertices (size = # unique or aggregate # unique_vertices for each label), and
 * label offsets to the renumber_map (size = num_labels + 1, valid only if @p
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
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
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
 * in renumbering. Ordering can be arbitrary among the vertices with the same (hop, flag) pairs. If
 * @p seed_vertices.has-value() is true, we assume (hop=0, flag=major) for every vertex in @p
 * *seed_vertices in renumbering (this is relevant when there are seed vertices with no neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
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
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the renumbered
 * and sorted edges (size = @p num_labels * @p num_hops + 1, valid only when @p
 * edgelist_hops.has_value() or @p edgelist_label_offsetes.has_value() is true), renumber_map to
 * query original vertices (size = # unique or aggregate # unique vertices for each label), and
 * label offsets to the renumber map (size = @p num_labels + 1, valid only if @p
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
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  size_t num_labels,
  size_t num_hops,
  bool src_is_major       = true,
  bool do_expensive_check = false);

/*
 * @brief renumber sampled edge list (vertex & edge IDs) per vertex/edge type and sort the
 * renumbered edges.
 *
 * This function renumbers sampling function (e.g. uniform_neighbor_sample) output edge
 * source/destination vertex IDs fulfilling the following requirements. Assume major = source if @p
 * src_is_major is true, major = destination if @p src_is_major is false.
 *
 * 1. If @p edgelist_hops is valid, we can consider (vertex ID, hop, flag=major) triplets for each
 * vertex ID in edge majors (@p edgelist_srcs if @p src_is_major is true, @p edgelist_dsts if false)
 * and (vertex ID, hop, flag=minor) triplets for each vertex ID in edge minors. From these triplets,
 * we can find the minimum (hop, flag) pairs for every unique vertex ID (hop is the primary key and
 * flag is the secondary key, flag=major is considered smaller than flag=minor if hop numbers are
 * same). Vertex IDs with smaller (hop, flag) pairs precede vertex IDs with larger (hop, flag) pairs
 * in renumbering (if their vertex types are same, vertices with different types are renumbered
 * separately). Ordering can be arbitrary among the vertices with the same (vertex type, hop, flag)
 * triplets. If @p seed_vertices.has-value() is true, we assume (hop=0, flag=major) for every vertex
 * in @p *seed_vertices in renumbering (this is relevant when there are seed vertices with no
 * neighbors).
 * 2. If @p edgelist_hops is invalid, unique vertex IDs in edge majors precede vertex IDs that
 * appear only in edge minors. If @p seed_vertices.has_value() is true, vertices in @p
 * *seed_vertices precede vertex IDs that appear only in edge minors as well.
 * 3. Vertices with different types will be renumbered separately. Unique vertex IDs for each vertex
 * type are mapped to consecutive integers starting from 0.
 * 4. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * Edge IDs are renumbered fulfilling the following requirements (This is relevant only when @p
 * edgelist_edge_ids.has_value() is true).
 *
 * 1. If @p edgelist_edge_types.has_value() is true, unique (edge type, edge ID) pairs are
 * renumbered to consecutive integers starting from 0 for each edge type. If @p
 * edgelist_edge_types.has_value() is true, unique edge IDs are renumbered to consecutive inetgers
 * starting from 0.
 * 2. If edgelist_label_offsets.has_value() is true, edge lists for different labels will be
 * renumbered separately.
 *
 * The renumbered edges are sorted based on the following rules.
 *
 * 1. If @p src_is_major is true, use ((edge type), (hop), src, dst) as the key in sorting. If @p
 * src_is_major is false, use ((edge type), (hop), dst, src) instead. edge type is used only if @p
 * edgelist_edge_types.has_value() is true. hop is used only if @p edgelist_hops.has_value() is
 * true.
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
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops should be valid if @p num_hops >= 2.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets should be valid if @p num_labels
 * >= 2.
 * @param seed_vertices An optional pointer to the array storing seed vertices in hop 0.
 * @param seed_vertex_label_offsets An optional pointer to the array storing label offsets to the
 * seed vertices (size = @p num_labels + 1). @p seed_vertex_label_offsets should be valid if @p
 * num_labels >= 2 and @p seed_vertices is valid and invalid otherwise.
 * ext_vertices A pointer to the array storing external vertex IDs for the local internal vertices.
 * The local internal vertex range can be obatined bgy invoking a graph_view_t object's
 * local_vertex_partition_range() function. ext_vertex_type offsets A pointer to the array storing
 * vertex type offsets for the entire external vertex ID range (array size = @p num_vertex_types +
 * 1). For example, if the array stores [0, 100, 200], external vertex IDs [0, 100) has vertex type
 * 0 and external vertex IDs [100, 200) has vertex type 1.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param num_vertex_types Number of vertex types.
 * @param num_edge_types Number of edge types.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), optional (label, hop) offset values to the renumbered
 * and sorted edges (size = @p num_labels * @p num_hops + 1, valid only when @p
 * edgelist_hops.has_value() or @p edgelist_label_offsetes.has_value() is true), renumber_map to
 * query original vertices (size = # unique or aggregate # unique vertices for each label), and
 * label offsets to the renumber map (size = @p num_labels + 1, valid only if @p
 * edgelist_label_offsets.has_value() is true).
 */
template <typename vertex_t,
          typename weight_t,
          typename edge_id_t,
          typename edge_type_t>
std::tuple<
  rmm::device_uvector<vertex_t>,                    // srcs
  rmm::device_uvector<vertex_t>,                    // dsts
  std::optional<rmm::device_uvector<weight_t>>,     // weights
  std::optional<rmm::device_uvector<edge_id_t>>,    // edge IDs
  std::optional<rmm::device_uvector<edge_type_t>>,  // edge types
  std::optional<rmm::device_uvector<size_t>>,       // (label, edge type, hop) offsets to the edges
  rmm::device_uvector<vertex_t>,                    // vertex renumber map
  std::optional<rmm::device_uvector<size_t>>,  // (label, type) offsets to the vertex renumber map
  std::optional<rmm::device_uvector<edge_id_t>>,  // edge ID renumber map
  std::optional<rmm::device_uvector<size_t>>>  // (label, type) offsets to the edge ID renumber map
heterogeneous_renumber_and_sort_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& edgelist_srcs,
  rmm::device_uvector<vertex_t>&& edgelist_dsts,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
  std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
  std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
  std::optional<raft::device_span<vertex_t const>> seed_vertices,
  std::optional<raft::device_span<size_t const>> seed_vertex_label_offsets,
  raft::device_span<vertex_t const> ext_vertices,
  raft::device_span<vertex_t const> ext_vertex_type_offsets,
  size_t num_labels,
  size_t num_hops,
  size_t num_vertex_types,
  size_t num_edge_types,
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
 * @param edgelist_hops An optional vector storing edge list hop numbers (size = @p
 * edgelist_srcs.size() if valid). @p edgelist_hops must be valid if @p num_hops >= 2.
 * @param edgelist_label_offsets An optional pointer to the array storing label offsets to the input
 * edges (size = @p num_labels + 1). @p edgelist_label_offsets must be valid if @p num_labels >= 2.
 * @param num_labels Number of labels. Labels are considered if @p num_labels >=2 and ignored if @p
 * num_labels = 1.
 * @param num_hops Number of hops. Hop numbers are considered if @p num_hops >=2 and ignored if @p
 * num_hops = 1.
 * @param src_is_major A flag to determine whether to use the source or destination as the
 * major key in renumbering and sorting.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of vectors storing edge sources, edge destinations, optional edge weights (valid
 * only if @p edgelist_weights.has_value() is true), optional edge IDs (valid only if @p
 * edgelist_edge_ids.has_value() is true), optional edge types (valid only if @p
 * edgelist_edge_types.has_value() is true), and optional (label, hop) offset values to the sorted
 * edges (size = @p num_labels * @p num_hops + 1, valid only when @p edgelist_hops.has_value() or @p
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
           std::optional<rmm::device_uvector<size_t>>>       // (label, hop) offsets to the edges
sort_sampled_edgelist(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t>&& edgelist_srcs,
                      rmm::device_uvector<vertex_t>&& edgelist_dsts,
                      std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                      std::optional<rmm::device_uvector<edge_id_t>>&& edgelist_edge_ids,
                      std::optional<rmm::device_uvector<edge_type_t>>&& edgelist_edge_types,
                      std::optional<rmm::device_uvector<int32_t>>&& edgelist_hops,
                      std::optional<raft::device_span<size_t const>> edgelist_label_offsets,
                      size_t num_labels,
                      size_t num_hops,
                      bool src_is_major       = true,
                      bool do_expensive_check = false);

}  // namespace cugraph
