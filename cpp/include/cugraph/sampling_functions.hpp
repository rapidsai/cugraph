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

#include <cugraph/src_dst_lookup_container.hpp>

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
/*
 * @brief Build map to lookup source and destination using edge id and type
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_id_view View object holding edge ids of the edges of the graph pointed @p graph_view
 * @param edge_type_view View object holding edge types of the edges of the graph pointed @p
 * graph_view
 * @return An object of type cugraph::lookup_container_t that encapsulates edge id and type to
 * source and destination lookup map.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
lookup_container_t<edge_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>>
build_edge_id_and_type_to_src_dst_lookup_map(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, edge_t const*> edge_id_view,
  edge_property_view_t<edge_t, edge_type_t const*> edge_type_view);

/*
 * @brief Lookup edge sources and destinations using edge ids and a single edge type.
 * Use this function to lookup endpoints of edges belonging to the same edge type.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param search_container Object of type cugraph::lookup_container_t that encapsulates edge id and
 * type to source and destination lookup map.
 * @param edge_ids_to_lookup Device span of edge ids to lookup
 * @param edge_type_to_lookup Type of the edges corresponding to edge ids in @p edge_ids_to_lookup
 * @return A tuple of device vector containing edge sources and destinations for edge ids
 * in @p edge_ids_to_lookup and edge type @. If an edge id in @p edge_ids_to_lookup or
 * edge type @edge_type_to_lookup is not found, the corresponding entry in the device vectors of
 * the returned tuple will contain cugraph::invalid_vertex_id<vertex_t>.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_of_single_type(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>> const&
    search_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  edge_type_t edge_type_to_lookup);

/*
 * @brief Lookup edge sources and destinations using edge ids and edge types.
 * Use this function to lookup endpoints of edges belonging to different edge types.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam edge_type_t Type of edge types. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param search_container Object of type cugraph::lookup_container_t that encapsulates edge id and
 * type to source and destination lookup map.
 * @param edge_ids_to_lookup Device span of edge ids to lookup
 * @param edge_types_to_lookup Device span of edge types corresponding to the edge ids
 * in @p edge_ids_to_lookup
 * @return A tuple of device vector containing edge sources and destinations for the edge ids
 * in @p edge_ids_to_lookup and the edge types in @p edge_types_to_lookup. If an edge id in
 * @p edge_ids_to_lookup or edge type in @p edge_types_to_lookup is not found, the
 * corresponding entry in the device vectors of the returned tuple will contain
 * cugraph::invalid_vertex_id<vertex_t>.
 */
template <typename vertex_t, typename edge_t, typename edge_type_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
lookup_endpoints_for_edge_ids_and_types(
  raft::handle_t const& handle,
  lookup_container_t<edge_t, edge_type_t, thrust::tuple<vertex_t, vertex_t>> const&
    search_container,
  raft::device_span<edge_t const> edge_ids_to_lookup,
  raft::device_span<edge_type_t const> edge_types_to_lookup);

}  // namespace cugraph
