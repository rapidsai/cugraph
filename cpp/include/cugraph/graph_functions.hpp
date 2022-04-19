/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {

template <typename vertex_t, typename edge_t, bool multi_gpu, typename Enable = void>
struct renumber_meta_t;

template <typename vertex_t, typename edge_t, bool multi_gpu>
struct renumber_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<multi_gpu>> {
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};
  partition_t<vertex_t> partition{};
  std::vector<vertex_t> segment_offsets{};
};

template <typename vertex_t, typename edge_t, bool multi_gpu>
struct renumber_meta_t<vertex_t, edge_t, multi_gpu, std::enable_if_t<!multi_gpu>> {
  std::vector<vertex_t> segment_offsets{};
};

/**
 * @brief renumber edgelist (multi-GPU)
 *
 * This function assumes that vertices are pre-shuffled to their target processes and edges are
 * pre-shuffled to their target processess and edge partitions using compute_gpu_id_from_vertex_t
 * and compute_gpu_id_from_edge_t & compute_partition_id_from_edge_t functors, respectively.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param local_vertices If valid, part of the entire set of vertices in the graph to be renumbered.
 * This parameter can be used to include isolated vertices. Applying the
 * compute_gpu_id_from_vertex_t to every vertex should return the local GPU ID for this function to
 * work (vertices should be pre-shuffled).
 * @param edgelist_srcs Pointers (one pointer per local edge partition assigned to this process) to
 * edge source vertex IDs. Source IDs are updated in-place ([INOUT] parameter). Applying the
 * compute_gpu_id_from_edge_t functor to every (destination ID, source ID) pair (if store_transposed
 * = true) or (source ID, destination ID) pair (if store_transposed = false) should return the local
 * GPU ID for this function to work (edges should be pre-shuffled). Applying the
 * compute_partition_id_from_edge_t to every (destination ID, source ID) pair (if store_transposed =
 * true) or (source ID, destination ID) pair (if store_transposed = false) should also return the
 * corresponding edge partition ID. The best way to enforce this is to use
 * shuffle_edgelist_by_gpu_id & groupby_and_count_edgelist_by_local_partition_id.
 * @param edgelist_dsts Pointers (one pointer per local edge partition assigned to this process) to
 * edge destination vertex IDs. Destination IDs are updated in-place ([INOUT] parameter).
 * @param edgelist_edge_counts Edge counts (one count per local edge partition assigned to this
 * process).
 * @param edgelist_intra_partition_segment_offsets If valid, store segment offsets within a local
 * edge partition; a local edge partition can be further segmented by applying the
 * compute_gpu_id_from_vertex_t function to edge minor vertex IDs. This optinoal information is used
 * for further memory footprint optimization if provided.
 * @param store_transposed Should be true if renumbered edges will be used to create a graph with
 * store_transposed = true. Should be false if the edges will be used to create a graph with
 * store_transposed = false.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>
 * Tuple of labels (vertex IDs before renumbering) for the entire set of vertices (assigned to this
 * process in multi-GPU) and meta-data collected while renumbering. The meta-data includes total
 * number of vertices, total number of edges, partition_t object storing graph partitioning
 * information, vertex partition segment offsets (a vertex partition is partitioned to multiple
 * segments based on vertex degrees), and the number of local unique edge major & minor vertex IDs.
 * This meta-data is expected to be used in graph construction & graph primitives.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(
  raft::handle_t const& handle,
  std::optional<rmm::device_uvector<vertex_t>>&& local_vertices,
  std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<edge_t> const& edgelist_edge_counts,
  std::optional<std::vector<std::vector<edge_t>>> const& edgelist_intra_partition_segment_offsets,
  bool store_transposed,
  bool do_expensive_check = false);

/**
 * @brief renumber edgelist (single-GPU)
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices If valid, vertices in the graph to be renumbered. This parameter can be used to
 * include isolated vertices.
 * @param edgelist_srcs A pointer to edge source vertex IDs. Source IDs are updated in-place
 * ([INOUT] parameter).
 * @param edgelist_dsts A pointer to edge destination vertex IDs. Destination IDs are updated
 * in-place ([INOUT] parameter).
 * @param num_edgelist_edges Number of edges in the edgelist.
 * @param store_transposed Should be true if renumbered edges will be used to create a graph with
 * store_transposed = true. Should be false if the edges will be used to create a graph with
 * store_transposed = false.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>
 * Tuple of labels (vertex IDs before renumbering) for the entire set of vertices and meta-data
 * collected while renumbering. The meta-data includes vertex partition segment offsets (a vertex
 * partition is partitioned to multiple segments based on vertex degrees). This meta-data is
 * expected to be used in graph construction & graph primitives.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<rmm::device_uvector<vertex_t>, renumber_meta_t<vertex_t, edge_t, multi_gpu>>>
renumber_edgelist(raft::handle_t const& handle,
                  std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                  vertex_t* edgelist_srcs /* [INOUT] */,
                  vertex_t* edgelist_dsts /* [INOUT] */,
                  edge_t num_edgelist_edges,
                  bool store_transposed,
                  bool do_expensive_check = false);

/**
 * @brief Renumber external vertices to internal vertices based on the provided @p
 * renumber_map_labels.
 *
 * Note cugraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the vertices to be renumbered. The input external vertices are
 * renumbered to internal vertices in-place.
 * @param num_vertices Number of vertices to be renumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool multi_gpu>
void renumber_ext_vertices(raft::handle_t const& handle,
                           vertex_t* vertices /* [INOUT] */,
                           size_t num_vertices,
                           vertex_t const* renumber_map_labels,
                           vertex_t local_int_vertex_first,
                           vertex_t local_int_vertex_last,
                           bool do_expensive_check = false);

/**
 * @brief Unrenumber local internal vertices to external vertices based on the providied @p
 * renumber_map_labels.
 *
 * Note cugraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the local internal vertices to be unrenumbered. Each input element
 * should be in [@p local_int_vertex_first, @p local_int_vertex_last). The input internal vertices
 * are renumbered to external vertices in-place.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t>
void unrenumber_local_int_vertices(
  raft::handle_t const& handle,
  vertex_t* vertices /* [INOUT] */,
  size_t num_vertices,
  vertex_t const* renumber_map_labels /* size = local_int_vertex_last - local_int_vertex_first */,
  vertex_t local_int_vertex_first,
  vertex_t local_int_vertex_last,
  bool do_expensive_check = false);

/**
 * @brief Unrenumber (possibly non-local) internal vertices to external vertices based on the
 * providied @p renumber_map_labels.
 *
 * Note cugraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the internal vertices to be unrenumbered. The input internal vertices
 * are renumbered to external vertices in-place.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param vertex_partition_range_lasts Last local internal vertices (exclusive, assigned to each
 * process in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool multi_gpu>
void unrenumber_int_vertices(raft::handle_t const& handle,
                             vertex_t* vertices /* [INOUT] */,
                             size_t num_vertices,
                             vertex_t const* renumber_map_labels,
                             std::vector<vertex_t> const& vertex_partition_range_lasts,
                             bool do_expensive_check = false);

/**
 * @brief Unrenumber local edges' internal source & destination IDs to external IDs based on the
 * provided @p renumber_map_labels (multi-GPU).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @params edgelist_srcs Pointers (one pointer per local edge partition assigned to this process) to
 * the local internal source vertex IDs to be unrenumbered. The input  source vertex IDs are
 * renumbered to external IDs in-place.
 * @params edgelist_dsts Pointers (one pointer per local edge partition assigned to this process) to
 * the local internal destination vertex IDs to be unrenumbered. The input destination vertex IDs
 * are renumbered to external IDs in-place.
 * @param edgelist_edge_counts Edge counts (one count per local edge partition assigned to this
 * process).
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range assigned to this process.
 * @param vertex_partition_range_lasts Last local internal vertices (exclusive, assigned to each
 * process in multi-GPU).
 * @param edgelist_intra_partition_segment_offsets If valid, store segment offsets within a local
 * edge partition; a local edge partition can be further segmented by applying the
 * compute_gpu_id_from_vertex_t function to edge minor vertex IDs. This optinoal information is used
 * for further memory footprint optimization if provided.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<multi_gpu, void> unrenumber_local_int_edges(
  raft::handle_t const& handle,
  std::vector<vertex_t*> const& edgelist_srcs /* [INOUT] */,
  std::vector<vertex_t*> const& edgelist_dsts /* [INOUT] */,
  std::vector<size_t> const& edgelist_edge_counts,
  vertex_t const* renumber_map_labels,
  std::vector<vertex_t> const& vertex_partition_range_lasts,
  std::optional<std::vector<std::vector<size_t>>> const& edgelist_intra_partition_segment_offsets,
  bool do_expensive_check = false);

/**
 * @brief Unrenumber local edges' internal source & destination IDs to external IDs based on the
 * provided @p renumber_map_labels (single-GPU).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @params edgelist_srcs Pointer to the local internal source vertex IDs to be unrenumbered. The
 * input source vertex IDs are renumbered to external IDs in-place.
 * @params edgelist_dsts Pointer to the local internal destination vertex IDs to be unrenumbered.
 * The input destination vertex IDs are renumbered to external IDs in-place.
 * @param num_edgelist_edges Number of edges in the edge list.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices.
 * @param num_vertices Number of vertices to be unrenumbered.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool store_transposed, bool multi_gpu>
std::enable_if_t<!multi_gpu, void> unrenumber_local_int_edges(raft::handle_t const& handle,
                                                              vertex_t* edgelist_srcs /* [INOUT] */,
                                                              vertex_t* edgelist_dsts /* [INOUT] */,
                                                              size_t num_edgelist_edges,
                                                              vertex_t const* renumber_map_labels,
                                                              vertex_t num_vertices,
                                                              bool do_expensive_check = false);

/**
 * @brief Renumber local external vertices to internal vertices based on the provided @p
 * renumber_map_labels.
 *
 * Note cugraph::invalid_id<vertex_t>::value remains unchanged.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices Pointer to the vertices to be renumbered. The input external vertices are
 * renumbered to internal vertices in-place.
 * @param num_vertices Number of vertices to be renumbered.
 * @param renumber_map_labels Pointer to the external vertices corresponding to the internal
 * vertices in the range [@p local_int_vertex_first, @p local_int_vertex_last).
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool multi_gpu>
void renumber_local_ext_vertices(raft::handle_t const& handle,
                                 vertex_t* vertices /* [INOUT] */,
                                 size_t num_vertices,
                                 vertex_t const* renumber_map_labels,
                                 vertex_t local_int_vertex_first,
                                 vertex_t local_int_vertex_last,
                                 bool do_expensive_check = false);

/**
 * @brief Symmetrize edgelist.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_srcs Vector of edge source vertex IDs. If multi-GPU, applying the
 * compute_gpu_id_from_edge_t to every edge should return the local GPU ID for this function to work
 * (edges should be pre-shuffled).
 * @param edgelist_dsts Vector of edge destination vertex IDs.
 * @param edgelist_weights Vector of edge weights.
 * @param reciprocal Flag indicating whether to keep (if set to `false`) or discard (if set to
 * `true`) edges that appear only in one direction.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * std::optional<rmm::device_uvector<weight_t>>> Tuple of symmetrized sources, destinations, and
 * optional weights (if @p edgelist_weights is valid).
 */
template <typename vertex_t, typename weight_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist(raft::handle_t const& handle,
                    rmm::device_uvector<vertex_t>&& edgelist_srcs,
                    rmm::device_uvector<vertex_t>&& edgelist_dsts,
                    std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                    bool reciprocal);

/**
 * @brief Compute the coarsened graph.
 *
 * Aggregates the vertices with the same label to a new vertex in the output coarsened graph.
 * Multi-edges in the coarsened graph are collapsed to a single edge with its weight equal to the
 * sum of multi-edge weights.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to be coarsened.
 * @param labels Vertex labels (assigned to this process in multi-GPU) to be used in coarsening.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
 * rmm::device_uvector<vertex_t>> Tuple of the coarsened graph and labels mapped to the vertices
 * (assigned to this process in multi-GPU) in the coarsened graph.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>>
coarsen_graph(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels,
  bool do_expensive_check = false);

/**
 * @brief Relabel old labels to new labels.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param old_new_label_pairs Pairs of an old label and the corresponding new label (each process
 * holds only part of the entire old labels and the corresponding new labels; partitioning can be
 * arbitrary).
 * @param num_label_pairs Number of (old, new) label pairs.
 * @param labels Labels to be relabeled. This initially holds old labels. Old labels are updated to
 * new labels in-place ([INOUT] parameter).
 * @param num_labels Number of labels to be relabeled.
 * @param skip_missing_labels Flag dictating the behavior on missing labels (@p labels contains old
 * labels missing in @p old_new_label_pairs). If set to true, missing elements are skipped (not
 * relabeled). If set to false, undefined behavior (if @p do_expensive_check is set to true, this
 * function will throw an exception).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return rmm::device_uvector<vertex_t> New labels corresponding to the @p old_labels.
 */
template <typename vertex_t, bool multi_gpu>
void relabel(raft::handle_t const& handle,
             std::tuple<vertex_t const*, vertex_t const*> old_new_label_pairs,
             vertex_t num_label_pairs,
             vertex_t* labels /* [INOUT] */,
             vertex_t num_labels,
             bool skip_missing_labels,
             bool do_expensive_check = false);

/**
 * @brief extract induced subgraph(s).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object, we extract induced subgraphs from @p graph_view.
 * @param subgraph_offsets Pointer to subgraph vertex offsets (size == @p num_subgraphs + 1).
 * @param subgraph_vertices Pointer to subgraph vertices (size == @p subgraph_offsets[@p
 * num_subgraphs]). The elements of @p subgraph_vertices for each subgraph should be sorted in
 * ascending order and unique.
 * @param num_subgraphs Number of induced subgraphs to extract.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * rmm::device_uvector<weight_t>, rmm::device_uvector<size_t>> Quadraplet of edge major (destination
 * if @p store_transposed is true, source otherwise) vertices, edge minor (source if @p
 * store_transposed  is true, destination otherwise) vertices, edge weights, and edge offsets for
 * each induced subgraphs (size == num_subgraphs + 1). The sizes of the edge major & minor vertices
 * are edge_offsets[num_subgraphs]. The size of the edge weights is either
 * edge_offsets[num_subgraphs] (if @p graph_view is weighted) or 0 (if @p graph_view is unweighted).
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_induced_subgraphs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  size_t const* subgraph_offsets /* size == num_subgraphs + 1 */,
  vertex_t const* subgraph_vertices /* size == subgraph_offsets[num_subgraphs] */,
  size_t num_subgraphs,
  bool do_expensive_check = false);

/**
 * @brief create a graph from (the optional vertex list and) the given edge list.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam store_transposed Flag indicating whether to use sources (if false) or destinations (if
 * true) as major indices in storing edges using a 2D sparse matrix. transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices  If valid, part of the entire set of vertices in the graph to be renumbered.
 * This parameter can be used to include isolated vertices. If multi-GPU, applying the
 * compute_gpu_id_from_vertex_t to every vertex should return the local GPU ID for this function to
 * work (vertices should be pre-shuffled).
 * @param edgelist_srcs Vector of edge source vertex IDs. If multi-GPU, applying the
 * compute_gpu_id_from_edge_t to every edge should return the local GPU ID for this function to work
 * (edges should be pre-shuffled).
 * @param edgelist_dsts Vector of edge destination vertex IDs.
 * @param edgelist_weights Vector of edge weights.
 * @param graph_properties Properties of the graph represented by the input (optional vertex list
 * and) edge list.
 * @param renumber Flag indicating whether to renumber vertices or not.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed,
 * multi_gpu>, rmm::device_uvector<vertex_t>> Pair of the generated graph and the renumber map (if
 * @p renumber is true) or std::nullopt (if @p renumber is false).
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(raft::handle_t const& handle,
                           std::optional<rmm::device_uvector<vertex_t>>&& vertices,
                           rmm::device_uvector<vertex_t>&& edgelist_srcs,
                           rmm::device_uvector<vertex_t>&& edgelist_dsts,
                           std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
                           graph_properties_t graph_properties,
                           bool renumber,
                           bool do_expensive_check = false);

}  // namespace cugraph
