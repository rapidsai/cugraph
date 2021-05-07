/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>
#include <experimental/graph_view.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <tuple>
#include <vector>

namespace cugraph {
namespace experimental {

/**
 * @brief renumber edgelist (multi-GPU)
 *
 * This function assumes that edges are pre-shuffled to their target processes using the
 * compute_gpu_id_from_edge_t functor.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param edgelist_major_vertices Pointers (one pointer per local graph adjacency matrix partition
 * assigned to this process) to edge source vertex IDs (if the graph adjacency matrix is stored as
 * is) or edge destination vertex IDs (if the transposed graph adjacency matrix is stored). Vertex
 * IDs are updated in-place ([INOUT] parameter). Edges should be pre-shuffled to their final target
 * process & matrix partition; i.e. applying the compute_gpu_id_from_edge_t functor to every (major,
 * minor) pair should return the GPU ID of this process and applying the
 * compute_partition_id_from_edge_t fuctor to every (major, minor) pair for a local matrix partition
 * should return the partition ID of the corresponding matrix partition.
 * @param edgelist_minor_vertices Pointers (one pointer per local graph adjacency matrix partition
 * assigned to this process) to edge destination vertex IDs (if the graph adjacency matrix is stored
 * as is) or edge source vertex IDs (if the transposed graph adjacency matrix is stored). Vertex IDs
 * are updated in-place ([INOUT] parameter). Edges should be pre-shuffled to their final target
 * process & matrix partition; i.e. applying the compute_gpu_id_from_edge_t functor to every (major,
 * minor) pair should return the GPU ID of this process and applying the
 * compute_partition_id_from_edge_t fuctor to every (major, minor) pair for a local matrix partition
 * should return the partition ID of the corresponding matrix partition.
 * @param edgelist_edge_counts Edge counts (one count per local graph adjacency matrix partition
 * assigned to this process).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>
 * Quadruplet of labels (vertex IDs before renumbering) for the entire set of vertices (assigned to
 * this process in multi-GPU), partition_t object storing graph partitioning information, total
 * number of vertices, and total number of edges.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  std::vector<vertex_t*> const& edgelist_major_vertices /* [INOUT] */,
                  std::vector<vertex_t*> const& edgelist_minor_vertices /* [INOUT] */,
                  std::vector<edge_t> const& edgelist_edge_counts,
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
 * @param edgelist_major_vertices Edge source vertex IDs (if the graph adjacency matrix is stored as
 * is) or edge destination vertex IDs (if the transposed graph adjacency matrix is stored). Vertex
 * IDs are updated in-place ([INOUT] parameter).
 * @param edgelist_minor_vertices Edge destination vertex IDs (if the graph adjacency matrix is
 * stored as is) or edge source vertex IDs (if the transposed graph adjacency matrix is stored).
 * Vertex IDs are updated in-place ([INOUT] parameter).
 * @param num_edgelist_edges Number of edges in the edgelist.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return rmm::device_uvector<vertex_t> Labels (vertex IDs before renumbering) for the entire set
 * of vertices.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check = false);

/**
 * @brief renumber edgelist (multi-GPU)
 *
 * This version takes the vertex set in addition; this allows renumbering to include isolated
 * vertices. This function assumes that vertices and edges are pre-shuffled to their target
 * processes using the compute_gpu_id_from_vertex_t & compute_gpu_id_from_edge_t functors,
 * respectively.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param local_vertices Part of the entire set of vertices in the graph to be renumbered. Applying
 * the compute_gpu_id_from_vertex_t to every vertex should return the local GPU ID for this function
 * to work (vertices should be pre-shuffled).
 * @param num_local_vertices Number of local vertices.
 * @param edgelist_major_vertices Pointers (one pointer per local graph adjacency matrix partition
 * assigned to this process) to edge source vertex IDs (if the graph adjacency matrix is stored as
 * is) or edge destination vertex IDs (if the transposed graph adjacency matrix is stored). Vertex
 * IDs are updated in-place ([INOUT] parameter). Edges should be pre-shuffled to their final target
 * process & matrix partition; i.e. applying the compute_gpu_id_from_edge_t functor to every (major,
 * minor) pair should return the GPU ID of this process and applying the
 * compute_partition_id_from_edge_t fuctor to every (major, minor) pair for a local matrix partition
 * should return the partition ID of the corresponding matrix partition.
 * @param edgelist_minor_vertices Pointers (one pointer per local graph adjacency matrix partition
 * assigned to this process) to edge destination vertex IDs (if the graph adjacency matrix is stored
 * as is) or edge source vertex IDs (if the transposed graph adjacency matrix is stored). Vertex IDs
 * are updated in-place ([INOUT] parameter). Edges should be pre-shuffled to their final target
 * process & matrix partition; i.e. applying the compute_gpu_id_from_edge_t functor to every (major,
 * minor) pair should return the GPU ID of this process and applying the
 * compute_partition_id_from_edge_t fuctor to every (major, minor) pair for a local matrix partition
 * should return the partition ID of the corresponding matrix partition.
 * @param edgelist_edge_counts Edge counts (one count per local graph adjacency matrix partition
 * assigned to this process).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>
 * Quadruplet of labels (vertex IDs before renumbering) for the entire set of vertices (assigned to
 * this process in multi-GPU), partition_t object storing graph partitioning information, total
 * number of vertices, and total number of edges.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  vertex_t const* local_vertices,
                  vertex_t num_local_vertices,
                  std::vector<vertex_t*> const& edgelist_major_vertices /* [INOUT] */,
                  std::vector<vertex_t*> const& edgelist_minor_vertices /* [INOUT] */,
                  std::vector<edge_t> const& edgelist_edge_counts,
                  bool do_expensive_check = false);

/**
 * @brief renumber edgelist (single-GPU)
 *
 * This version takes the vertex set in addition; this allows renumbering to include isolated
 * vertices.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param vertices The entire set of vertices in the graph to be renumbered.
 * @param num_vertices Number of vertices.
 * @param edgelist_major_vertices Edge source vertex IDs (if the graph adjacency matrix is stored as
 * is) or edge destination vertex IDs (if the transposed graph adjacency matrix is stored). Vertex
 * IDs are updated in-place ([INOUT] parameter).
 * @param edgelist_minor_vertices Edge destination vertex IDs (if the graph adjacency matrix is
 * stored as is) or edge source vertex IDs (if the transposed graph adjacency matrix is stored).
 * Vertex IDs are updated in-place ([INOUT] parameter).
 * @param num_edgelist_edges Number of edges in the edgelist.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return rmm::device_uvector<vertex_t> Labels (vertex IDs before renumbering) for the entire set
 * of vertices.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  vertex_t const* vertices,
  vertex_t num_vertices,
  vertex_t* edgelist_major_vertices /* [INOUT] */,
  vertex_t* edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool do_expensive_check = false);

/**
 * @brief Renumber external vertices to internal vertices based on the provoided @p
 * renumber_map_labels.
 *
 * Note cugraph::experimental::invalid_id<vertex_t>::value remains unchanged.
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
 * Note cugraph::experimental::invalid_id<vertex_t>::value remains unchanged.
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

// FIXME: We may add unrenumber_int_rows(or cols) as this will require communication only within a
// sub-communicator and potentially be more efficient.
/**
 * @brief Unrenumber (possibly non-local) internal vertices to external vertices based on the
 * providied @p renumber_map_labels.
 *
 * Note cugraph::experimental::invalid_id<vertex_t>::value remains unchanged.
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
 * @param local_int_vertex_first The first local internal vertex (inclusive, assigned to this
 * process in multi-GPU).
 * @param local_int_vertex_last The last local internal vertex (exclusive, assigned to this process
 * in multi-GPU).
 * @param vertex_partition_lasts Last local internal vertices (exclusive, assigned to each process
 * in multi-GPU).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, bool multi_gpu>
void unrenumber_int_vertices(raft::handle_t const& handle,
                             vertex_t* vertices /* [INOUT] */,
                             size_t num_vertices,
                             vertex_t const* renumber_map_labels,
                             vertex_t local_int_vertex_first,
                             vertex_t local_int_vertex_last,
                             std::vector<vertex_t>& vertex_partition_lasts,
                             bool do_expensive_check = false);

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
 * @tparam store_transposed Flag indicating whether to store the graph adjacency matrix as is or as
 * transposed.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object of the input graph to be coarsened.
 * @param labels Vertex labels (assigned to this process in multi-GPU) to be used in coarsening.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed,
 * multi_gpu>>, rmm::device_uvector<vertex_t>> Tuple of the coarsened graph and labels mapped to the
 * vertices (assigned to this process in multi-GPU) in the coarsened graph.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<std::unique_ptr<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>>,
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
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return rmm::device_uvector<vertex_t> New labels corresponding to the @p old_labels.
 */
template <typename vertex_t, bool multi_gpu>
void relabel(raft::handle_t const& handle,
             std::tuple<vertex_t const*, vertex_t const*> old_new_label_pairs,
             vertex_t num_label_pairs,
             vertex_t* labels /* [INOUT] */,
             vertex_t num_labels,
             bool do_expensive_check = false);

/**
 * @brief extract induced subgraph(s).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights.
 * @tparam store_transposed Flag indicating whether to store the graph adjacency matrix as is or as
 * transposed.
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
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<size_t>>
extract_induced_subgraphs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  size_t const* subgraph_offsets /* size == num_subgraphs + 1 */,
  vertex_t const* subgraph_vertices /* size == subgraph_offsets[num_subgraphs] */,
  size_t num_subgraphs,
  bool do_expensive_check = false);

}  // namespace experimental
}  // namespace cugraph
