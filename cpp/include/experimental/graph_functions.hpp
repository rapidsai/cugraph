/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <utilities/error.hpp>

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <tuple>

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
 * @param edgelist_major_vertices Edge source vertex IDs (if the graph adjacency matrix is stored as
 * is) or edge destination vertex IDs (if the transposed graph adjacency matrix is stored). Vertex
 * IDs are updated in-place ([INOUT] parameter). Applying the compute_gpu_id_from_edge_t functor to
 * every (source, destination) pair should return the local GPU ID for this function to work (edges
 * should be pre-shuffled).
 * @param edgelist_minor_vertices Edge destination vertex IDs (if the graph adjacency matrix is
 * stored as is) or edge source vertex IDs (if the transposed graph adjacency matrix is stored).
 * Vertex IDs are updated in-place ([INOUT] parameter).
 * @param is_hypergraph_partitioned Flag indicating whether we are assuming hypergraph partitioning
 * (this flag will be removed in the future). Applying the compute_gpu_id_from_edge_t functor to
 * every (source, destination) pair should return the local GPU ID for this function to work (edges
 * should be pre-shuffled).
 * @return std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>
 * Quadruplet of labels (vertex IDs before renumbering) for the entire set of vertices (assigned to
 * this process in multi-GPU), partition_t object storing graph partitioning information, total
 * number of vertices, and total number of edges.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>, partition_t<vertex_t>, vertex_t, edge_t>>
renumber_edgelist(raft::handle_t const& handle,
                  rmm::device_uvector<vertex_t>& edgelist_major_vertices /* [INOUT] */,
                  rmm::device_uvector<vertex_t>& edgelist_minor_vertices /* [INOUT] */,
                  bool is_hypergraph_partitioned);

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
 * @return rmm::device_uvector<vertex_t> Labels (vertex IDs before renumbering) for the entire set
 * of vertices.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::enable_if_t<!multi_gpu, rmm::device_uvector<vertex_t>> renumber_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>& edgelist_major_vertices /* [INOUT] */,
  rmm::device_uvector<vertex_t>& edgelist_minor_vertices /* [INOUT] */);

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
  raft::handel_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> const& graph_view,
  vertex_t const* labels);

/**
 * @brief Relabel old labels to new labels.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param old_labels Old labels to be relabeled.
 * @param old_new_label_pairs Pairs of an old label and the corresponding new label (each process
 * holds only part of the entire old labels and the corresponding new labels; partitioning can be
 * arbitrary).
 * @return rmm::device_uvector<vertex_t> New labels corresponding to the @p old_labels.
 */
template <typename vertex_t, bool multi_gpu>
rmm::device_uvector<vertex_t> relabel(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t> const& old_labels,
  std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> const&
    old_new_label_pairs);

}  // namespace experimental
}  // namespace cugraph
