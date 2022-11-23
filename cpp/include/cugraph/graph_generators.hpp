/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <optional>
#include <tuple>

namespace cugraph {

/**
 * @brief generate an edge list for an R-mat graph.
 *
 * This function allows multi-edges and self-loops similar to the Graph 500 reference
 * implementation.
 *
 * NOTE: The scramble_vertex_ids function needs to be called in order to generate a
 * graph conforming to the Graph 500 specification (note that scrambling does not
 * affect cuGraph's graph construction performance, so this is generally unnecessary).
 * If `edge_factor` is given (e.g. Graph 500), set @p num_edges to
 * (size_t{1} << @p scale) * `edge_factor`. To generate an undirected graph, set @p b == @p c and @p
 * clip_and_flip = true. All the resulting edges will be placed in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix.
 *
 * For multi-GPU generation with `P` GPUs, @p seed should be set to different values in different
 * GPUs to avoid every GPU generating the same set of edges. @p num_edges should be adjusted as
 * well; e.g. assuming `edge_factor` is given, set @p num_edges = (size_t{1} << @p scale) *
 * `edge_factor` / `P` + (rank < (((size_t{1} << @p scale) * `edge_factor`) % P) ? 1 : 0).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param scale Scale factor to set the number of verties in the graph. Vertex IDs have values in
 * [0, V), where V = 1 << @p scale.
 * @param num_edges Number of edges to generate.
 * @param a a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param b a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param c a, b, c, d (= 1.0 - (a + b + c)) in the R-mat graph generator (vist https://graph500.org
 * for additional details). a, b, c, d should be non-negative and a + b + c should be no larger
 * than 1.0.
 * @param seed Seed value for the random number generator.
 * @param clip_and_flip Flag controlling whether to generate edges only in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix (if set to `true`) or not (if set to
 * `false`).
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> generate_rmat_edgelist(
  raft::handle_t const& handle,
  size_t scale,
  size_t num_edges,
  double a           = 0.57,
  double b           = 0.19,
  double c           = 0.19,
  uint64_t seed      = 0,
  bool clip_and_flip = false);

enum class generator_distribution_t { POWER_LAW = 0, UNIFORM };

/**
 * @brief generate multiple edge lists using the R-mat graph generator.
 *
 * This function allows multi-edges and self-loops similar to the Graph 500 reference
 * implementation.
 *
 * NOTE: The scramble_vertex_ids function needs to be called in order to generate a
 * graph conforming to the Graph 500 specification (note that scrambling does not
 * affect cuGraph's graph construction performance, so this is generally unnecessary).
 * If `edge_factor` is given (e.g. Graph 500), set @p num_edges to
 * (size_t{1} << @p scale) * `edge_factor`. To generate an undirected graph, set @p b == @p c and @p
 * clip_and_flip = true. All the resulting edges will be placed in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param n_edgelists Number of edge lists (graphs) to generate
 * @param min_scale Scale factor to set the minimum number of verties in the graph.
 * @param max_scale Scale factor to set the maximum number of verties in the graph.
 * @param edge_factor Average number of edges per vertex to generate.
 * @param size_distribution Distribution of the graph sizes, impacts the scale parameter of the
 * R-MAT generator
 * @param edge_distribution Edges distribution for each graph, impacts how R-MAT parameters a,b,c,d,
 * are set.
 * @param seed Seed value for the random number generator.
 * @param clip_and_flip Flag controlling whether to generate edges only in the lower triangular part
 * (including the diagonal) of the graph adjacency matrix (if set to `true`) or not (if set to
 * `false`).
 * @return A vector of std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> of
 *size @p n_edgelists, each vector element being a tuple of rmm::device_uvector objects for edge
 *source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::vector<std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>>
generate_rmat_edgelists(
  raft::handle_t const& handle,
  size_t n_edgelists,
  size_t min_scale,
  size_t max_scale,
  size_t edge_factor                         = 16,
  generator_distribution_t size_distribution = generator_distribution_t::POWER_LAW,
  generator_distribution_t edge_distribution = generator_distribution_t::POWER_LAW,
  uint64_t seed                              = 0,
  bool clip_and_flip                         = false);

/**
 * @brief generate an edge list for path graph
 *
 * A path graph of size n connects the vertices from 0 to (n - 1)
 * in a single long path: ((0,1), (1,2), ..., (n - 2, n - 1)
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * the path will span all GPUs including an edge from the last vertex on
 * GPU i to the first vertex on GPU (i+1)
 *
 * This function will generate a collection of path graphs.  @p component_parameters_v
 * defines the parameters for generating each component.  Each element of
 * @p component_parameters_v defines a tuple consisting of the number of vertices
 * and the base vertex id for the component.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param component_parameters_v A vector containing tuples consisting of the number of vertices and
 * base vertex id for each component to generate.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_path_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t>> const& component_parameters_v);

/**
 * @brief generate an edge list for a 2D Mesh Graph
 *
 * A sequence of 2D mesh graphs will be constructed according to the
 * component specifications.  Each 2D mesh graph is configured with a tuple
 * containing (x, y, base_vertex_id).  @p component_parameters_v will contain
 * a tuple for each component.
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will generate disjoint 2D mesh constructs of equal size.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param component_parameters_v Vector containing tuple defining the configuration of each
 * component
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_2d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t>> const& component_parameters_v);

/**
 * @brief generate an edge list for a 3D Mesh Graph
 *
 * A sequence of 3D mesh graphs will be constructed according to the
 * component specifications.  Each 3D mesh graph is configured with a tuple
 * containing (x, y, z, base_vertex_id).  @p component_parameters_v will contain
 * a tuple for each component.
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will generate disjoint 3D mesh constructs of equal size.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param component_parameters_v Vector containing tuple defining the configuration of each
 * component
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_3d_mesh_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t, vertex_t, vertex_t>> const& component_parameters_v);

/**
 * @brief generate an edge lists for some complete graphs
 *
 * A sequence of complete graphs will be constructed according to the
 * component specifications.  Each complete graph is configured with a tuple
 * containing (n, base_vertex_id).  @p component_parameters_v will contain
 * a tuple for each component.
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will generate disjoint complete graph constructs of equal size.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param component_parameters_v Vector containing tuple defining the configuration of each
 * component
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_complete_graph_edgelist(
  raft::handle_t const& handle,
  std::vector<std::tuple<vertex_t, vertex_t>> const& component_parameters_v);

/**
 * @brief generate an edge lists for an Erdos-Renyi graph
 *
 * This API supports the G(n,p) model which requires O(n^2) work.
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will generate Erdos-Renyi edges for its portion of the 2D
 * partitioning of the adjacency matrix.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param num_vertices Number of vertices to use in the generated graph
 * @param p Probability for edge creation
 * @param base_vertex_id Starting vertex id for the generated graph
 * @param seed Seed value for the random number generator.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnp(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        float p,
                                        vertex_t base_vertex_id,
                                        uint64_t seed = 0);

/**
 * @brief generate an edge lists for an Erdos-Renyi graph
 *
 * This API supports the G(n,m) model
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will generate Erdos-Renyi edges for its portion of the 2D
 * partitioning of the adjacency matrix.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param num_vertices Number of vertices to use in each complete graph
 * @param m Number of edges to generate
 * @param base_vertex_id Starting vertex id for the generated graph
 * @param seed Seed value for the random number generator.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
generate_erdos_renyi_graph_edgelist_gnm(raft::handle_t const& handle,
                                        vertex_t num_vertices,
                                        size_t m,
                                        vertex_t base_vertex_id,
                                        uint64_t seed = 0);

/**
 * @brief symmetrize an edgelist from the edges in the lower (or upper but not both) triangular part
 * of a graph adjacency matrix
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam weight_t Type of weights.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param d_src_v Vector of source vertices
 * @param d_dst_v Vector of destination vertices
 * @param d_weights_v Optional vector of edge weights
 * @param check_diagonal Flag indicating whether to check for diagonal edges or not. If set to true,
 * symmetrize only the edges with source != destination (to avoid duplicating every self-loops).
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> A tuple of
 * rmm::device_uvector objects for edge source vertex IDs and edge destination vertex IDs.
 */
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
symmetrize_edgelist_from_triangular(
  raft::handle_t const& handle,
  rmm::device_uvector<vertex_t>&& d_src_v,
  rmm::device_uvector<vertex_t>&& d_dst_v,
  std::optional<rmm::device_uvector<weight_t>>&& optional_d_weights_v,
  bool check_diagonal = false);

/**
 * @brief scramble vertex ids in a graph
 *
 * Given an edgelist for a graph, scramble all vertex ids by the given offset.
 * This translation is done in place.
 *
 * The scramble code here follows the algorithm in the Graph 500 reference
 * implementation version 3.0.0.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param d_src_v Vector of source vertices
 * @param d_dst_v Vector of destination vertices
 * @param vertex_id_offset Offset to add to each vertex id
 * @param seed Used to initialize random number generator
 */
template <typename vertex_t>
void scramble_vertex_ids(raft::handle_t const& handle,
                         rmm::device_uvector<vertex_t>& d_src_v,
                         rmm::device_uvector<vertex_t>& d_dst_v,
                         vertex_t vertex_id_offset,
                         uint64_t seed = 0);

/**
 * @brief Combine edgelists from multiple sources into a single edgelist
 *
 * If executed in a multi-gpu context (handle comms has been initialized)
 * each GPU will operate only on its subset of data.  Any shuffling to get
 * edges onto the same GPU should be done prior to calling this function.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param sources The source vertex ids to combine
 * @param dests The destination vertex ids to combine
 * @param weights Optional vector of weights to combine
 * @param remove_multi_edges If true (the default) then remove multi edges, if false leave them in
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * rmm::device_uvector<weight_t>> A tuple of rmm::device_uvector objects for edge source vertex IDs
 * and edge destination vertex IDs and edge weights.
 */
template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
combine_edgelists(raft::handle_t const& handle,
                  std::vector<rmm::device_uvector<vertex_t>>&& d_sources,
                  std::vector<rmm::device_uvector<vertex_t>>&& d_dests,
                  std::optional<std::vector<rmm::device_uvector<weight_t>>>&& optional_d_weights,
                  bool remove_multi_edges = true);

}  // namespace cugraph
