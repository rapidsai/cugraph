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

#include <cugraph/api_helpers.hpp>

#include <cugraph/dendrogram.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>

#include <cugraph/legacy/graph.hpp>
#include <cugraph/legacy/internals.hpp>

#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <raft/core/device_span.hpp>
#include <raft/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <optional>
#include <tuple>
#include <variant>

/** @ingroup cpp_api
 *  @{
 */

namespace cugraph {

/**
 * @brief     Compute jaccard similarity coefficient for all vertices
 *
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      device pointer to input vertex weights for weighted Jaccard, may be NULL
 * for unweighted Jaccard.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void jaccard(legacy::GraphCSRView<VT, ET, WT> const& graph, WT const* weights, WT* result);

/**
 * @brief     Compute jaccard similarity coefficient for selected vertex pairs
 *
 * Computes the Jaccard similarity coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      The input vertex weights for weighted Jaccard, may be NULL for
 *                         unweighted Jaccard.
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void jaccard_list(legacy::GraphCSRView<VT, ET, WT> const& graph,
                  WT const* weights,
                  ET num_pairs,
                  VT const* first,
                  VT const* second,
                  WT* result);

/**
 * @brief     Compute overlap coefficient for all vertices in the graph
 *
 * Computes the Overlap Coefficient for every pair of vertices in the graph which are
 * connected by an edge.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      device pointer to input vertex weights for weighted overlap, may be NULL
 * for unweighted overlap.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void overlap(legacy::GraphCSRView<VT, ET, WT> const& graph, WT const* weights, WT* result);

/**
 * @brief     Compute overlap coefficient for select pairs of vertices
 *
 * Computes the overlap coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam VT              Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET              Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT              Type of edge weights. Supported value : float or double.
 *
 * @param[in] graph        The input graph object
 * @param[in] weights      device pointer to input vertex weights for weighted overlap, may be NULL
 * for unweighted overlap.
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by
 * caller
 */
template <typename VT, typename ET, typename WT>
void overlap_list(legacy::GraphCSRView<VT, ET, WT> const& graph,
                  WT const* weights,
                  ET num_pairs,
                  VT const* first,
                  VT const* second,
                  WT* result);

/**
 *
 * @brief                                       ForceAtlas2 is a continuous graph layout algorithm
 * for handy network visualization.
 *
 *                                              NOTE: Peak memory allocation occurs at 17*V.
 *
 * @throws                                      cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                                   Type of vertex identifiers. Supported value :
 * int (signed, 32-bit)
 * @tparam edge_t                                   Type of edge identifiers.  Supported value : int
 * (signed, 32-bit)
 * @tparam weight_t                                   Type of edge weights. Supported values : float
 * or double.
 *
 * @param[in] handle                          Library handle (RAFT). If a communicator is set in the
 * handle, the multi GPU version will be selected.
 * @param[in] graph                             cuGraph graph descriptor, should contain the
 * connectivity information as a COO. Graph is considered undirected. Edge weights are used for this
 * algorithm and set to 1 by default.
 * @param[out] pos                              Device array (2, n) containing x-axis and y-axis
 * positions;
 * @param[in] max_iter                          The maximum number of iterations Force Atlas 2
 * should run for.
 * @param[in] x_start                           Device array containing starting x-axis positions;
 * @param[in] y_start                           Device array containing starting y-axis positions;
 * @param[in] outbound_attraction_distribution  Distributes attraction along outbound edges. Hubs
 * attract less and thus are pushed to the borders.
 * @param[in] lin_log_mode                      Switch ForceAtlas’ model from lin-lin to lin-log
 * (tribute to Andreas Noack). Makes clusters more tight.
 * @param[in] prevent_overlapping               Prevent nodes from overlapping.
 * @param[in] edge_weight_influence             How much influence you give to the edges weight. 0
 * is “no influence” and 1 is “normal”.
 * @param[in] jitter_tolerance                  How much swinging you allow. Above 1 discouraged.
 * Lower gives less speed and more precision.
 * @param[in] barnes_hut_optimize:              Whether to use the Barnes Hut approximation or the
 * slower exact version.
 * @param[in] barnes_hut_theta:                 Float between 0 and 1. Tradeoff for speed (1) vs
 * accuracy (0) for Barnes Hut only.
 * @params[in] scaling_ratio                    Float strictly positive. How much repulsion you
 * want. More makes a more sparse graph. Switching from regular mode to LinLog mode needs a
 * readjustment of the scaling parameter.
 * @params[in] strong_gravity_mode              Sets a force
 * that attracts the nodes that are distant from the center more. It is so strong that it can
 * sometimes dominate other forces.
 * @params[in] gravity                          Attracts nodes to the center. Prevents islands from
 * drifting away.
 * @params[in] verbose                          Output convergence info at each interation.
 * @params[in] callback                         An instance of GraphBasedDimRedCallback class to
 * intercept the internal state of positions while they are being trained.
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void force_atlas2(raft::handle_t const& handle,
                  legacy::GraphCOOView<vertex_t, edge_t, weight_t>& graph,
                  float* pos,
                  const int max_iter                            = 500,
                  float* x_start                                = nullptr,
                  float* y_start                                = nullptr,
                  bool outbound_attraction_distribution         = true,
                  bool lin_log_mode                             = false,
                  bool prevent_overlapping                      = false,
                  const float edge_weight_influence             = 1.0,
                  const float jitter_tolerance                  = 1.0,
                  bool barnes_hut_optimize                      = true,
                  const float barnes_hut_theta                  = 0.5,
                  const float scaling_ratio                     = 2.0,
                  bool strong_gravity_mode                      = false,
                  const float gravity                           = 1.0,
                  bool verbose                                  = false,
                  internals::GraphBasedDimRedCallback* callback = nullptr);

/**
 * @brief     Compute betweenness centrality for a graph
 *
 * Betweenness centrality for a vertex is the sum of the fraction of
 * all pairs shortest paths that pass through the vertex.
 *
 * The current implementation does not support a weighted graph.
 *
 * @throws                                  cugraph::logic_error if `result == nullptr` or
 * `number_of_sources < 0` or `number_of_sources !=0 and sources == nullptr`.
 * @tparam vertex_t                               Type of vertex identifiers. Supported value : int
 * (signed, 32-bit)
 * @tparam edge_t                               Type of edge identifiers.  Supported value : int
 * (signed, 32-bit)
 * @tparam weight_t                               Type of edge weights. Supported values : float or
 * double.
 * @tparam result_t                         Type of computed result.  Supported values :  float or
 * double
 * @param[in] handle                        Library handle (RAFT). If a communicator is set in the
 * handle, the multi GPU version will be selected.
 * @param[in] graph                         cuGRAPH graph descriptor, should contain the
 * connectivity information as a CSR
 * @param[out] result                       Device array of centrality scores
 * @param[in] normalized                    If true, return normalized scores, if false return
 * unnormalized scores.
 * @param[in] endpoints                     If true, include endpoints of paths in score, if false
 * do not
 * @param[in] weight                        If specified, device array of weights for each edge
 * @param[in] k                             If specified, number of vertex samples defined in the
 * vertices array.
 * @param[in] vertices                      If specified, host array of vertex ids to estimate
 * betweenness these vertices will serve as sources for the traversal
 * algorihtm to obtain shortest path counters.
 * @param[in] total_number_of_source_used   If specified use this number to normalize results
 * when using subsampling, it allows accumulation of results across multiple calls.
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void betweenness_centrality(const raft::handle_t& handle,
                            legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                            result_t* result,
                            bool normalized          = true,
                            bool endpoints           = false,
                            weight_t const* weight   = nullptr,
                            vertex_t k               = 0,
                            vertex_t const* vertices = nullptr);

/**
 * @brief     Compute edge betweenness centrality for a graph
 *
 * Betweenness centrality of an edge is the sum of the fraction of all-pairs shortest paths that
 * pass through this edge. The weight parameter is currenlty not supported
 *
 * @throws                                  cugraph::logic_error if `result == nullptr` or
 * `number_of_sources < 0` or `number_of_sources !=0 and sources == nullptr` or `endpoints ==
 * true`.
 * @tparam vertex_t                               Type of vertex identifiers. Supported value : int
 * (signed, 32-bit)
 * @tparam edge_t                               Type of edge identifiers.  Supported value : int
 * (signed, 32-bit)
 * @tparam weight_t                               Type of edge weights. Supported values : float or
 * double.
 * @tparam result_t                         Type of computed result.  Supported values :  float or
 * double
 * @param[in] handle                        Library handle (RAFT). If a communicator is set in the
 * handle, the multi GPU version will be selected.
 * @param[in] graph                         cuGraph graph descriptor, should contain the
 * connectivity information as a CSR
 * @param[out] result                       Device array of centrality scores
 * @param[in] normalized                    If true, return normalized scores, if false return
 * unnormalized scores.
 * @param[in] weight                        If specified, device array of weights for each edge
 * @param[in] k                             If specified, number of vertex samples defined in the
 * vertices array.
 * @param[in] vertices                      If specified, host array of vertex ids to estimate
 * betweenness these vertices will serve as sources for the traversal
 * algorihtm to obtain shortest path counters.
 * @param[in] total_number_of_source_used   If specified use this number to normalize results
 * when using subsampling, it allows accumulation of results across multiple calls.
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void edge_betweenness_centrality(const raft::handle_t& handle,
                                 legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                 result_t* result,
                                 bool normalized          = true,
                                 weight_t const* weight   = nullptr,
                                 vertex_t k               = 0,
                                 vertex_t const* vertices = nullptr);

/**
 * @brief     Compute betweenness centrality for a graph
 *
 * Betweenness centrality for a vertex is the sum of the fraction of
 * all pairs shortest paths that pass through the vertex.
 *
 * The current implementation does not support a weighted graph.
 *
 * If @p vertices is an optional variant.  If it is not specified the algorithm
 * will compute exact betweenness (compute betweenness using a traversal from all vertices).
 *
 * If @p vertices is specified as a vertex_t, it will compute approximate betweenness by
 * random sampling @p vertices as the seeds of the traversals.
 *
 * If @p vertices is specified as a device_span, it will compute approximate betweenness
 * using the provided @p vertices as the seeds of the traversals.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. Currently,
 * edge_weight_view.has_value() should be false as we don't support weighted graphs, yet.
 * @param vertices Optional, if specified this provides either a vertex_t count of how many
 *         random seeds to select, or a device_span identifying a list of pre-selected vertices
 *         to use as seeds for the traversals for approximating betweenness.
 * @param normalized         A flag indicating results should be normalized
 * @param include_endpoints  A flag indicating whether endpoints of a path should be counted
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 *
 * @return device vector containing the centralities.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<std::variant<vertex_t, raft::device_span<vertex_t const>>> vertices,
  bool const normalized         = true,
  bool const include_endpoints  = false,
  bool const do_expensive_check = false);

/**
 * @brief     Compute edge betweenness centrality for a graph
 *
 * Betweenness centrality of an edge is the sum of the fraction of all-pairs shortest paths that
 * pass through this edge. The weight parameter is currenlty not supported
 *
 * If @p vertices is an optional variant.  If it is not specified the algorithm
 * will compute exact betweenness (compute betweenness using a traversal from all vertices).
 *
 * If @p vertices is specified as a vertex_t, it will compute approximate betweenness by
 * random sampling @p vertices as the seeds of the traversals.
 *
 * If @p vertices is specified as a device_span, it will compute approximate betweenness
 * using the provided @p vertices as the seeds of the traversals.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 *
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. Currently,
 * edge_weight_view.has_value() should be false as we don't support weighted graphs, yet.
 * @param vertices Optional, if specified this provides either a vertex_t count of how many
 *         random seeds to select, or a device_span identifying a list of pre-selected vertices
 *         to use as seeds for the traversals for approximating betweenness.
 * @param normalized         A flag indicating whether or not to normalize the result
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 *
 * @return device vector containing the centralities.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> edge_betweenness_centrality(
  const raft::handle_t& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<std::variant<vertex_t, raft::device_span<vertex_t const>>> vertices,
  bool normalized         = true,
  bool do_expensive_check = false);

enum class cugraph_cc_t {
  CUGRAPH_WEAK = 0,  ///> Weakly Connected Components
  CUGRAPH_STRONG,    ///> Strongly Connected Components
  NUM_CONNECTIVITY_TYPES
};

/**
 * @brief      Compute connected components.
 *
 * The weak version (for undirected graphs, only) was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 * The strong version (for directed or undirected graphs) is based on:
 * [2] Gilbert, J. et al, 2011. "Graph Algorithms in the Language of Linear Algebra"
 *
 * C = I | A | A^2 |...| A^k
 * where matrix multiplication is via semi-ring:
 * (combine, reduce) == (&, |) (bitwise ops)
 * Then: X = C & transpose(C); and finally, apply get_labels(X);
 *
 * @throws                cugraph::logic_error when an error occurs.
 *
 * @tparam VT                     Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                     Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                     Type of edge weights. Supported values : float or double.
 *
 * @param[in] graph               cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 * @param[in] connectivity_type   STRONG or WEAK
 * @param[out] labels             Device array of component labels (labels[i] indicates the label
 * associated with vertex id i.
 */
template <typename VT, typename ET, typename WT>
void connected_components(legacy::GraphCSRView<VT, ET, WT> const& graph,
                          cugraph_cc_t connectivity_type,
                          VT* labels);

/**
 * @brief     Compute k truss for a graph
 *
 * K Truss is the maximal subgraph of a graph which contains at least three
 * vertices where every edge is incident to at least k-2 triangles.
 *
 * Note that current implementation does not support a weighted graph.
 *
 * @throws                           cugraph::logic_error with a custom message when an error
 * occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a COO
 * @param[in] k                      The order of the truss
 * @param[in] mr                     Memory resource used to allocate the returned graph
 * @return                           Unique pointer to K Truss subgraph in COO format
 *
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> k_truss_subgraph(
  legacy::GraphCOOView<VT, ET, WT> const& graph,
  int k,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief         Compute the Core Number for the nodes of the graph G
 *
 * @param[in]  graph                cuGraph graph descriptor with a valid edgeList or adjList
 * @param[out] core_number          Populated by the core number of every vertex in the graph
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
template <typename VT, typename ET, typename WT>
void core_number(legacy::GraphCSRView<VT, ET, WT> const& graph, VT* core_number);

/**
 * @brief   Compute K Core of the graph G
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 cuGraph graph in coordinate format
 * @param[in]  k                     Order of the core. This value must not be negative.
 * @param[in]  vertex_id             User specified vertex identifiers for which core number values
 * are supplied
 * @param[in]  core_number           User supplied core number values corresponding to vertex_id
 * @param[in]  num_vertex_ids        Number of elements in vertex_id/core_number arrays
 * @param[in]  mr                    Memory resource used to allocate the returned graph
 *
 * @param[out] out_graph             Unique pointer to K Core subgraph in COO format
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> k_core(
  legacy::GraphCOOView<VT, ET, WT> const& graph,
  int k,
  VT const* vertex_id,
  VT const* core_number,
  VT num_vertex_ids,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * Find pairs of vertices in the input graph such that each pair is connected by
 * a path that is two hops in length.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph        The input graph object
 * @return                  Graph in COO format
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> get_two_hop_neighbors(
  legacy::GraphCSRView<VT, ET, WT> const& graph);

// FIXME: Internally distances is of int (signed 32-bit) data type, but current
// template uses data from VT, ET, WT from the legacy::GraphCSR View even if weights
// are not considered
/**
 * @Synopsis   Performs a breadth first search traversal of a graph starting from a vertex.
 *
 * @throws     cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : int (signed, 32-bit)
 *
 * @param[in] handle                 Library handle (RAFT). If a communicator is set in the handle,
 the multi GPU version will be selected.
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 *
 * @param[out] distances             If set to a valid pointer, this is populated by distance of
 * every vertex in the graph from the starting vertex
 *
 * @param[out] predecessors          If set to a valid pointer, this is populated by bfs traversal
 * predecessor of every vertex
 *
 * @param[out] sp_counters           If set to a valid pointer, this is populated by bfs traversal
 * shortest_path counter of every vertex
 *
 * @param[in] start_vertex           The starting vertex for breadth first search traversal
 *
 * @param[in] directed               Treat the input graph as directed
 *
 * @param[in] mg_batch               If set to true use SG BFS path when comms are initialized.
 *
 */
template <typename VT, typename ET, typename WT>
void bfs(raft::handle_t const& handle,
         legacy::GraphCSRView<VT, ET, WT> const& graph,
         VT* distances,
         VT* predecessors,
         double* sp_counters,
         const VT start_vertex,
         bool directed = true,
         bool mg_batch = false);

/**
 * @brief      Compute Hungarian algorithm on a weighted bipartite graph
 *
 * The Hungarian algorithm computes an assigment of "jobs" to "workers".  This function accepts
 * a weighted graph and a vertex list identifying the "workers".  The weights in the weighted
 * graph identify the cost of assigning a particular job to a worker.  The algorithm computes
 * a minimum cost assignment and returns the cost as well as a vector identifying the assignment.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam edge_t                    Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the
 * handle,
 * @param[in]  graph                 cuGRAPH COO graph
 * @param[in]  num_workers           number of vertices in the worker set
 * @param[in]  workers               device pointer to an array of worker vertex ids
 * @param[out] assignments           device pointer to an array to which the assignment will be
 * written. The array should be num_workers long, and will identify which vertex id (job) is
 * assigned to that worker
 */
template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t num_workers,
                   vertex_t const* workers,
                   vertex_t* assignments);

/**
 * @brief      Compute Hungarian algorithm on a weighted bipartite graph
 *
 * The Hungarian algorithm computes an assigment of "jobs" to "workers".  This function accepts
 * a weighted graph and a vertex list identifying the "workers".  The weights in the weighted
 * graph identify the cost of assigning a particular job to a worker.  The algorithm computes
 * a minimum cost assignment and returns the cost as well as a vector identifying the assignment.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam edge_t                    Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph                 cuGRAPH COO graph
 * @param[in]  num_workers           number of vertices in the worker set
 * @param[in]  workers               device pointer to an array of worker vertex ids
 * @param[out] assignments           device pointer to an array to which the assignment will be
 * written. The array should be num_workers long, and will identify which vertex id (job) is
 * assigned to that worker
 * @param[in]  epsilon               parameter to define precision of comparisons
 *                                   in reducing weights to zero.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   legacy::GraphCOOView<vertex_t, edge_t, weight_t> const& graph,
                   vertex_t num_workers,
                   vertex_t const* workers,
                   vertex_t* assignments,
                   weight_t epsilon);

/**
 * @brief      Louvain implementation
 *
 * Compute a clustering of the graph by maximizing modularity
 *
 * Computed using the Louvain method described in:
 *
 *    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
 *    community hierarchies in large networks, J Stat Mech P10008 (2008),
 *    http://arxiv.org/abs/0803.0476
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam     graph_view_t          Type of graph
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph                 input graph object
 * @param[out] clustering            Pointer to device array where the clustering should be stored
 * @param[in]  max_level             (optional) maximum number of levels to run (default 100)
 * @param[in]  resolution            (optional) The value of the resolution parameter to use.
 *                                   Called gamma in the modularity formula, this changes the size
 *                                   of the communities.  Higher resolutions lead to more smaller
 *                                   communities, lower resolutions lead to fewer larger
 *                                   communities. (default 1)
 *
 * @return                           a pair containing:
 *                                     1) number of levels of the returned clustering
 *                                     2) modularity of the returned clustering
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<size_t, weight_t> louvain(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  vertex_t* clustering,
  size_t max_level    = 100,
  weight_t resolution = weight_t{1});

template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<size_t, weight_t> louvain(
  raft::handle_t const& handle,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph_view,
  vertex_t* clustering,
  size_t max_level    = 100,
  weight_t resolution = weight_t{1});

/**
 * @brief      Louvain implementation, returning dendrogram
 *
 * Compute a clustering of the graph by maximizing modularity
 *
 * Computed using the Louvain method described in:
 *
 *    VD Blondel, J-L Guillaume, R Lambiotte and E Lefebvre: Fast unfolding of
 *    community hierarchies in large networks, J Stat Mech P10008 (2008),
 *    http://arxiv.org/abs/0803.0476
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam     graph_view_t          Type of graph
 *
 * @param[in]  handle                Library handle (RAFT)
 * @param[in]  graph_view            Input graph view object
 * @param[in]  max_level             (optional) maximum number of levels to run (default 100)
 * @param[in]  resolution            (optional) The value of the resolution parameter to use.
 *                                   Called gamma in the modularity formula, this changes the size
 *                                   of the communities.  Higher resolutions lead to more smaller
 *                                   communities, lower resolutions lead to fewer larger
 *                                   communities. (default 1)
 *
 * @return                           a pair containing:
 *                                     1) unique pointer to dendrogram
 *                                     2) modularity of the returned clustering
 *
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::pair<std::unique_ptr<Dendrogram<vertex_t>>, weight_t> louvain(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  size_t max_level    = 100,
  weight_t resolution = weight_t{1});

/**
 * @brief      Flatten a Dendrogram at a particular level
 *
 * A Dendrogram represents a hierarchical clustering/partitioning of
 * a graph.  This function will flatten the hierarchical clustering into
 * a label for each vertex representing the final cluster/partition to
 * which it is assigned
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam     graph_view_t          Type of graph
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph                 input graph object
 * @param[in]  dendrogram            input dendrogram object
 * @param[out] clustering            Pointer to device array where the clustering should be stored
 *
 */
template <typename graph_view_t>
void flatten_dendrogram(raft::handle_t const& handle,
                        graph_view_t const& graph_view,
                        Dendrogram<typename graph_view_t::vertex_type> const& dendrogram,
                        typename graph_view_t::vertex_type* clustering);

/**
 * @brief      Leiden implementation
 *
 * Compute a clustering of the graph by maximizing modularity using the Leiden improvements
 * to the Louvain method.
 *
 * Computed using the Leiden method described in:
 *
 *    Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden:
 *    guaranteeing well-connected communities. Scientific reports, 9(1), 5233.
 *    doi: 10.1038/s41598-019-41695-z
 *
 * @throws cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam edge_t                    Type of edge identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph                 input graph object (CSR)
 * @param[out] clustering            Pointer to device array where the clustering should be stored
 * @param[in]  max_iter              (optional) maximum number of iterations to run (default 100)
 * @param[in]  resolution            (optional) The value of the resolution parameter to use.
 *                                   Called gamma in the modularity formula, this changes the size
 *                                   of the communities.  Higher resolutions lead to more smaller
 *                                   communities, lower resolutions lead to fewer larger
 * communities. (default 1)
 *
 * @return                           a pair containing:
 *                                     1) number of levels of the returned clustering
 *                                     2) modularity of the returned clustering
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::pair<size_t, weight_t> leiden(raft::handle_t const& handle,
                                   legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                                   vertex_t* clustering,
                                   size_t max_iter     = 100,
                                   weight_t resolution = weight_t{1});

/**
 * @brief Computes the ecg clustering of the given graph.
 *
 * ECG runs truncated Louvain on an ensemble of permutations of the input graph,
 * then uses the ensemble partitions to determine weights for the input graph.
 * The final result is found by running full Louvain on the input graph using
 * the determined weights. See https://arxiv.org/abs/1809.05578 for further
 * information.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam edge_t                    Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph_coo             input graph object (COO)
 * @param[in]  graph_csr             input graph object (CSR)
 * @param[in]  min_weight            The minimum weight parameter
 * @param[in]  ensemble_size         The ensemble size parameter
 * @param[out] clustering            A device pointer to array where the partitioning should be
 * written
 */
template <typename vertex_t, typename edge_t, typename weight_t>
void ecg(raft::handle_t const& handle,
         legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
         weight_t min_weight,
         vertex_t ensemble_size,
         vertex_t* clustering);

/**
 * @brief Generate edges in a minimum spanning forest of an undirected weighted graph.
 *
 * A minimum spanning tree is a subgraph of the graph (a tree) with the minimum sum of edge weights.
 * A spanning forest is a union of the spanning trees for each connected component of the graph.
 * If the graph is connected it returns the minimum spanning tree.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam edge_t                    Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  graph_csr             input graph object (CSR) expected to be symmetric
 * @param[in]  mr                    Memory resource used to allocate the returned graph
 * @return out_graph             Unique pointer to MSF subgraph in COO format
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<legacy::GraphCOO<vertex_t, edge_t, weight_t>> minimum_spanning_tree(
  raft::handle_t const& handle,
  legacy::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

namespace subgraph {
/**
 * @brief             Extract subgraph by vertices
 *
 * This function will identify all edges that connect pairs of vertices
 * that are both contained in the vertices list and return a COO containing
 * these edges.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (COO)
 * @param[in]  vertices              device pointer to an array of vertex ids
 * @param[in]  num_vertices          number of vertices in the array vertices
 * @param[out] result                a graph in COO format containing the edges in the subgraph
 */
template <typename VT, typename ET, typename WT>
std::unique_ptr<legacy::GraphCOO<VT, ET, WT>> extract_subgraph_vertex(
  legacy::GraphCOOView<VT, ET, WT> const& graph, VT const* vertices, VT num_vertices);
}  // namespace subgraph

/**
 * @brief     Wrapper function for Nvgraph balanced cut clustering
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[in]  num_clusters          The desired number of clusters
 * @param[in]  num_eigen_vects       The number of eigenvectors to use
 * @param[in]  evs_tolerance         The tolerance to use for the eigenvalue solver
 * @param[in]  evs_max_iter          The maximum number of iterations of the eigenvalue solver
 * @param[in]  kmean_tolerance       The tolerance to use for the kmeans solver
 * @param[in]  kmean_max_iter        The maximum number of iteration of the k-means solver
 * @param[out] clustering            Pointer to device memory where the resulting clustering will
 * be stored
 */

namespace ext_raft {
template <typename VT, typename ET, typename WT>
void balancedCutClustering(legacy::GraphCSRView<VT, ET, WT> const& graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT* clustering);

/**
 * @brief      Wrapper function for Nvgraph spectral modularity maximization algorithm
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[in]  num_clusters          The desired number of clusters
 * @param[in]  num_eigen_vects       The number of eigenvectors to use
 * @param[in]  evs_tolerance         The tolerance to use for the eigenvalue solver
 * @param[in]  evs_max_iter          The maximum number of iterations of the eigenvalue solver
 * @param[in]  kmean_tolerance       The tolerance to use for the kmeans solver
 * @param[in]  kmean_max_iter        The maximum number of iteration of the k-means solver
 * @param[out] clustering            Pointer to device memory where the resulting clustering will
 * be stored
 */
template <typename VT, typename ET, typename WT>
void spectralModularityMaximization(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                    VT n_clusters,
                                    VT n_eig_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT* clustering);

/**
 * @brief      Wrapper function for Nvgraph clustering modularity metric
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[in]  n_clusters            Number of clusters in the clustering
 * @param[in]  clustering            Pointer to device array containing the clustering to analyze
 * @param[out] score                 Pointer to a float in which the result will be written
 */
template <typename VT, typename ET, typename WT>
void analyzeClustering_modularity(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                  int n_clusters,
                                  VT const* clustering,
                                  WT* score);

/**
 * @brief      Wrapper function for Nvgraph clustering edge cut metric
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[in]  n_clusters            Number of clusters in the clustering
 * @param[in]  clustering            Pointer to device array containing the clustering to analyze
 * @param[out] score                 Pointer to a float in which the result will be written
 */
template <typename VT, typename ET, typename WT>
void analyzeClustering_edge_cut(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                int n_clusters,
                                VT const* clustering,
                                WT* score);

/**
 * @brief      Wrapper function for Nvgraph clustering ratio cut metric
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[in]  n_clusters            Number of clusters in the clustering
 * @param[in]  clustering            Pointer to device array containing the clustering to analyze
 * @param[out] score                 Pointer to a float in which the result will be written
 */
template <typename VT, typename ET, typename WT>
void analyzeClustering_ratio_cut(legacy::GraphCSRView<VT, ET, WT> const& graph,
                                 int n_clusters,
                                 VT const* clustering,
                                 WT* score);

}  // namespace ext_raft

namespace dense {
/**
 * @brief      Compute Hungarian algorithm on a weighted bipartite graph
 *
 * The Hungarian algorithm computes an assigment of "jobs" to "workers".  This function accepts
 * a weighted graph and a vertex list identifying the "workers".  The weights in the weighted
 * graph identify the cost of assigning a particular job to a worker.  The algorithm computes
 * a minimum cost assignment and returns the cost as well as a vector identifying the assignment.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  costs                 pointer to array of costs, stored in row major order
 * @param[in]  num_rows              number of rows in dense matrix
 * @param[in]  num_cols              number of cols in dense matrix
 * @param[out] assignments           device pointer to an array to which the assignment will be
 *                                   written. The array should be num_cols long, and will identify
 *                                   which vertex id (job) is assigned to that worker
 */
template <typename vertex_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   weight_t const* costs,
                   vertex_t num_rows,
                   vertex_t num_columns,
                   vertex_t* assignments);

/**
 * @brief      Compute Hungarian algorithm on a weighted bipartite graph
 *
 * The Hungarian algorithm computes an assigment of "jobs" to "workers".  This function accepts
 * a weighted graph and a vertex list identifying the "workers".  The weights in the weighted
 * graph identify the cost of assigning a particular job to a worker.  The algorithm computes
 * a minimum cost assignment and returns the cost as well as a vector identifying the assignment.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t                  Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam weight_t                  Type of edge weights. Supported values : float or double.
 *
 * @param[in]  handle                Library handle (RAFT). If a communicator is set in the handle,
 * @param[in]  costs                 pointer to array of costs, stored in row major order
 * @param[in]  num_rows              number of rows in dense matrix
 * @param[in]  num_cols              number of cols in dense matrix
 * @param[out] assignments           device pointer to an array to which the assignment will be
 *                                   written. The array should be num_cols long, and will identify
 *                                   which vertex id (job) is assigned to that worker
 * @param[in]  epsilon               parameter to define precision of comparisons
 *                                   in reducing weights to zero.
 */
template <typename vertex_t, typename weight_t>
weight_t hungarian(raft::handle_t const& handle,
                   weight_t const* costs,
                   vertex_t num_rows,
                   vertex_t num_columns,
                   vertex_t* assignments,
                   weight_t epsilon);

}  // namespace dense

/**
 * @brief Run breadth-first search to find the distances (and predecessors) from the source
 * vertex.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not `nullptr`, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @throws cugraph::logic_error on erroneous input arguments.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param distances Pointer to the output distance array.
 * @param predecessors Pointer to the output predecessor array or `nullptr`.
 * @param sources Source vertices to start breadth-first search (root vertex of the breath-first
 * search tree). If more than one source is passed, there must be a single source per component.
 * In a multi-gpu context the source vertices should be local to this GPU.
 * @param n_sources number of sources (one source per component at most).
 * @param direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
void bfs(raft::handle_t const& handle,
         graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
         vertex_t* distances,
         vertex_t* predecessors,
         vertex_t const* sources,
         size_t n_sources          = 1,
         bool direction_optimizing = false,
         vertex_t depth_limit      = std::numeric_limits<vertex_t>::max(),
         bool do_expensive_check   = false);

/**
 * @brief Extract paths from breadth-first search output
 *
 * This function extracts paths from the BFS output.  BFS outputs distances
 * and predecessors.  The path from a vertex v back to the original source vertex
 * can be extracted by recursively looking up the predecessor vertex until you arrive
 * back at the original source vertex.
 *
 * @throws cugraph::logic_error on erroneous input arguments.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param distances Pointer to the distance array constructed by bfs.
 * @param predecessors Pointer to the predecessor array constructed by bfs.
 * @param destinations Destination vertices, extract path from source to each of these destinations
 * In a multi-gpu context the destination vertex should be local to this GPU.
 * @param n_destinations number of destinations (one source per component at most).
 *
 * @return std::tuple<rmm::device_uvector<vertex_t>, vertex_t> pair containing
 *       the paths as a dense matrix in the vector and the maximum path length.
 *       Unused elements in the paths * will be set to invalid_vertex_id (-1 for a signed
 *       vertex_t, std::numeric_limits<vertex_t>::max() for an unsigned vertex_t type).
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, vertex_t> extract_bfs_paths(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  vertex_t const* distances,
  vertex_t const* predecessors,
  vertex_t const* destinations,
  size_t n_destinations);

/**
 * @brief Run single-source shortest-path to compute the minimum distances (and predecessors) from
 * the source vertex.
 *
 * This function computes the distances (minimum edge weight sums) from the source vertex. If @p
 * predecessors is not `nullptr`, this function calculates the predecessor of each vertex in the
 * shortest-path as well. Graph edge weights should be non-negative.
 *
 * @throws cugraph::logic_error on erroneous input arguments.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @param distances Pointer to the output distance array.
 * @param predecessors Pointer to the output predecessor array or `nullptr`.
 * @param source_vertex Source vertex to start single-source shortest-path.
 * In a multi-gpu context the source vertex should be local to this GPU.
 * @param cutoff Single-source shortest-path terminates if no more vertices are reachable within the
 * distance of @p cutoff. Any vertex farther than @p cutoff will be marked as unreachable.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void sssp(raft::handle_t const& handle,
          graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
          edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
          weight_t* distances,
          vertex_t* predecessors,
          vertex_t source_vertex,
          weight_t cutoff         = std::numeric_limits<weight_t>::max(),
          bool do_expensive_check = false);

/**
 * @brief Compute PageRank scores.
 *
 * This function computes general (if @p personalization_vertices is `nullptr`) or personalized (if
 * @p personalization_vertices is not `nullptr`.) PageRank scores.
 *
 * @throws cugraph::logic_error on erroneous input arguments or if fails to converge before @p
 * max_iterations.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam result_t Type of PageRank scores.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == false, edge weights are assumed to be 1.0.
 * @param precomputed_vertex_out_weight_sums Pointer to an array storing sums of out-going edge
 * weights for the vertices (for re-use) or `std::nullopt`. If `std::nullopt`, these values are
 * freshly computed. Computing these values outside this function reduces the number of memory
 * allocations/deallocations and computing if a user repeatedly computes PageRank scores using the
 * same graph with different personalization vectors.
 * @param personalization_vertices Pointer to an array storing personalization vertex identifiers
 * (compute personalized PageRank) or `std::nullopt` (compute general PageRank).
 * @param personalization_values Pointer to an array storing personalization values for the vertices
 * in the personalization set. Relevant only if @p personalization_vertices is not `std::nullopt`.
 * @param personalization_vector_size Size of the personalization set. If @personalization_vertices
 * is not `std::nullopt`, the sizes of the arrays pointed by @p personalization_vertices and @p
 * personalization_values should be @p personalization_vector_size.
 * @param pageranks Pointer to the output PageRank score array.
 * @param alpha PageRank damping factor.
 * @param epsilon Error tolerance to check convergence. Convergence is assumed if the sum of the
 * differences in PageRank values between two consecutive iterations is less than the number of
 * vertices in the graph multiplied by @p epsilon.
 * @param max_iterations Maximum number of PageRank iterations.
 * @param has_initial_guess If set to `true`, values in the PageRank output array (pointed by @p
 * pageranks) is used as initial PageRank values. If false, initial PageRank values are set to 1.0
 * divided by the number of vertices in the graph.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void pagerank(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
              std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
              std::optional<weight_t const*> precomputed_vertex_out_weight_sums,
              std::optional<vertex_t const*> personalization_vertices,
              std::optional<result_t const*> personalization_values,
              std::optional<vertex_t> personalization_vector_size,
              result_t* pageranks,
              result_t alpha,
              result_t epsilon,
              size_t max_iterations   = 500,
              bool has_initial_guess  = false,
              bool do_expensive_check = false);

/**
 * @brief Compute Eigenvector Centrality scores.
 *
 * This function computes eigenvector centrality scores using the power method.
 *
 * @throws cugraph::logic_error on erroneous input arguments or if fails to converge before @p
 * max_iterations.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == false, edge weights are assumed to be 1.0.
 * @param initial_centralities Optional device span containing initial values for the eigenvector
 * centralities
 * @param epsilon Error tolerance to check convergence. Convergence is assumed if the sum of the
 * differences in eigenvector centrality values between two consecutive iterations is less than the
 * number of vertices in the graph multiplied by @p epsilon.
 * @param max_iterations Maximum number of power iterations.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return device vector containing the centralities.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<weight_t const>> initial_centralities,
  weight_t epsilon,
  size_t max_iterations   = 500,
  bool do_expensive_check = false);

/**
 * @brief Compute HITS scores.
 *
 * This function computes HITS scores for the vertices of a graph
 *
 * @throws cugraph::logic_error on erroneous input arguments
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param hubs Pointer to the input/output hub score array.
 * @param authorities Pointer to the output authorities score array.
 * @param epsilon Error tolerance to check convergence. Convergence is assumed if the sum of the
 * differences in hub values between two consecutive iterations is less than @p epsilon
 * @param max_iterations Maximum number of HITS iterations.
 * @param has_initial_guess If set to `true`, values in the hubs output array (pointed by @p
 * hubs) is used as initial hub values. If false, initial hub values are set to 1.0
 * divided by the number of vertices in the graph.
 * @param normalize If set to `true`, final hub and authority scores are normalized (the L1-norm of
 * the returned hub and authority score arrays is 1.0) before returning.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple<result_t, size_t> A tuple of sum of the differences of hub scores of the last
 * two iterations and the total number of iterations taken to reach the final result
 */
template <typename vertex_t, typename edge_t, typename result_t, bool multi_gpu>
std::tuple<result_t, size_t> hits(raft::handle_t const& handle,
                                  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                                  result_t* hubs,
                                  result_t* authorities,
                                  result_t epsilon,
                                  size_t max_iterations,
                                  bool has_initial_hubs_guess,
                                  bool normalize,
                                  bool do_expensive_check);

/**
 * @brief Compute Katz Centrality scores.
 *
 * This function computes Katz Centrality scores.
 *
 * @throws cugraph::logic_error on erroneous input arguments or if fails to converge before @p
 * max_iterations.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam result_t Type of Katz Centrality scores.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == false, edge weights are assumed to be 1.0.
 * @param betas Pointer to an array holding the values to be added to each vertex's new Katz
 * Centrality score in every iteration or `nullptr`. If set to `nullptr`, constant @p beta is used
 * instead.
 * @param katz_centralities Pointer to the output Katz Centrality score array.
 * @param alpha Katz Centrality attenuation factor. This should be smaller than the inverse of the
 * maximum eigenvalue of the adjacency matrix of @p graph.
 * @param beta Constant value to be added to each vertex's new Katz Centrality score in every
 * iteration. Relevant only when @p betas is `nullptr`.
 * @param epsilon Error tolerance to check convergence. Convergence is assumed if the sum of the
 * differences in Katz Centrality values between two consecutive iterations is less than the number
 * of vertices in the graph multiplied by @p epsilon.
 * @param max_iterations Maximum number of Katz Centrality iterations.
 * @param has_initial_guess If set to `true`, values in the Katz Centrality output array (pointed by
 * @p katz_centralities) is used as initial Katz Centrality values. If false, zeros are used as
 * initial Katz Centrality values.
 * @param normalize If set to `true`, final Katz Centrality scores are normalized (the L2-norm of
 * the returned Katz Centrality score array is 1.0) before returning.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void katz_centrality(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     result_t const* betas,
                     result_t* katz_centralities,
                     result_t alpha,
                     result_t beta,
                     result_t epsilon,
                     size_t max_iterations   = 500,
                     bool has_initial_guess  = false,
                     bool normalize          = false,
                     bool do_expensive_check = false);

/**
 * @brief returns induced EgoNet subgraph(s) of neighbors centered at nodes in source_vertex within
 * a given radius.
 *
 * @deprecated This algorithm will be deprecated to replaced by the new version
 *             that uses the raft::device_span.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms. Must have at least one worker stream.
 * @param graph_view Graph view object of, we extract induced egonet subgraphs from @p graph_view.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param source_vertex Pointer to egonet center vertices (size == @p n_subgraphs).
 * @param n_subgraphs Number of induced EgoNet subgraphs to extract (ie. number of elements in @p
 * source_vertex).
 * @param radius  Include all neighbors of distance <= radius from @p source_vertex.
 * @return Quadraplet of edge source vertices, edge destination vertices, edge weights (if @p
 * edge_weight_view.has_value() == true), and edge offsets for each induced EgoNet subgraph.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            vertex_t* source_vertex,
            vertex_t n_subgraphs,
            vertex_t radius);

/**
 * @brief returns induced EgoNet subgraph(s) of neighbors centered at nodes in source_vertex within
 * a given radius.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms. Must have at least one worker stream.
 * @param graph_view Graph view object of, we extract induced egonet subgraphs from @p graph_view.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param source_vertex Pointer to egonet center vertices (size == @p n_subgraphs).
 * @param n_subgraphs Number of induced EgoNet subgraphs to extract (ie. number of elements in @p
 * source_vertex).
 * @param radius  Include all neighbors of distance <= radius from @p source_vertex.
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>,
 * rmm::device_uvector<weight_t>, rmm::device_uvector<size_t>> Quadraplet of edge source vertices,
 * edge destination vertices, edge weights, and edge offsets for each induced EgoNet subgraph.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<size_t>>
extract_ego(raft::handle_t const& handle,
            graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
            std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
            raft::device_span<vertex_t const> source_vertices,
            vertex_t radius,
            bool do_expensive_check = false);

/**
 * @brief returns random walks (RW) from starting sources, where each path is of given maximum
 * length. Uniform distribution is assumed for the random engine.
 *
 * @deprecated This algorithm will be deprecated once all of the functionality is migrated
 *             to the newer APIS: uniform_random_walks(), biased_random_walks(), and
 *             node2vec_random_walks().
 *
 * @tparam graph_t Type of graph/view (typically, graph_view_t).
 * @tparam index_t Type used to store indexing and sizes.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph Graph (view )object to generate RW on.
 * @param ptr_d_start Device pointer to set of starting vertex indices for the RW.
 * @param num_paths = number(paths).
 * @param max_depth maximum length of RWs.
 * @param use_padding (optional) specifies if return uses padded format (true), or coalesced
 * (compressed) format; when padding is used the output is a matrix of vertex paths and a matrix of
 * edges paths (weights); in this case the matrices are stored in row major order; the vertex path
 * matrix is padded with `num_vertices` values and the weight matrix is padded with `0` values;
 * @param sampling_strategy pointer for sampling strategy: uniform, biased, etc.; possible
 * values{0==uniform, 1==biased, 2==node2vec}; defaults to nullptr == uniform;
 * @return std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>,
 * rmm::device_uvector<index_t>> Triplet of either padded or coalesced RW paths; in the coalesced
 * case (default), the return consists of corresponding vertex and edge weights for each, and
 * corresponding path sizes. This is meant to minimize the number of DF's to be passed to the Python
 * layer. The meaning of "coalesced" here is that a 2D array of paths of different sizes is
 * represented as a 1D contiguous array. In the padded case the return is a matrix of num_paths x
 * max_depth vertex paths; and num_paths x (max_depth-1) edge (weight) paths, with an empty array of
 * sizes. Note: if the graph is un-weighted the edge (weight) paths consists of `weight_t{1}`
 * entries;
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename index_t, bool multi_gpu>
std::
  tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<weight_t>, rmm::device_uvector<index_t>>
  random_walks(raft::handle_t const& handle,
               graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
               std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
               vertex_t const* ptr_d_start,
               index_t num_paths,
               index_t max_depth,
               bool use_padding                                     = false,
               std::unique_ptr<sampling_params_t> sampling_strategy = nullptr);

/**
 * @brief returns uniform random walks from starting sources, where each path is of given
 * maximum length.
 *
 * @p start_vertices can contain duplicates, in which case different random walks will
 * be generated for each instance.
 *
 * If @p edge_weight_view.has_value() is true, the return contains edge weights.  If the graph is @p
 * edge_weight_view.has_value() is false, the returned value will be std::nullopt.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view graph view to operate on
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param start_vertices Device span defining the starting vertices
 * @param max_length maximum length of random walk
 * @param seed (optional, defaults to system time), seed for random number generation
 * @return tuple containing device vectors of vertices and the edge weights (if
 *         @p edge_weight_view.has_value() is true)<br>
 *         For each input selector there will be (max_length+1) elements in the
 *         vertex vector with the starting vertex followed by the subsequent
 *         vertices in the random walk.  If a path terminates before max_length,
 *         the vertices will be populated with invalid_vertex_id
 *         (-1 for signed vertex_t, std::numeric_limits<vertex_t>::max() for an
 *         unsigned vertex_t type)<br>
 *         For each input selector there will be max_length elements in the weights
 *         vector with the edge weight for the edge in the path.  If a path
 *         terminates before max_length the subsequent edge weights will be
 *         set to weight_t{0}.
 */
// FIXME: Do I care about transposed or not?  I want to be able to operate in either
// direction.
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
uniform_random_walks(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length,
                     uint64_t seed = std::numeric_limits<uint64_t>::max());

/**
 * @brief returns biased random walks from starting sources, where each path is of given
 * maximum length.
 *
 * The next vertex is biased based on the edge weights.  The probability of traversing a
 * departing edge will be the edge weight divided by the sum of the departing edge weights.
 *
 * @p start_vertices can contain duplicates, in which case different random walks will
 * be generated for each instance.
 *
 * @throws                 cugraph::logic_error if the graph is unweighted
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view graph view to operate on
 * @param edge_weight_view View object holding edge weights for @p graph_view.
 * @param start_vertices Device span defining the starting vertices
 * @param max_length maximum length of random walk
 * @param seed (optional, defaults to system time), seed for random number generation
 * @return tuple containing device vectors of vertices and the edge weights<br>
 *         For each input selector there will be (max_length+1) elements in the
 *         vertex vector with the starting vertex followed by the subsequent
 *         vertices in the random walk.  If a path terminates before max_length,
 *         the vertices will be populated with invalid_vertex_id
 *         (-1 for signed vertex_t, std::numeric_limits<vertex_t>::max() for an
 *         unsigned vertex_t type)<br>
 *         For each input selector there will be max_length elements in the weights
 *         vector with the edge weight for the edge in the path.  If a path
 *         terminates before max_length the subsequent edge weights will be
 *         set to weight_t{0}.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length,
                    uint64_t seed = std::numeric_limits<uint64_t>::max());

/**
 * @brief returns biased random walks with node2vec biases from starting sources,
 * where each path is of given maximum length.
 *
 * @p start_vertices can contain duplicates, in which case different random walks will
 * be generated for each instance.
 *
 * If the @p edge_weight_view.has_value() = true, the return contains edge weights and the node2vec
 * computation will utilize the edge weights.  If @p edge_weight_view.has_value() == false, then the
 * return will not contain edge weights and the node2vec computation will assume an edge weight of 1
 * for all edges.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view graph view to operate on
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == false, edge weights are assumed to be 1.0.
 * @param start_vertices Device span defining the starting vertices
 * @param max_length maximum length of random walk
 * @param p node2vec return parameter
 * @param q node2vec in-out parameter
 * @param seed (optional, defaults to system time), seed for random number generation
 * @return tuple containing device vectors of vertices and the edge weights<br>
 *         For each input selector there will be (max_length+1) elements in the
 *         vertex vector with the starting vertex followed by the subsequent
 *         vertices in the random walk.  If a path terminates before max_length,
 *         the vertices will be populated with invalid_vertex_id
 *         (-1 for signed vertex_t, std::numeric_limits<vertex_t>::max() for an
 *         unsigned vertex_t type)<br>
 *         For each input selector there will be max_length elements in the weights
 *         vector with the edge weight for the edge in the path.  If a path
 *         terminates before max_length the subsequent edge weights will be
 *         set to weight_t{0}.
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q,
                      uint64_t seed = std::numeric_limits<uint64_t>::max());

#ifndef NO_CUGRAPH_OPS
/**
 * @brief generate sub-sampled graph as an adjacency list (CSR format) given input graph,
 * list of vertices and sample size per vertex. The output graph consists of the given
 * vertices with each vertex having at most `sample_size` neighbors from the original graph
 *
 * @tparam graph_t Type of input graph/view (typically, graph_view_t, non-transposed and
 * single-gpu).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param rng_state The RngState instance holding pseudo-random number generator state.
 * @param graph Graph (view )object to sub-sample.
 * @param ptr_d_start Device pointer to set of starting vertex indices for the sub-sampling.
 * @param num_start_vertices = number(vertices) to use for sub-sampling.
 * @param sampling_size = max number of neighbors per output vertex.
 * @param sampling_algo = the sampling algorithm (algo R/algo L/etc.) used to produce outputs.
 * @return std::tuple<rmm::device_uvector<typename graph_t::edge_type>,
 *                    rmm::device_uvector<typename graph_t::vertex_type>>
 * Tuple consisting of two arrays representing the offsets and indices of
 * the sub-sampled graph.
 */
template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>>
sample_neighbors_adjacency_list(raft::handle_t const& handle,
                                raft::random::RngState& rng_state,
                                graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
                                vertex_t const* ptr_d_start,
                                size_t num_start_vertices,
                                size_t sampling_size,
                                ops::gnn::graph::SamplingAlgoT sampling_algo);

/**
 * @brief generate sub-sampled graph as an edge list (COO format) given input graph,
 * list of vertices and sample size per vertex. The output graph consists of the given
 * vertices with each vertex having at most `sample_size` neighbors from the original graph
 *
 * @tparam graph_t Type of input graph/view (typically, graph_view_t, non-transposed and
 * single-gpu).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param rng_state The RngState instance holding pseudo-random number generator state.
 * @param graph Graph (view )object to sub-sample.
 * @param ptr_d_start Device pointer to set of starting vertex indices for the sub-sampling.
 * @param num_start_vertices = number(vertices) to use for sub-sampling.
 * @param sampling_size = max number of neighbors per output vertex.
 * @param sampling_algo = the sampling algorithm (algo R/algo L/etc.) used to produce outputs.
 * @return std::tuple<rmm::device_uvector<typename graph_t::edge_type>,
 *                    rmm::device_uvector<typename graph_t::vertex_type>>
 * Tuple consisting of two arrays representing the source and destination nodes of
 * the sub-sampled graph.
 */
template <typename vertex_t, typename edge_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>> sample_neighbors_edgelist(
  raft::handle_t const& handle,
  raft::random::RngState& rng_state,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  vertex_t const* ptr_d_start,
  size_t num_start_vertices,
  size_t sampling_size,
  ops::gnn::graph::SamplingAlgoT sampling_algo);
#endif

/**
 * @brief Finds (weakly-connected-)component IDs of each vertices in the input graph.
 *
 * The input graph must be symmetric. Component IDs can be arbitrary integers (they can be
 * non-consecutive and are not ordered by component size or any other criterion).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param components Pointer to the output component ID array.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
void weakly_connected_components(raft::handle_t const& handle,
                                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                                 vertex_t* components,
                                 bool do_expensive_check = false);

/**
 * @brief  Identify whether the core number computation should be based off incoming edges,
 *         outgoing edges or both.
 */
enum class k_core_degree_type_t { IN = 0, OUT = 1, INOUT = 2 };

/**
 * @brief   Compute core numbers of individual vertices from K-core decomposition.
 *
 * The input graph should not have self-loops nor multi-edges. Currently, only undirected graphs are
 * supported.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * or multi-GPU (true).
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param core_numbers Pointer to the output core number array.
 * @param degree_type Dictate whether to compute the K-core decomposition based on in-degrees,
 * out-degrees, or in-degrees + out_degrees.
 * @param k_first Find K-cores from K = k_first. Any vertices that do not belong to k_first-core
 * will have core numbers of 0.
 * @param k_last Find K-cores to K = k_last. Any vertices that belong to (k_last)-core will have
 * their core numbers set to their degrees on k_last-core.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
void core_number(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                 edge_t* core_numbers,
                 k_core_degree_type_t degree_type,
                 size_t k_first          = 0,
                 size_t k_last           = std::numeric_limits<size_t>::max(),
                 bool do_expensive_check = false);

/**
 * @brief   Extract K Core of a graph
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param  graph_view      Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param  k               Order of the core. This value must not be negative.
 * @param degree_type Optional parameter to dictate whether to compute the K-core decomposition
 *                    based on in-degrees, out-degrees, or in-degrees + out_degrees.  One of @p
 *                    degree_type and @p core_numbers must be specified.
 * @param  core_numbers    Optional output from core_number algorithm.  If not specified then
 *                         k_core will call core_number itself using @p degree_type
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 *
 * @return edge list for the graph
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_core(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       size_t k,
       std::optional<k_core_degree_type_t> degree_type,
       std::optional<raft::device_span<edge_t const>> core_numbers,
       bool do_expensive_check = false);

/**
 * @brief Uniform Neighborhood Sampling.
 *
 * This function traverses from a set of starting vertices, traversing outgoing edges and
 * randomly selects from these outgoing neighbors to extract a subgraph.
 *
 * Output from this function a set of tuples (src, dst, weight, count), identifying the randomly
 * selected edges.  src is the source vertex, dst is the destination vertex, weight is the weight
 * of the edge and count identifies the number of times this edge was encountered during the
 * sampling of this graph (so it is >= 1).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph View object to generate NBR Sampling on.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view.
 * @param starting_vertices Device span of starting vertex IDs for the NBR Sampling.
 * @param fan_out Host span defining branching out (fan-out) degree per source vertex for each
 * level
 * @param with_replacement boolean flag specifying if random sampling is done with replacement
 * (true); or, without replacement (false); default = true;
 * @param seed A seed to initialize the random number generator
 * @return tuple device vectors (vertex_t source_vertex, vertex_t destination_vertex, weight_t
 * weight, edge_t count)
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
                   std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                   raft::device_span<vertex_t> starting_vertices,
                   raft::host_span<const int> fan_out,
                   bool with_replacement = true,
                   uint64_t seed         = 0);

/*
 * @brief Compute triangle counts.
 *
 * Compute triangle counts for the entire set of vertices (if @p vertices is std::nullopt) or the
 * given vertices (@p vertices.has_value() is true).
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param vertices Vertices to compute triangle counts. If @p vertices.has_value() is false, compute
 * triangle counts for the entire set of vertices.
 * @param counts Output triangle count array. The size of the array should be the local vertex
 * partition range size (if @p vertices is std::nullopt) or the size of @p vertices (if @p
 * vertices.has_value() is true).
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
void triangle_count(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    std::optional<raft::device_span<vertex_t const>> vertices,
                    raft::device_span<edge_t> counts,
                    bool do_expensive_check = false);

/**
 * @brief     Compute Jaccard similarity coefficient
 *
 * Similarity is computed for every pair of vertices specified. Note that
 * similarity algorithms expect a symmetric graph.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == true, use the weights associated with the graph. If false, assume
 * a weight of 1 for all edges.
 * @param vertex_pairs tuple of device spans defining the vertex pairs to compute similarity for
 * In a multi-gpu context each vertex pair should be local to this GPU.
 * @return similarity coefficient for the corresponding @p vertex_pairs
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> jaccard_coefficients(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs);

/**
 * @brief     Compute Sorensen similarity coefficient
 *
 * Similarity is computed for every pair of vertices specified. Note that
 * similarity algorithms expect a symmetric graph.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == true, use the weights associated with the graph. If false, assume
 * a weight of 1 for all edges.
 * @param vertex_pairs tuple of device spans defining the vertex pairs to compute similarity for
 * @param vertex_pairs tuple of device spans defining the vertex pairs to compute similarity for
 * In a multi-gpu context each vertex pair should be local to this GPU.
 * @return similarity coefficient for the corresponding @p vertex_pairs
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> sorensen_coefficients(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs);

/**
 * @brief     Compute overlap similarity coefficient
 *
 * Similarity is computed for every pair of vertices specified. Note that
 * similarity algorithms expect a symmetric graph.
 *
 * @throws                 cugraph::logic_error when an error occurs.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam weight_t Type of edge weights. Needs to be a floating point type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param edge_weight_view Optional view object holding edge weights for @p graph_view. If @p
 * edge_weight_view.has_value() == true, use the weights associated with the graph. If false, assume
 * a weight of 1 for all edges.
 * @param vertex_pairs tuple of device spans defining the vertex pairs to compute similarity for
 * @param vertex_pairs tuple of device spans defining the vertex pairs to compute similarity for
 * In a multi-gpu context each vertex pair should be local to this GPU.
 * @return similarity coefficient for the corresponding @p vertex_pairs
 */
template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> overlap_coefficients(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs);

/*
 * @brief Enumerate K-hop neighbors
 *
 * Note that the number of K-hop neighbors (and memory footprint) can grow very fast if there are
 * high-degree vertices. Limit the number of start vertices and @p k to avoid rapid increase in
 * memory footprint.
 *
 * @tparam vertex_t Type of vertex identifiers. Needs to be an integral type.
 * @tparam edge_t Type of edge identifiers. Needs to be an integral type.
 * @tparam multi_gpu Flag indicating whether template instantiation should target single-GPU (false)
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Graph view object.
 * @param start_vertices Find K-hop neighbors from each vertex in @p start_vertices.
 * @param k Number of hops to make to enumerate neighbors.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return Tuple of two arrays: offsets and K-hop neighbors. The size of the offset array is @p
 * start_vertices.size() + 1. The i'th and (i+1)'th elements of the offset array demarcates the
 * beginning (inclusive) and end (exclusive) of the K-hop neighbors of the i'th element of @p
 * start_vertices, respectively.
 */
template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<vertex_t>> k_hop_nbrs(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> start_vertices,
  size_t k,
  bool do_expensive_check = false);

}  // namespace cugraph

/**
 * @}
 */
