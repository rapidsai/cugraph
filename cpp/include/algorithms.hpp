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

#include <graph.hpp>
#include <internals.hpp>

namespace cugraph {

/**
 * @brief     Find the PageRank vertex values for a graph.
 *
 * cuGraph computes an approximation of the Pagerank eigenvector using the power method.
 * The number of iterations depends on the properties of the network itself; it increases
 * when the tolerance descreases and/or alpha increases toward the limiting value of 1.
 * The user is free to use default values or to provide inputs for the initial guess,
 * tolerance and maximum number of iterations.

 *
 * @throws                           cugraph::logic_error with a custom message when an error
 occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 32-bit)
 * @tparam ET                        Type of edge identifiers. Supported value : int (signed,
 32-bit)
 * @tparam WT                        Type of edge weights. Supported value : float or double.
 *
 * @param[in] handle                 cuGraph library handle. If a communicator is set in the handle,
 the multi GPU version will be selected.
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 information as a transposed adjacency list (CSC). Edge weights are not used for this algorithm.
 * @param[in] alpha                  The damping factor alpha represents the probability to follow
 an outgoing edge, standard value is 0.85. Thus, 1.0-alpha is the probability to “teleport” to a
 random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 *                                   The initial guess must not be the vector of 0s. Any value other
 than 1 or 0 is treated as an invalid value.
 * @param[in] pagerank               Array of size V. Should contain the initial guess if
 has_guess=true. In this case the initial guess cannot be the vector of 0s. Memory is provided and
 owned by the caller.
 * @param[in] personalization_subset_size (optional) The number of vertices for to personalize.
 Initialized to 0 by default.
 * @param[in] personalization_subset (optional) Array of size personalization_subset_size containing
 vertices for running personalized pagerank. Initialized to nullptr by default. Memory is provided
 and owned by the caller.
 * @param[in] personalization_values (optional) Array of size personalization_subset_size containing
 values associated with personalization_subset vertices. Initialized to nullptr by default. Memory
 is provided and owned by the caller.
 * @param[in] tolerance              Set the tolerance the approximation, this parameter should be a
 small magnitude value.
 *                                   The lower the tolerance the better the approximation. If this
 value is 0.0f, cuGraph will use the default value which is 1.0E-5.
 *                                   Setting too small a tolerance can lead to non-convergence due
 to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @param[in] max_iter               (optional) The maximum number of iterations before an answer is
 returned. This can be used to limit the execution time and do an early exit before the solver
 reaches the convergence tolerance.
 *                                   If this value is lower or equal to 0 cuGraph will use the
 default value, which is 500.
 * @param[in] has_guess              (optional) This parameter is used to notify cuGraph if it
 should use a user-provided initial guess. False means the user does not have a guess, in this case
 cuGraph will use a uniform vector set to 1/V.
 *                                   If the value is True, cuGraph will read the pagerank parameter
 and use this as an initial guess.
 * @param[out] *pagerank             The PageRank : pagerank[i] is the PageRank of vertex i. Memory
 remains provided and owned by the caller.
 *
 */
template <typename VT, typename ET, typename WT>
void pagerank(raft::handle_t const &handle,
              experimental::GraphCSCView<VT, ET, WT> const &graph,
              WT *pagerank,
              VT personalization_subset_size = 0,
              VT *personalization_subset     = nullptr,
              WT *personalization_values     = nullptr,
              double alpha                   = 0.85,
              double tolerance               = 1e-5,
              int64_t max_iter               = 500,
              bool has_guess                 = false);

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
void jaccard(experimental::GraphCSRView<VT, ET, WT> const &graph, WT const *weights, WT *result);

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
void jaccard_list(experimental::GraphCSRView<VT, ET, WT> const &graph,
                  WT const *weights,
                  ET num_pairs,
                  VT const *first,
                  VT const *second,
                  WT *result);

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
void overlap(experimental::GraphCSRView<VT, ET, WT> const &graph, WT const *weights, WT *result);

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
void overlap_list(experimental::GraphCSRView<VT, ET, WT> const &graph,
                  WT const *weights,
                  ET num_pairs,
                  VT const *first,
                  VT const *second,
                  WT *result);

/**
 *
 * @brief                                       ForceAtlas2 is a continuous graph layout algorithm
 * for handy network visualization.
 *
 *                                              NOTE: Peak memory allocation occurs at 17*V.
 *
 * @throws                                      cugraph::logic_error when an error occurs.
 *
 * @tparam VT                                   Type of vertex identifiers. Supported value : int
 * (signed, 32-bit)
 * @tparam ET                                   Type of edge identifiers.  Supported value : int
 * (signed, 32-bit)
 * @tparam WT                                   Type of edge weights. Supported values : float or
 * double.
 *
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
 * @param[in] barnes_hut_optimize:              Whether to use the fast Barnes Hut or use the slower
 * exact version.
 * @param[in] barnes_hut_theta:                 Float between 0 and 1. Tradeoff for speed (1) vs
 * accuracy (0) for Barnes Hut only.
 * @params[in] scaling_ratio                    Float strictly positive. How much repulsion you
 * want. More makes a more sparse graph. Switching from regular mode to LinLog mode needs a
 * readjustment of the scaling parameter.
 * @params[in] strong_gravity_mode                      The “Strong gravity” option sets a force
 * that attracts the nodes that are distant from the center more ( is this distance). This force has
 * the drawback of being so strong that it is sometimes stronger than the other forces. It may
 * result in a biased placement of the nodes. However, its advantage is to force a very compact
 * layout, which may be useful for certain purposes.
 * @params[in] gravity                          Attracts nodes to the center. Prevents islands from
 * drifting away.
 * @params[in] verbose                          Output convergence info at each interation.
 * @params[in] callback                         An instance of GraphBasedDimRedCallback class to
 * intercept the internal state of positions while they are being trained.
 *
 */
template <typename VT, typename ET, typename WT>
void force_atlas2(experimental::GraphCOOView<VT, ET, WT> &graph,
                  float *pos,
                  const int max_iter                            = 500,
                  float *x_start                                = nullptr,
                  float *y_start                                = nullptr,
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
                  internals::GraphBasedDimRedCallback *callback = nullptr);

/**
 * @brief     Compute betweenness centrality for a graph
 *
 * Betweenness centrality for a vertex is the sum of the fraction of
 * all pairs shortest paths that pass through the vertex.
 *
 * The current implementation does not support a weighted graph.
 *
 * @throws                           cugraph::logic_error if `result == nullptr` or
 * `number_of_sources < 0` or `number_of_sources !=0 and sources == nullptr`.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 * @tparam result_t                  Type of computed result.  Supported values :  float or double
 * (double only supported in default implementation)
 *
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 * @param[out] result                Device array of centrality scores
 * @param[in] normalized             If true, return normalized scores, if false return unnormalized
 * scores.
 * @param[in] endpoints              If true, include endpoints of paths in score, if false do not
 * @param[in] weight                 If specified, device array of weights for each edge
 * @param[in] k                      If specified, number of vertex samples defined in the vertices
 * array.
 * @param[in] vertices               If specified, host array of vertex ids to estimate betweenness
 * centrality, these vertices will serve as sources for the traversal algorihtm to obtain
 * shortest path counters.
 *
 */
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                            result_t *result,
                            bool normalized    = true,
                            bool endpoints     = false,
                            WT const *weight   = nullptr,
                            VT k               = 0,
                            VT const *vertices = nullptr);

/**
 * @brief     Compute edge betweenness centrality for a graph
 *
 * Betweenness centrality of an edge is the sum of the fraction of all-pairs shortest paths that
 * pass through this edge. The weight parameter is currenlty not supported
 *
 *  * @throws                           cugraph::logic_error if `result == nullptr` or
 * `number_of_sources < 0` or `number_of_sources !=0 and sources == nullptr` or `endpoints ==
 true`.

 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 * @tparam result_t                  Type of computed result.  Supported values :  float or double
 * (double only supported in default implementation)
 *
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 * @param[out] result                Device array of centrality scores
 * @param[in] normalized             If true, return normalized scores, if false return unnormalized
 * scores.
 * @param[in] weight                 If specified, device array of weights for each edge
 * @param[in] k                      If specified, number of vertex samples defined in the vertices
 * array.
 * @param[in] vertices               If specified, host array of vertex ids to estimate betweenness
 * centrality, these vertices will serve as sources for the traversal algorihtm to obtain
 * shortest path counters.
 *
 */
template <typename VT, typename ET, typename WT, typename result_t>
void edge_betweenness_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 result_t *result,
                                 bool normalized    = true,
                                 WT const *weight   = nullptr,
                                 VT k               = 0,
                                 VT const *vertices = nullptr);

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
void connected_components(experimental::GraphCSRView<VT, ET, WT> const &graph,
                          cugraph_cc_t connectivity_type,
                          VT *labels);

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
std::unique_ptr<experimental::GraphCOO<VT, ET, WT>> k_truss_subgraph(
  experimental::GraphCOOView<VT, ET, WT> const &graph,
  int k,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief        Compute the Katz centrality for the nodes of the graph G
 *
 * @throws                           cugraph::logic_error with a custom message when an error
 * occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 * @tparam result_t                  Type of computed result.  Supported values :  float
 *
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 * @param[out] result                Device array of centrality scores
 * @param[in] alpha                  Attenuation factor with a default value of 0.1. Alpha is set to
 *                                   1/(lambda_max) if it is greater where lambda_max is the maximum
 * degree of the graph.
 * @param[in] max_iter               The maximum number of iterations before an answer is returned.
 * This can be used to limit the execution time and do an early exit before the solver reaches the
 * convergence tolerance. If this value is lower or equal to 0 cuGraph will use the default value,
 * which is 100.
 * @param[in] tol                    Set the tolerance the approximation, this parameter should be a
 * small magnitude value. The lower the tolerance the better the approximation. If this value is
 *                                   0.0f, cuGraph will use the default value which is 1.0E-5.
 *                                   Setting too small a tolerance can lead to non-convergence due
 * to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @param[in] has_guess              Flag to determine whether \p katz_centrality contains an
 * initial guess for katz centrality values
 * @param[in] normalized             If True normalize the resulting katz centrality values
 */
template <typename VT, typename ET, typename WT, typename result_t>
void katz_centrality(experimental::GraphCSRView<VT, ET, WT> const &graph,
                     result_t *result,
                     double alpha,
                     int max_iter,
                     double tol,
                     bool has_guess,
                     bool normalized);

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
void core_number(experimental::GraphCSRView<VT, ET, WT> const &graph, VT *core_number);

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
std::unique_ptr<experimental::GraphCOO<VT, ET, WT>> k_core(
  experimental::GraphCOOView<VT, ET, WT> const &graph,
  int k,
  VT const *vertex_id,
  VT const *core_number,
  VT num_vertex_ids,
  rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

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
std::unique_ptr<cugraph::experimental::GraphCOO<VT, ET, WT>> get_two_hop_neighbors(
  experimental::GraphCSRView<VT, ET, WT> const &graph);

/**
 * @Synopsis   Performs a single source shortest path traversal of a graph starting from a vertex.
 *
 * @throws     cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in] graph                  cuGraph graph descriptor, should contain the connectivity
 * information as a CSR
 *
 * @param[out] distances            If set to a valid pointer, array of size V populated by distance
 * of every vertex in the graph from the starting vertex. Memory is provided and owned by the
 * caller.
 *
 * @param[out] predecessors         If set to a valid pointer, array of size V populated by the SSSP
 * predecessor of every vertex. Memory is provided and owned by the caller.
 *
 * @param[in] start_vertex           The starting vertex for SSSP
 *
 */
template <typename VT, typename ET, typename WT>
void sssp(experimental::GraphCSRView<VT, ET, WT> const &graph,
          WT *distances,
          VT *predecessors,
          const VT source_vertex);

// FIXME: Internally distances is of int (signed 32-bit) data type, but current
// template uses data from VT, ET, WT from he GraphCSR View even if weights
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
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT, typename ET, typename WT>
void bfs(experimental::GraphCSRView<VT, ET, WT> const &graph,
         VT *distances,
         VT *predecessors,
         double *sp_counters,
         const VT start_vertex,
         bool directed = true);

/**
 * @brief      Louvain implementation
 *
 * Compute a clustering of the graph by minimizing modularity
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph                 input graph object (CSR)
 * @param[out] final_modularity      modularity of the returned clustering
 * @param[out] num_level             number of levels of the returned clustering
 * @param[out] clustering            Pointer to device array where the clustering should be stored
 * @param[in]  max_iter              (optional) maximum number of iterations to run (default 100)
 */
template <typename VT, typename ET, typename WT>
void louvain(experimental::GraphCSRView<VT, ET, WT> const &graph,
             WT *final_modularity,
             int *num_level,
             VT *louvain_parts,
             int max_iter = 100);

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
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed,
 * 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed,
 * 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in]  graph_coo             input graph object (COO)
 * @param[in]  graph_csr             input graph object (CSR)
 * @param[in]  min_weight            The minimum weight parameter
 * @param[in]  ensemble_size         The ensemble size parameter
 * @param[out] ecg_parts             A device pointer to array where the partitioning should be
 * written
 */
template <typename VT, typename ET, typename WT>
void ecg(experimental::GraphCSRView<VT, ET, WT> const &graph_csr,
         WT min_weight,
         VT ensemble_size,
         VT *ecg_parts);

namespace nvgraph {

/**
 * @brief             Count the number of triangles in the graph
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
 *
 * @return                           The number of triangles
 */
template <typename VT, typename ET, typename WT>
uint64_t triangle_count(experimental::GraphCSRView<VT, ET, WT> const &graph);

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
std::unique_ptr<experimental::GraphCOO<VT, ET, WT>> extract_subgraph_vertex(
  experimental::GraphCOOView<VT, ET, WT> const &graph, VT const *vertices, VT num_vertices);

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
 * @param[out] clustering            Pointer to device memory where the resulting clustering will be
 * stored
 */
template <typename VT, typename ET, typename WT>
void balancedCutClustering(experimental::GraphCSRView<VT, ET, WT> const &graph,
                           VT num_clusters,
                           VT num_eigen_vects,
                           WT evs_tolerance,
                           int evs_max_iter,
                           WT kmean_tolerance,
                           int kmean_max_iter,
                           VT *clustering);

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
 * @param[out] clustering            Pointer to device memory where the resulting clustering will be
 * stored
 */
template <typename VT, typename ET, typename WT>
void spectralModularityMaximization(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                    VT n_clusters,
                                    VT n_eig_vects,
                                    WT evs_tolerance,
                                    int evs_max_iter,
                                    WT kmean_tolerance,
                                    int kmean_max_iter,
                                    VT *clustering);

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
void analyzeClustering_modularity(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                  int n_clusters,
                                  VT const *clustering,
                                  WT *score);

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
void analyzeClustering_edge_cut(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                int n_clusters,
                                VT const *clustering,
                                WT *score);

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
void analyzeClustering_ratio_cut(experimental::GraphCSRView<VT, ET, WT> const &graph,
                                 int n_clusters,
                                 VT const *clustering,
                                 WT *score);

}  // namespace nvgraph

namespace gunrock {

/**
 * @brief     Compute the HITS vertex values for a graph
 *
 * cuGraph uses the gunrock implementation of HITS
 *
 * @throws                           cugraph::logic_error on an error
 *
 * @tparam VT                        Type of vertex identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.
 *                                   Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights.
 *                                   Supported value : float
 *
 * @param[in] graph                  input graph object (CSR). Edge weights are not used
 *                                   for this algorithm.
 * @param[in] max_iter               Maximum number of iterations to run
 * @param[in] tolerance              Currently ignored.  gunrock implementation runs
 *                                   the specified number of iterations and stops
 * @param[in] starting value         Currently ignored.  gunrock does not support.
 * @param[in] normalized             Currently ignored, gunrock computes this as true
 * @param[out] *hubs                 Device memory pointing to the node value based
 *                                   on outgoing links
 * @param[out] *authorities          Device memory pointing to the node value based
 *                                   on incoming links
 *
 */
template <typename VT, typename ET, typename WT>
void hits(experimental::GraphCSRView<VT, ET, WT> const &graph,
          int max_iter,
          WT tolerance,
          WT const *starting_value,
          bool normalized,
          WT *hubs,
          WT *authorities);

}  // namespace gunrock

}  // namespace cugraph
