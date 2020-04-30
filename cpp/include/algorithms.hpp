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
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers. Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported value : float or double.   
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a transposed adjacency list (CSC). Edge weights are not used for this algorithm.
 * @param[in] alpha                  The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85.
                                     Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 *                                   The initial guess must not be the vector of 0s. Any value other than 1 or 0 is treated as an invalid value.
 * @param[in] pagerank               Array of size V. Should contain the initial guess if has_guess=true. In this case the initial guess cannot be the vector of 0s. Memory is provided and owned by the caller.
 * @param[in] personalization_subset_size (optional) The number of vertices for to personalize. Initialized to 0 by default.
 * @param[in] personalization_subset (optional) Array of size personalization_subset_size containing vertices for running personalized pagerank. Initialized to nullptr by default. Memory is provided and owned by the caller.
 * @param[in] personalization_values (optional) Array of size personalization_subset_size containing values associated with personalization_subset vertices. Initialized to nullptr by default. Memory is provided and owned by the caller.
 * @param[in] tolerance              Set the tolerance the approximation, this parameter should be a small magnitude value.
 *                                   The lower the tolerance the better the approximation. If this value is 0.0f, cuGRAPH will use the default value which is 1.0E-5.
 *                                   Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @param[in] max_iter               (optional) The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance.
 *                                   If this value is lower or equal to 0 cuGRAPH will use the default value, which is 500.
 * @param[in] has_guess              (optional) This parameter is used to notify cuGRAPH if it should use a user-provided initial guess. False means the user does not have a guess, in this case cuGRAPH will use a uniform vector set to 1/V.
 *                                   If the value is True, cuGRAPH will read the pagerank parameter and use this as an initial guess.
 * @param[out] *pagerank             The PageRank : pagerank[i] is the PageRank of vertex i. Memory remains provided and owned by the caller.
 *
 */
template <typename VT, typename ET, typename WT>
void pagerank(experimental::GraphCSCView<VT,ET,WT> const &graph,
              WT* pagerank,
              VT personalization_subset_size=0, 
              VT* personalization_subset=nullptr, 
              WT* personalization_values=nullptr,
              double alpha = 0.85,
              double tolerance = 1e-5, 
              int64_t max_iter = 500,
              bool has_guess = false);

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
 * @param[in] weights      device pointer to input vertex weights for weighted Jaccard, may be NULL for
 *                         unweighted Jaccard.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by caller
 */
template <typename VT, typename ET, typename WT>
void jaccard(experimental::GraphCSRView<VT,ET,WT> const &graph,
             WT const *weights,
             WT *result);

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
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by caller
 */
template <typename VT, typename ET, typename WT>
void jaccard_list(experimental::GraphCSRView<VT,ET,WT> const &graph,
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
 * @param[in] weights      device pointer to input vertex weights for weighted overlap, may be NULL for
 *                         unweighted overlap.
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by caller
 */
template <typename VT, typename ET, typename WT>
void overlap(experimental::GraphCSRView<VT,ET,WT> const &graph,
             WT const *weights,
             WT *result);

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
 * @param[in] weights      device pointer to input vertex weights for weighted overlap, may be NULL for
 *                         unweighted overlap.
 * @param[in] num_pairs    The number of vertex ID pairs specified
 * @param[in] first        Device pointer to first vertex ID of each pair
 * @param[in] second       Device pointer to second vertex ID of each pair
 * @param[out] result      Device pointer to result values, memory needs to be pre-allocated by caller
 */
template <typename VT, typename ET, typename WT>
void overlap_list(experimental::GraphCSRView<VT,ET,WT> const &graph,
                  WT const *weights,
                  ET num_pairs,
                  VT const *first,
                  VT const *second,
                  WT *result);

/**
 * @brief     Compute betweenness centrality for a graph
 *
 * Betweenness centrality for a vertex is the sum of the fraction of
 * all pairs shortest paths that pass through the vertex.
 *
 * Note that gunrock (current implementation) does not support a weighted graph.
 * 
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 * @tparam result_t                  Type of computed result.  Supported values :  float
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 * @param[out] result                Device array of centrality scores
 * @param[in] normalized             If true, return normalized scores, if false return unnormalized scores.
 * @param[in] endpoints              If true, include endpoints of paths in score, if false do not
 * @param[in] weight                 If specified, device array of weights for each edge
 * @param[in] k                      If specified, number of vertex samples defined in the vertices array
 * @param[in] vertices               If specified, device array of sampled vertex ids to estimate betweenness centrality.
 *
 */
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSRView<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalized = true,
                            bool endpoints = false,
                            WT const *weight = nullptr,
                            VT k = 0,
                            VT const *vertices = nullptr);

enum class cugraph_cc_t {
  CUGRAPH_WEAK = 0,       ///> Weakly Connected Components
  CUGRAPH_STRONG,         ///> Strongly Connected Components
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
 * @param[in] graph               cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 * @param[in] connectivity_type   STRONG or WEAK
 * @param[out] labels             Device array of component labels (labels[i] indicates the label associated with
 *                                vertex id i.
 */
template <typename VT, typename ET, typename WT>
void connected_components(experimental::GraphCSRView<VT,ET,WT> const &graph,
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
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a COO
 * @param[in] k                      The order of the truss
 * @param[out] output_graph          cuGRAPH graph descriptor with the k-truss subgraph as a COO
 *
 */
template <typename VT, typename ET, typename WT>
void k_truss_subgraph(experimental::GraphCOOView<VT, ET, WT> const &graph,
                      int k,
                      experimental::GraphCOOView<VT, ET, WT> &output_graph);

/**                                                                             
 * @brief        Compute the Katz centrality for the nodes of the graph G
 *                                                                              
 * @throws                           cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 * @tparam result_t                  Type of computed result.  Supported values :  float
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 * @param[out] result                Device array of centrality scores
 * @param[in] alpha                  Attenuation factor with a default value of 0.1. Alpha is set to
 *                                   1/(lambda_max) if it is greater where lambda_max is the maximum degree
 *                                   of the graph.
 * @param[in] max_iter               The maximum number of iterations before an answer is returned. This can
 *                                   be used to limit the execution time and do an early exit before the
 *                                   solver reaches the convergence tolerance.
 *                                   If this value is lower or equal to 0 cuGraph will use the default
 *                                   value, which is 100.
 * @param[in] tol                    Set the tolerance the approximation, this parameter should be a small
 *                                   magnitude value.
 *                                   The lower the tolerance the better the approximation. If this value is
 *                                   0.0f, cuGraph will use the default value which is 1.0E-5.
 *                                   Setting too small a tolerance can lead to non-convergence due to
 *                                   numerical roundoff. Usually values between 0.01 and 0.00001 are
 *                                   acceptable.
 * @param[in] has_guess              Flag to determine whether \p katz_centrality contains an initial guess for katz centrality values
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
 * @param[in]  graph                cuGRAPH graph descriptor with a valid edgeList or adjList
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
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 *                                                                              
 * @param[in]  graph                 cuGRAPH graph in coordinate format
 * @param[in]  k                     Order of the core. This value must not be negative.
 * @param[in]  vertex_id             User specified vertex identifiers for which core number values are supplied
 * @param[in]  core_number           User supplied core number values corresponding to vertex_id
 * @param[in]  num_vertex_ids        Number of elements in vertex_id/core_number arrays
 * @param[out] out_graph             Unique pointer to K Core subgraph in COO formate
 */                                                                             
template <typename VT, typename ET, typename WT>
std::unique_ptr<experimental::GraphCOO<VT, ET, WT>>
k_core(experimental::GraphCOOView<VT, ET, WT> const &graph,
            int k,
            VT const *vertex_id,
            VT const *core_number,
            VT num_vertex_ids);

/**
 * @brief      Find all 2-hop neighbors in the graph
 *
 * Find pairs of vertices in the input graph such that each pair is connected by
 * a path that is two hops in length.
 *
 * @throws     cugraph::logic_error when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.   
 *
 * @param[in]  graph        The input graph object
 * @param[out] first        Upon return will be a device pointer pointing to an array containing
 *                          the first entry of each result pair.
 * @param[out] second       Upon return will be a device pointer pointing to an array containing
 *                          the second entry of each result pair.
 * @return    The number of pairs
 */
template <typename VT, typename ET, typename WT>
ET get_two_hop_neighbors(experimental::GraphCSRView<VT, ET, WT> const &graph,
                         VT **first,
                         VT **second);

/**
 * @Synopsis   Performs a single source shortest path traversal of a graph starting from a vertex.
 *
 * @throws     cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : float or double.
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 *
 * @param[out] distances            If set to a valid pointer, array of size V populated by distance of every vertex in the graph from the starting vertex. Memory is provided and owned by the caller.
 *
 * @param[out] predecessors         If set to a valid pointer, array of size V populated by the SSSP predecessor of every vertex. Memory is provided and owned by the caller.
 *
 * @param[in] start_vertex           The starting vertex for SSSP
 *
 */
template <typename VT, typename ET, typename WT>
void sssp(experimental::GraphCSRView<VT,ET,WT> const &graph,
          WT *distances,
          VT *predecessors,
          const VT source_vertex);

// TODO: Either distances is in VT or in WT, even if there should be no weights
/**
 * @Synopsis   Performs a breadth first search traversal of a graph starting from a vertex.
 *
 * @throws     cugraph::logic_error with a custom message when an error occurs.
 *
 * @tparam VT                        Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET                        Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT                        Type of edge weights. Supported values : int (signed, 32-bit)
 *
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 *
 * @param[out] distances            If set to a valid column, this is populated by distance of every vertex in the graph from the starting vertex
 *
 * @param[out] predecessors         If set to a valid column, this is populated by bfs traversal predecessor of every vertex
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
         const VT start_vertex,
         bool directed = true);
} //namespace cugraph
