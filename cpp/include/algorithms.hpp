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
 * @param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as a transposed adjacency list (CSR). Edge weights are not used for this algorithm.
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
void pagerank(experimental::GraphCSC<VT,ET,WT> const &graph,
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
void jaccard(experimental::GraphCSR<VT,ET,WT> const &graph,
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
void jaccard_list(experimental::GraphCSR<VT,ET,WT> const &graph,
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
void overlap(experimental::GraphCSR<VT,ET,WT> const &graph,
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
void overlap_list(experimental::GraphCSR<VT,ET,WT> const &graph,
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
 * @param[in] implem                 Cugraph currently supports 2 implementations: native and gunrock
 * @param[in] endpoints              If true, include endpoints of paths in score, if false do not
 * @param[in] weight                 If specified, device array of weights for each edge
 * @param[in] k                      If specified, number of vertex samples defined in the vertices array if sample_seed is defined, or  the number of vertices to start traversal from
 * @param[in] vertices               If specified, device array of sampled vertex ids to estimate betweenness centrality.
 *
 */
enum class cugraph_bc_implem_t {
  CUGRAPH_DEFAULT = 0,
  CUGRAPH_GUNROCK
};
template <typename VT, typename ET, typename WT, typename result_t>
void betweenness_centrality(experimental::GraphCSR<VT,ET,WT> const &graph,
                            result_t *result,
                            bool normalized = true,
                            bool endpoints = false,
                            cugraph_bc_implem_t implem = cugraph_bc_implem_t::CUGRAPH_DEFAULT, // TODO(xcadet) That could be somewhere else (After result, or last parameter)
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
 * @tparam VT             Type of vertex identifiers. Supported value : int (signed, 32-bit)
 * @tparam ET             Type of edge identifiers.  Supported value : int (signed, 32-bit)
 * @tparam WT             Type of edge weights. Supported values : float or double.   
 *
 * @param[in] graph       cuGRAPH graph descriptor, should contain the connectivity information as a CSR
 * @param[out] labels     Device array of component labels (labels[i] indicates the label associated with
 *                        vertex id i.
 */
template <typename VT, typename ET, typename WT>
void connected_components(experimental::GraphCSR<VT,ET,WT> const &graph,
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
void k_truss_subgraph(experimental::GraphCOO<VT, ET, WT> const &graph,
                      int k,
                      experimental::GraphCOO<VT, ET, WT> &output_graph);

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
void sssp(experimental::GraphCSR<VT,ET,WT> const &graph,
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
void bfs(experimental::GraphCSR<VT, ET, WT> const &graph,
         VT *distances,
         VT *predecessors,
         VT *sp_counters,
         const VT start_vertex,
         bool directed = true);
} //namespace cugraph
