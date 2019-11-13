/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include "types.h"

namespace cugraph {

/**
 * @Synopsis   Find the PageRank vertex values for a graph. cuGraph computes an approximation of the Pagerank eigenvector using the power method.
 * The number of iterations depends on the properties of the network itself; it increases when the tolerance descreases and/or alpha increases toward the limiting value of 1.
 * The user is free to use default values or to provide inputs for the initial guess, tolerance and maximum number of iterations.
 *
 * @Param[in] graph                  cuGRAPH graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm).
 *                                   The transposed adjacency list will be computed if not already present.
 * @Param[in] alpha                  The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85.
 Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 * @Param[in] has_guess              This parameter is used to notify cuGRAPH if it should use a user-provided initial guess. False means the user doesn't have a guess, in this case cuGRAPH will use a uniform vector set to 1/V.
 *                                   If the value is True, cuGRAPH will read the pagerank parameter and use this as an initial guess.
 *                                   The initial guess must not be the vector of 0s. Any value other than 1 or 0 is treated as an invalid value.
 * @Param[in] pagerank (optional)    Initial guess if has_guess=true
 * @Param[in] personalization_subset (optional) Vertices for running personalized page rank
 * @Param[in] personalization_values (optional) Values associated with personalization_subset vertices
 * @Param[in] tolerance              Set the tolerance the approximation, this parameter should be a small magnitude value.
 *                                   The lower the tolerance the better the approximation. If this value is 0.0f, cuGRAPH will use the default value which is 1.0E-6.
 *                                   Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @Param[in] max_iter               The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance.
 *                                   If this value is lower or equal to 0 cuGRAPH will use the default value, which is 500.
 *
 * @Param[out] *pagerank             The PageRank : pagerank[i] is the PageRank of vertex i.
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void pagerank(Graph* graph,
              gdf_column *pagerank,
              gdf_column *personalization_subset,
              gdf_column *personalization_values,
              float alpha,
              float tolerance,
              int max_iter,
              bool has_guess);

/**
 * @Synopsis   Creates source, destination and value columns based on the specified R-MAT model
 *
 * @Param[in] *argv                  String that accepts the following arguments
 *                                   rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)
 *                                               Generate R-MAT graph as input
 *                                               --rmat_scale=<vertex-scale>
 *                                               --rmat_nodes=<number-nodes>
 *                                               --rmat_edgefactor=<edge-factor>
 *                                               --rmat_edges=<number-edges>
 *                                               --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>
 *                                               --rmat_self_loops If this option is supplied, then self loops will be retained
 *                                               --rmat_undirected If this option is not mentioned, then the graps will be undirected
 *                                       Optional arguments:
 *                                       [--device=<device_index>] Set GPU(s) for testing (Default: 0).
 *                                       [--quiet]                 No output (unless --json is specified).
 *                                       [--random_seed]           This will enable usage of random seed, else it will use same seed
 *
 * @Param[out] &vertices             Number of vertices in the generated edge list
 *
 * @Param[out] &edges                Number of edges in the generated edge list
 *
 * @Param[out] *src                  Columns containing the sources
 *
 * @Param[out] *dst                  Columns containing the destinations
 *
 * @Param[out] *val                  Columns containing the edge weights
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void grmat_gen(const char* argv,
               size_t &vertices,
               size_t &edges,
               gdf_column* src,
               gdf_column* dest,
               gdf_column* val);

/**
 * @Synopsis   Performs a breadth first search traversal of a graph starting from a vertex.
 *
 * @Param[in] *graph                 cuGRAPH graph descriptor with a valid edgeList or adjList
 *
 * @Param[out] *distances            If set to a valid column, this is populated by distance of every vertex in the graph from the starting vertex
 *
 * @Param[out] *predecessors         If set to a valid column, this is populated by bfs traversal predecessor of every vertex
 *
 * @Param[in] start_vertex           The starting vertex for breadth first search traversal
 *
 * @Param[in] directed               Treat the input graph as directed
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void bfs(Graph* graph,
         gdf_column *distances,
         gdf_column *predecessors,
         int start_vertex,
         bool directed);
/**                                                                             
 * @Synopsis   Performs a single source shortest path traversal of a graph starting from a vertex.
 *                                                                              
 * @Param[in] *graph                 cuGRAPH graph descriptor with a valid edgeList or adjList
 *                                                                              
 * @Param[out] *distances            If set to a valid column, this is populated by distance of every vertex in the graph from the starting vertex
 *                                                                              
 * @Param[out] *predecessors         If set to a valid column, this is populated by the sssp predecessor of every vertex
 *                                                                              
 * @Param[in] start_vertex           The starting vertex for SSSP               
 *                                                                              
 * @throws     cugraph::logic_error when an error occurs.
 */                                                                             
/* ----------------------------------------------------------------------------*/
void sssp(Graph* graph,                                            
          gdf_column *distances,                                                  
          gdf_column *predecessors,                                               
          const int source_vertex);                                               
/**
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 * @param graph The input graph object
 * @param weights The input vertex weights for weighted Jaccard, may be NULL for
 * unweighted Jaccard.
 * @param result The result values are stored here, memory needs to be pre-allocated
 * @throws     cugraph::logic_error when an error occurs.
 */
void jaccard(Graph* graph,
             gdf_column *weights,
             gdf_column *result);

/**
 * Computes the Jaccard similarity coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 * @param graph The input graph object
 * @param weights The input vertex weights for weighted Jaccard, may be NULL for
 * unweighted Jaccard.
 * @param first A column containing the first vertex ID of each pair.
 * @param second A column containing the second vertex ID of each pair.
 * @param result The result values are stored here, memory needs to be pre-allocated.
 * @throws     cugraph::logic_error when an error occurs.
 */
void jaccard_list(Graph* graph,
                  gdf_column *weights,
                  gdf_column *first,
                  gdf_column *second,
                  gdf_column *result);

/**
 * Computes the Overlap Coefficient for every pair of vertices in the graph which are
 * connected by an edge.
 * @param graph The input graph object
 * @param weights The input vertex weights for weighted overlap, may be NULL for
 * unweighted.
 * @param result The result values are stored here, memory needs to be pre-allocated.
 * @throws     cugraph::logic_error when an error occurs.
 */
void overlap(Graph* graph,
             gdf_column *weights,
             gdf_column *result);

/**
 * Computes the overlap coefficient for each pair of specified vertices.
 * Vertices are specified as pairs where pair[n] = (first[n], second[n])
 * @param graph The input graph object.
 * @param weights The input vertex weights for weighted overlap, may be NULL for
 * unweighted.
 * @param first A column containing the first vertex Ids of each pair
 * @param second A column containing the second vertex Ids of each pair
 * @param result The result values are stored here, memory needs to be pre-allocated
 * @throws     cugraph::logic_error when an error occurs.
 */
void overlap_list(Graph* graph,
                  gdf_column *weights,
                  gdf_column *first,
                  gdf_column *second,
                  gdf_column *result);

void louvain(Graph* graph,
             void *final_modularity,
             void *num_level,
             gdf_column *louvain_parts);

/**
 * Computes the in-degree, out-degree, or the sum of both (determined by x) for the given graph. This is
 * a multi-gpu operation operating on a partitioned graph.
 * @param x 0 for in+out, 1 for in, 2 for out
 * @param part_offsets Contains the start/end of each partitions vertex id range
 * @param off The local partition offsets
 * @param ind The local partition indices
 * @param x_cols The results (located on each GPU)
 * @throws     cugraph::logic_error when an error occurs.
 */
void snmg_degree(int x,
                 size_t* part_offsets,
                 gdf_column* off,
                 gdf_column* ind,
                 gdf_column** x_cols);

/**
 * Converts the input edge list (partitioned and loaded onto the GPUs) into a partitioned csr representation.
 * This is a multi-gpu operation operating on partitioned data.
 * @param part_offsets Set to contain the start/end of each partition's vertex ID range. (output)
 * @param comm1 A pointer to void pointer which will be used for inter-thread communication
 * @param cooRow The local partition's initial COO row indices (input)
 * @param cooCol The local partition's initial COO column indices (input)
 * @param cooVal The local partition's initial COO values (input)
 * @param csrOff The local partition's CSR Offsets (output)
 * @param csrInd The local partition's CSR Indices (output)
 * @param csrVal The local partition's CSR Values (output)
 * @throws     cugraph::logic_error when an error occurs.
 */
void snmg_coo2csr(size_t* part_offsets,
                  bool free_input,
                  void** comm1,
                  gdf_column* cooRow,
                  gdf_column* cooCol,
                  gdf_column* cooVal,
                  gdf_column* csrOff,
                  gdf_column* csrInd,
                  gdf_column* csrVal);

typedef enum {CUGRAPH_WEAK = 0, CUGRAPH_STRONG, NUM_CONNECTIVITY_TYPES} cugraph_cc_t;

/**
 * @brief Compute connected components. 
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
 *
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param connectivity_type CUGRAPH_WEAK or CUGRAPH_STRONG [in]
 * @param table of 2 gdf_columns: output labels and vertex indices [out]
 * @throws     cugraph::logic_error when an error occurs.
 */
 void connected_components(Graph* graph,
                           cugraph_cc_t connectivity_type,
                           cudf::table *table);

 /**
Find the PageRank vertex values for a graph. cuGraph computes an approximation of the Pagerank eigenvector using the power method.
 * @param[in] src_col_ptrs      Array of size n_gpu containing pointers to gdf columns. The column src_col_ptrs[i] contains the index of the source for each edge on GPU i. Indices must be in the range [0, V-1], where V is the global number of vertices.
 * @param[in] dest_col_ptrs     Array of size n_gpu containing pointers to gdf columns. The column dest_col_ptrs[i] contains the index of the destination for each edge on GPU i. Indices must be in the range [0, V-1], where V is the global number of vertices.
 * @param[out] pr_col_ptrs      Array of size n_gpu containing pointers to gdf columns. The column pr_col_ptrs[i] contains a copy of the full pagerank result on GPU i.
 * @Param[in] alpha             The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85.
 *                              Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 * @param[in] n_gpus            The number of GPUs. This function will launch n_gpus threads and set devices [0, n_gpu-1]. 
 * @Param[in] n_iter            The number of iterations before an answer is returned. This must be greater than 0. It is recommended to run between 10 and 100 iterations.  
 *                              The number of iterations should vary depending on the properties of the network itself and the desired approximation quality; it should be increased when alpha increases toward the limiting value of 1.

 * @throws     cugraph::logic_error when an error occurs.
 */
void snmg_pagerank (gdf_column **src_col_ptrs, 
                    gdf_column **dest_col_ptrs, 
                    gdf_column *pr_col_ptrs, 
                    const size_t n_gpus, 
                    const float damping_factor, 
                    const int n_iter);
/**                                                                             
 * @Synopsis   Compute the Katz centrality for the nodes of the graph G
 *                                                                              
 * @Param[in] *graph                 cuGRAPH graph descriptor with a valid edgeList or adjList
 *                                                                              
 * @Param[out] *katz_centrality      If set to a valid column, this is populated by the katz centrality of every vertex in the graph
 *                                                                              
 * @Param[in] alpha                  Attenuation factor with a default value of 0.1. Alpha is set to
                                     1/(lambda_max) if it is greater where lambda_max is the maximum degree
                                     of the graph.
 *
 * @Param[in] max_iter               The maximum number of iterations before an answer is returned. This can
                                     be used to limit the execution time and do an early exit before the
                                     solver reaches the convergence tolerance.
                                     If this value is lower or equal to 0 cuGraph will use the default
                                     value, which is 100.
 *                                                                              
 * @Param[in] tol                    Set the tolerance the approximation, this parameter should be a small
                                     magnitude value.
                                     The lower the tolerance the better the approximation. If this value is
                                     0.0f, cuGraph will use the default value which is 1.0E-5.
                                     Setting too small a tolerance can lead to non-convergence due to
                                     numerical roundoff. Usually values between 0.01 and 0.00001 are
                                     acceptable.
 *                                                                              
 * @Param[in] has_guess              Flag to determine whether \p katz_centrality contains an initial guess for katz centrality values
 *                                                                              
 * @Param[in] normalized             If True normalize the resulting katz centrality values
 *                                                                              
 * @throws     cugraph::logic_error when an error occurs.
 */                                                                             
/* ----------------------------------------------------------------------------*/
void katz_centrality(Graph* graph,                                         
                     gdf_column *katz_centrality,
                     double alpha,
                     int max_iter,
                     double tol,
                     bool has_guess,
                     bool normalized);

/**                                                                             
 * @Synopsis   Compute the Core Number for the nodes of the graph G
 *                                                                              
 * @Param[in] *graph                 cuGRAPH graph descriptor with a valid edgeList or adjList
 *                                                                              
 * @Param[out] *core_number          If set to a valid column, this is populated by the core number of every vertex in the graph
 *                                                                              
 * @throws     cugraph::logic_error when an error occurs.
 */                                                                             
/* ----------------------------------------------------------------------------*/
void core_number(Graph* graph,                                         
                 gdf_column *core_number);

/**                                                                             
 * @Synopsis   Compute K Core of the graph G
 *                                                                              
 * @Param[in] *in_graph              cuGRAPH graph descriptor with a valid edgeList or adjList
 *                                                                              
 * @Param[in] k                      Order of the core. This value must not be negative.
 *                                                                              
 * @Param[in] *vertex_id             User specified vertex identifiers for which core number values are supplied
 *                                                                              
 * @Param[in] *core_number           User supplied core number values corresponding to vertex_id
 *                                                                              
 * @Param[out] *out_graph            K Core subgraph
 *                                                                              
 * @throws     cugraph::logic_error when an error occurs.
 */                                                                             
/* ----------------------------------------------------------------------------*/
void k_core(Graph* in_graph,
            int k,
            gdf_column *vertex_id,
            gdf_column *core_number,
            Graph* out_graph);
} //namespace cugraph