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

#include "algo_types.h"

/**
 * @Synopsis   Find the PageRank vertex values for a graph. cuGraph computes an approximation of the Pagerank eigenvector using the power method.
 * The number of iterations depends on the properties of the network itself; it increases when the tolerance descreases and/or alpha increases toward the limiting value of 1.
 * The user is free to use default values or to provide inputs for the initial guess, tolerance and maximum number of iterations.
 *
 * @Param[in] graph               cuGRAPH graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm).
 *                                The transposed adjacency list will be computed if not already present.
 * @Param[in] alpha               The damping factor alpha represents the probability to follow an outgoing edge, standard value is 0.85.
 Thus, 1.0-alpha is the probability to “teleport” to a random vertex. Alpha should be greater than 0.0 and strictly lower than 1.0.
 * @Param[in] has_guess           This parameter is used to notify cuGRAPH if it should use a user-provided initial guess. False means the user doesn't have a guess, in this case cuGRAPH will use a uniform vector set to 1/V.
 *                                If the value is True, cuGRAPH will read the pagerank parameter and use this as an initial guess.
 *                                The initial guess must not be the vector of 0s. Any value other than 1 or 0 is treated as an invalid value.
 * @Param[in] pagerank (optional) Initial guess if has_guess=true
 * @Param[in] tolerance           Set the tolerance the approximation, this parameter should be a small magnitude value.
 *                                The lower the tolerance the better the approximation. If this value is 0.0f, cuGRAPH will use the default value which is 1.0E-6.
 *                                Setting too small a tolerance can lead to non-convergence due to numerical roundoff. Usually values between 0.01 and 0.00001 are acceptable.
 * @Param[in] max_iter            The maximum number of iterations before an answer is returned. This can be used to limit the execution time and do an early exit before the solver reaches the convergence tolerance.
 *                                If this value is lower or equal to 0 cuGRAPH will use the default value, which is 500.
 *
 * @Param[out] *pagerank          The PageRank : pagerank[i] is the PageRank of vertex i.
 *
 * @Returns                       GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_pagerank(gdf_graph *graph,
                       gdf_column *pagerank,
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
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_grmat_gen(const char* argv,
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
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_bfs(gdf_graph *graph,
                  gdf_column *distances,
                  gdf_column *predecessors,
                  int start_vertex,
                  bool directed);

/**
 * Computes the Jaccard similarity coefficient for every pair of vertices in the graph
 * which are connected by an edge.
 * @param graph The input graph object
 * @param weights The input vertex weights for weighted Jaccard, may be NULL for
 * unweighted Jaccard.
 * @param result The result values are stored here, memory needs to be pre-allocated
 * @return Error code
 */
gdf_error gdf_jaccard(gdf_graph *graph,
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
 * @return Error code
 */
gdf_error gdf_jaccard_list(gdf_graph *graph,
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
 * @return Error code
 */
gdf_error gdf_overlap(gdf_graph *graph,
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
 * @return Error code
 */
gdf_error gdf_overlap_list(gdf_graph *graph,
                           gdf_column *weights,
                           gdf_column *first,
                           gdf_column *second,
                           gdf_column *result);

gdf_error gdf_louvain(gdf_graph *graph,
                      void *final_modularity,
                      void *num_level,
                      gdf_column *louvain_parts);

/**
 * @brief Compute connected components. 
 * The weak version was imported from cuML.
 * This implementation comes from [1] and solves component labeling problem in
 * parallel on CSR-indexes based upon the vertex degree and adjacency graph.
 *
 * [1] Hawick, K.A et al, 2010. "Parallel graph component labelling with GPUs and CUDA"
 *
 
 * @param graph input graph; assumed undirected for weakly CC [in]
 * @param labels gdf_column for the output labels [out]
 * @param connectivity_type 0=WEAK; 1=STRONG
 */
gdf_error gdf_connected_components(gdf_graph *graph,
                                   gdf_column *labels,
                                   cugraph_connect_t connectivity_type);

/** 
 * Computes the in-degree, out-degree, or the sum of both (determined by x) for the given graph. This is
 * a multi-gpu operation operating on a partitioned graph.
 * @param x 0 for in+out, 1 for in, 2 for out
 * @param part_offsets Contains the start/end of each partitions vertex id range
 * @param off The local partition offsets
 * @param ind The local partition indices
 * @param x_cols The results (located on each GPU)
 * @return Error code
 */
gdf_error gdf_snmg_degree(int x,
                          size_t* part_offsets,
                          gdf_column* off,
                          gdf_column* ind,
                          gdf_column** x_cols);
