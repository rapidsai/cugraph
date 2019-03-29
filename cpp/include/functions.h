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

/**
 * @Synopsis   Wrap existing gdf columns representing an edge list in a gdf_graph.
 *             cuGRAPH does not own the memory used to represent this graph. This function does not allocate memory.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in] *source_indices        This gdf_column of size E (number of edges) contains the index of the source for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in] *destination_indices   This gdf_column of size E (number of edges) contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in] *edge_data (optional)  This pointer can be nullptr. If not, this gdf_column of size E (number of edges) contains the weiht for each edge.
 *                                   The type expected to be floating point.
 *
 * @Param[out] *graph                cuGRAPH graph descriptor containing the newly added edge list (edge_data is optional).
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_edge_list_view(gdf_graph *graph,
                             const gdf_column *source_indices,
                             const gdf_column *destination_indices,
                             const gdf_column *edge_data);

/**
 * @Synopsis   Wrap existing gdf columns representing adjacency lists in a gdf_graph.
 *             cuGRAPH does not own the memory used to represent this graph. This function does not allocate memory.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in] *offsets               This gdf_column of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in] *indices               This gdf_column of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in] *edge_data (optional)  This pointer can be nullptr. If not, this gdf_column of size E (number of edges) contains the weiht for each edge.
 *                                   The type expected to be floating point.
 *
 * @Param[out] *graph                cuGRAPH graph descriptor containing the newly added adjacency list (edge_data is optional).
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_adj_list_view (gdf_graph *graph,
                             const gdf_column *offsets,
                             const gdf_column *indices,
                             const gdf_column *edge_data);

/**
 * @Synopsis   Create the adjacency lists of a gdf_graph from its edge list.
 *             cuGRAPH allocates and owns the memory required for storing the created adjacency list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out] *graph            in  : graph descriptor containing a valid gdf_edge_list structure pointed by graph->edgeList
 *                                   out : graph->adjList is set to a gdf_adj_list structure containing the generated adjacency list
 *
 * @Returns                          GDF_SUCCESS upon successful completion. If graph->edgeList is nullptr then GDF_INVALID_API_CALL is returned.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_add_adj_list(gdf_graph *graph);

/**
 * @Synopsis   Create the transposed adjacency list from the edge list of a gdf_graph.
 *             cuGRAPH allocates and owns the memory required for storing the created transposed adjacency list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out] *graph            in  : graph descriptor containing either a valid gdf_edge_list structure pointed by graph->edgeList
 *                                         or a valid gdf_adj_list structure pointed by graph->adjList
 *                                   out : graph->transposedAdjList is set to a gdf_adj_list structure containing the generated transposed adjacency list
 *
 * @Returns                          GDF_SUCCESS upon successful completion. If both graph->edgeList and graph->adjList are nullptr then GDF_INVALID_API_CALL is returned.
 */
/* ----------------------------------------------------------------------------*/

gdf_error gdf_add_transpose(gdf_graph *graph);

/**
 * @Synopsis   Create the edge lists of a gdf_graph from its adjacency list.
 *             cuGRAPH allocates and owns the memory required for storing the created edge list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out] *graph            in  : graph descriptor containing a valid gdf_adj_list structure pointed by graph->adjList
 *                                   out : graph->edgeList is set to a gdf_edge_list structure containing the generated edge list
 *
 * @Returns                          GDF_SUCCESS upon successful completion. If graph->adjList is nullptr then GDF_INVALID_API_CALL is returned.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_add_edge_list(gdf_graph *graph);

/**
 * @Synopsis   Deletes the adjacency list of a gdf_graph
 *
 * @Param[in, out] *graph            in  : graph descriptor with graph->adjList pointing to a gdf_adj_list structure
 *                                   out : graph descriptor with graph->adjList set to nullptr
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_delete_adj_list(gdf_graph *graph);

/**
 * @Synopsis   Deletes the edge list of a gdf_graph
 *
 * @Param[in, out] *graph            in  : graph descriptor with graph->edgeList pointing to a gdf_edge_list structure
 *                                   out : graph descriptor with graph->edgeList set to nullptr
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_delete_edge_list(gdf_graph *graph);

/**
 * @Synopsis   Deletes the transposed adjacency list of a gdf_graph
 *
 * @Param[in, out] *graph            in  : graph descriptor with graph->transposedAdjList pointing to a gdf_adj_list structure
 *                                   out : graph descriptor with graph->transposedAdjList set to nullptr
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_delete_transpose(gdf_graph *graph);

/**
 * @Synopsis Find pairs of vertices in the input graph such that each pair is connected by
 *  a path that is two hops in length.
 * @param graph The input graph
 * @param first An uninitialized gdf_column which will be initialized to contain the
 * first entry of each result pair.
 * @param second An uninitialized gdf_column which will be initialized to contain the
 * second entry of each result pair.
 * @return GDF_SUCCESS upon successful completion. */
gdf_error gdf_get_two_hop_neighbors(gdf_graph* graph, gdf_column* first, gdf_column* second);

<<<<<<< HEAD
=======
/**
 * @Synopsis   Performs a breadth first search traversal of a graph starting from a node.
 *
 * @Param[in] *graph                 cuGRAPH graph descriptor with a valid edgeList or adjList
 *
 * @Param[out] *distances            If set to a valid column, this is populated by distance of every vertex in the graph from the starting node
 *
 * @Param[out] *predecessors         If set to a valid column, this is populated by bfs traversal predecessor of every vertex
 *
 * @Param[in] start_node             The starting node for breadth first search traversal
 *
 * @Param[in] directed               Treat the input graph as directed
 *
 * @Returns                          GDF_SUCCESS upon successful completion.
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_bfs(gdf_graph *graph, gdf_column *distances, gdf_column *predecessors, int start_node, bool directed);
gdf_error gdf_jaccard(gdf_graph *graph, void *c_gamma, gdf_column *weights, gdf_column *weight_j);
gdf_error gdf_louvain(gdf_graph *graph, void *final_modularity, void *num_level, gdf_column *louvain_parts);
>>>>>>> master
