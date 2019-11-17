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
/* ----------------------------------------------------------------------------*/

/**
 * @Synopsis Renumber source and destination indexes to be a dense numbering,
 *           using contiguous values between 0 and number of vertices minus 1.
 *
 *    Assumptions:
 *       * source and dest have same size and type
 *       * source and dest are either GDF_INT32 or GDF_INT64
 *       * source and dest have a size greater than 0
 *
 *    Note that this function allocates memory for the src_renumbered,
 *    dst_renumbered and numbering_map arrays.
 *
 * @Param[in]  src - the original source vertices
 * @Param[in]  dst - the original dest vertices
 * @Param[out] src_renumbered - the renumbered source vertices.  This array
 *                              will be a GDF_INT32 array.
 * @Param[out] dst_renumbered - the renumbered dest vertices.  This array
 *                              will be a GDF_INT32 array.
 * @Param[out] numbering_map - mapping of new vertex ids to old vertex ids.
 *                             This array will match the type of src/dst input.
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
void renumber_vertices(const gdf_column *src, const gdf_column *dst,
				               gdf_column *src_renumbered, gdf_column *dst_renumbered,
				               gdf_column *numbering_map);

/**
 * @Synopsis   Wrap existing gdf columns representing an edge list in a Graph.
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
 * @Param[out]* graph                cuGRAPH graph descriptor containing the newly added edge list (edge_data is optional).
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void edge_list_view(Graph* graph,
                    const gdf_column *source_indices,
                    const gdf_column *destination_indices,
                    const gdf_column *edge_data);

/**
 * @Synopsis   Wrap existing gdf columns representing adjacency lists in a Graph.
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
 * @Param[out]* graph                cuGRAPH graph descriptor containing the newly added adjacency list (edge_data is optional).
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void adj_list_view (Graph* graph,
                    const gdf_column *offsets,
                    const gdf_column *indices,
                    const gdf_column *edge_data);

/**
 * @Synopsis   Create the adjacency lists of a Graph from its edge list.
 *             cuGRAPH allocates and owns the memory required for storing the created adjacency list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out]* graph            in  : graph descriptor containing a valid gdf_edge_list structure pointed by graph->edgeList
 *                                   out : graph->adjList is set to a gdf_adj_list structure containing the generated adjacency list
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void transposed_adj_list_view (Graph *graph,
                               const gdf_column *offsets,
                               const gdf_column *indices,
                               const gdf_column *edge_data);

/**
 * @Synopsis   Create the transposed adjacency lists of a gdf_graph from its edge list.
 *             cuGRAPH allocates and owns the memory required for storing the created adjacency list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out] *graph            in  : graph descriptor containing a valid gdf_edge_list structure pointed by graph->edgeList
 *                                   out : graph->adjList is set to a gdf_adj_list structure containing the generated adjacency list
 *
 * @Returns                          GDF_SUCCESS upon successful completion. If graph->edgeList is nullptr then GDF_INVALID_API_CALL is returned.
 */
/* ----------------------------------------------------------------------------*/
void add_adj_list(Graph* graph);

/**
 * @Synopsis   Create the transposed adjacency list from the edge list of a Graph.
 *             cuGRAPH allocates and owns the memory required for storing the created transposed adjacency list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out]* graph            in  : graph descriptor containing either a valid gdf_edge_list structure pointed by graph->edgeList
 *                                         or a valid gdf_adj_list structure pointed by graph->adjList
 *                                   out : graph->transposedAdjList is set to a gdf_adj_list structure containing the generated transposed adjacency list
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/

void add_transposed_adj_list(Graph* graph);

/**
 * @Synopsis   Create the edge lists of a Graph from its adjacency list.
 *             cuGRAPH allocates and owns the memory required for storing the created edge list.
 *             This function does not delete any existing data in the cuGRAPH graph descriptor
 *
 * @Param[in, out]* graph            in  : graph descriptor containing a valid gdf_adj_list structure pointed by graph->adjList
 *                                   out : graph->edgeList is set to a gdf_edge_list structure containing the generated edge list
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void add_edge_list(Graph* graph);

/**
 * @Synopsis   Deletes the adjacency list of a Graph
 *
 * @Param[in, out]* graph            in  : graph descriptor with graph->adjList pointing to a gdf_adj_list structure
 *                                   out : graph descriptor with graph->adjList set to nullptr
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void delete_adj_list(Graph* graph);

/**
 * @Synopsis   Deletes the edge list of a Graph
 *
 * @Param[in, out]* graph            in  : graph descriptor with graph->edgeList pointing to a gdf_edge_list structure
 *                                   out : graph descriptor with graph->edgeList set to nullptr
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void delete_edge_list(Graph* graph);

/**
 * @Synopsis   Deletes the transposed adjacency list of a Graph
 *
 * @Param[in, out]* graph            in  : graph descriptor with graph->transposedAdjList pointing to a gdf_adj_list structure
 *                                   out : graph descriptor with graph->transposedAdjList set to nullptr
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void delete_transposed_adj_list(Graph* graph);

/**
 * @Synopsis   Find pairs of vertices in the input graph such that each pair is connected by
 *             a path that is two hops in length.
 *
 * @param[in]* graph                 in  : graph descriptor with graph->adjList pointing to a gdf_adj_list structure
 *
 * @param[out] first                 out : An uninitialized gdf_column which will be initialized to contain the
 *                                         first entry of each result pair.
 * @param[out] second                out : An uninitialized gdf_column which will be initialized to contain the
 *                                         second entry of each result pair.
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void get_two_hop_neighbors(Graph* graph, gdf_column* first, gdf_column* second);

/**
 * @Synopsis   Single node Multi GPU CSR sparse matrix multiply, x=Ax. 
 *             Should be called in an omp parallel section with one thread per device.
 *             Each device is expected to have a part of the matrix and a copy of the vector
 *             This function is designed for 1D decomposition. Each partition should have local offsets.
 *
 * @Param[in] *part_offsets          in  : Vertex offsets for each partition. This information should be available on all threads/devices
 *                                         part_offsets[device_id] contains the global ID of the first vertex of the partion owned by device_id. 
 *                                         part_offsets[num_devices] contains the global number of vertices
 * @Param[in] off                    in  : Local adjacency list offsets. Starting at 0. The last element contains the local number of edges owned by the partition.
 * @Param[in] ind                    in  : Local adjacency list indices. Indices are between 0 and the global number of edges. 
 * @Param[in] val                    in  : Local adjacency list values. Type should be float or double.
 *
 * @Param[in, out] **x_col           in  : x[device_id] contains the input vector of the spmv for a device_id. The input should be duplicated on all devices.
 *                                   out : Overwritten on output by the result of x = A*x, on all devices.
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void snmg_csrmv (size_t * part_offsets, gdf_column * off, gdf_column * ind, gdf_column * val, gdf_column ** x_col);

/**
 * @Synopsis   Computes degree(in, out, in+out) of all the nodes of a Graph
 *
 * @Param[in]* graph                 in  : graph descriptor with graph->transposedAdjList or graph->adjList present
 * @Param[in] x                      in  : integer value indicating type of degree calculation
 *                                         0 : in+out degree
 *                                         1 : in-degree
 *                                         2 : out-degree
 *
 * @Param[out] *degree               out : gdf_column of size V (V is number of vertices) initialized to zeros.
 *                                         Contains the computed degree of every vertex.
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void degree(Graph* graph, gdf_column *degree, int x);
int get_device(const void *ptr);

/**
 * @Synopsis   Compute number of vertices from the edge list
 *
 * @Param[in, out]* graph            in  : graph descriptor with graph->edgeList populated
 *                                   out : graph descriptor with graph->numberOfVertices populated
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
/* ----------------------------------------------------------------------------*/
void number_of_vertices(Graph* graph);

} //namespace cugraph
