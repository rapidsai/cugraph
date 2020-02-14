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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <graph.hpp>
#include <utilities/error_utils.h>

namespace cugraph {
namespace experimental {

namespace detail {

void check_edge_list_columns(cudf::column_view const &src_indices,
                             cudf::column_view const &dst_indices) {
  CUGRAPH_EXPECTS( src_indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( dst_indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( src_indices.type() == dst_indices.type() , "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices.type().id() == cudf::experimental::type_to_id<int32_t>() , "Unsupported data type" );
  CUGRAPH_EXPECTS( src_indices.size() == dst_indices.size() , "Source and Destination must have same number of rows" );
  CUGRAPH_EXPECTS( src_indices.size() > 0 , "No edges");
}

void check_adj_list_columns(cudf::column_view const &indices,
                            cudf::column_view const &offsets) {
  CUGRAPH_EXPECTS( offsets.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( indices.null_count() == 0 , "Input column has non-zero null count");
  CUGRAPH_EXPECTS( offsets.type() == indices.type() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.type().id() == cudf::experimental::type_to_id<int32_t>() , "Unsupported data type" );
  CUGRAPH_EXPECTS( offsets.size() > 0 , "Offsets column is empty");
}

}

/**
 * @Synopsis   Wrap existing cudf columns representing an edge list in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  source_indices        This column_view of size E (number of edges) contains the index of the source for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  destination_indices   This column_view of size E (number of edges) contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    GraphCOO object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCOO<VT,WT> from_edge_list(cudf::column_view const &source_indices,
                               cudf::column_view const &destination_indices,
                               size_t number_of_vertices,
                               size_t number_of_edges) {
  detail::check_edge_list_columns(source_indices, destination_indices);
  return GraphCOO<VT,WT>(source_indices.data<VT>(), destination_indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing an edge list in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  source_indices        This column_view of size E (number of edges) contains the index of the source for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  destination_indices   This column_view of size E (number of edges) contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCOO<VT,WT> from_edge_list(cudf::column_view const &source_indices,
                               cudf::column_view const &destination_indices,
                               cudf::column_view const &edge_data,
                               size_t number_of_vertices,
                               size_t number_of_edges) {
  detail::check_edge_list_columns(source_indices, destination_indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return GraphCOO<VT,WT>(source_indices.data<VT>(), destination_indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCSR<VT,WT> from_adj_list(cudf::column_view const &offsets,
                              cudf::column_view const &indices,
                              size_t number_of_vertices,
                              size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  return GraphCSR<VT,WT>(offsets.data<VT>(), indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCSR<VT,WT> from_adj_list(cudf::column_view const &offsets,
                              cudf::column_view const &indices,
                              cudf::column_view const &edge_data,
                              size_t number_of_vertices,
                              size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return GraphCSR<VT,WT>(offsets.data<VT>(), indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing transposed adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCSC<VT,WT> from_transposed_adj_list(cudf::column_view const &offsets,
                                         cudf::column_view const &indices,
                                         size_t number_of_vertices,
                                         size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  return GraphCSC<VT,WT>(offsets.data<VT>(), indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing transposed adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
GraphCSC<VT,WT> from_transposed_adj_list(cudf::column_view const &offsets,
                                         cudf::column_view const &indices,
                                         cudf::column_view const &edge_data,
                                         size_t number_of_vertices,
                                         size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return GraphCSC<VT,WT>(offsets.data<VT>(), indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}


#if 0
/**
 * @Synopsis   Wrap existing cudf columns representing an edge list in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  source_indices        This column_view of size E (number of edges) contains the index of the source for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  destination_indices   This column_view of size E (number of edges) contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_edge_list(cudf::column_view const &source_indices,
                                                    cudf::column_view const &destination_indices,
                                                    size_t number_of_vertices,
                                                    size_t number_of_edges) {
  detail::check_edge_list_columns(source_indices, destination_indices);
  return Graph<VT,WT>::from_edge_list(source_indices.data<VT>(), destination_indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing an edge list in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  source_indices        This column_view of size E (number of edges) contains the index of the source for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  destination_indices   This column_view of size E (number of edges) contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_edge_list(cudf::column_view const &source_indices,
                                                    cudf::column_view const &destination_indices,
                                                    cudf::column_view const &edge_data,
                                                    size_t number_of_vertices,
                                                    size_t number_of_edges) {
  detail::check_edge_list_columns(source_indices, destination_indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return Graph<VT,WT>::from_edge_list(source_indices.data<VT>(), destination_indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_adj_list(cudf::column_view const &offsets,
                                                   cudf::column_view const &indices,
                                                   size_t number_of_vertices,
                                                   size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  return Graph<VT,WT>::from_adj_list(offsets.data<VT>(), indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_adj_list(cudf::column_view const &offsets,
                                                   cudf::column_view const &indices,
                                                   cudf::column_view const &edge_data,
                                                   size_t number_of_vertices,
                                                   size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return Graph<VT,WT>::from_adj_list(offsets.data<VT>(), indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing transposed adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_transposed_adj_list(cudf::column_view const &offsets,
                                                              cudf::column_view const &indices,
                                                              size_t number_of_vertices,
                                                              size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  return Graph<VT,WT>::from_transposed_adj_list(offsets.data<VT>(), indices.data<VT>(), nullptr, number_of_vertices, number_of_edges);
}

/**
 * @Synopsis   Wrap existing cudf columns representing transposed adjacency lists in a Graph.
 *             cuGRAPH does not own the memory used to represent this graph. This
 *             function does not allocate memory.
 *
 * @Param[in]  offsets               This column_view of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
 *                                   Offsets must be in the range [0, E] (number of edges).
 * @Param[in]  indices               This column_view of size E contains the index of the destination for each edge.
 *                                   Indices must be in the range [0, V-1].
 * @Param[in]  edge_data             This column_view of size E (number of edges) contains the weight for each edge.
 * @Param[in]  number_of_vertices    The number of vertices in the graph
 * @Param[in]  number_of_edges       The number of edges in the graph
 *
 * @Returns    unique pointer to a Graph object
 *
 * @throws     cugraph::logic_error when an error occurs.
 */
template <typename VT = int, typename WT = float>
static std::unique_ptr<Graph<VT,WT>> from_transposed_adj_list(cudf::column_view const &offsets,
                                                              cudf::column_view const &indices,
                                                              cudf::column_view const &edge_data,
                                                              size_t number_of_vertices,
                                                              size_t number_of_edges) {
  detail::check_adj_list_columns(offsets, indices);
  CUGRAPH_EXPECTS( size_t{edge_data.size()} == number_of_edges , "Edge data must contain number_of_edges elements");
  return Graph<VT,WT>::from_transposed_adj_list(offsets.data<VT>(), indices.data<VT>(), edge_data.data<WT>(), number_of_vertices, number_of_edges);
}
#endif


} //namespace experimental
} //namespace cugraph
