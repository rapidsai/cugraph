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

namespace cugraph {
namespace experimental {

typedef enum prop_type{PROP_UNDEF, PROP_FALSE, PROP_TRUE} PropType;

struct GraphProperties {
  bool directed;
  bool weighted;
  bool multigraph;
  bool bipartite;
  bool tree;
  PropType has_negative_edges;
  GraphProperties() : directed(false), weighted(false), multigraph(false), bipartite(false), tree(false), has_negative_edges(PROP_UNDEF){}
};

class EdgeList {
public:
  cudf::column_view const src_indices;   ///< rowInd
  cudf::column_view const dst_indices;  ///< colInd
  cudf::column_view const edge_data;     ///< val
  
  EdgeList(cudf::column_view const &src_indices_,
           cudf::column_view const &dst_indices_,
           cudf::column_view const &edge_data_):
    src_indices(src_indices_),
    dst_indices(dst_indices_),
    edge_data(edge_data_) {}
};

class AdjList {
public:
  cudf::column_view const offsets;       ///< rowPtr
    cudf::column_view const indices;       ///< colInd
    cudf::column_view const edge_data;     ///< val

  /*
  void get_vertex_identifiers(cudf::column_view const &identifiers);
  void get_source_indices(cudf::column_view const &indices);
  */

  AdjList(cudf::column_view const &offsets_,
          cudf::column_view const &indices_,
          cudf::column_view const &edge_data_):
    offsets(offsets_),
    indices(indices_),
    edge_data(edge_data_) {}
};

class Graph {
public:
  EdgeList          edgeList;          ///< COO
  AdjList           adjList;           ///< CSR
  AdjList           transposedAdjList; ///< CSC
  GraphProperties   prop;

  size_t numberOfVertices;
  size_t numberOfEdges;
  
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
  static std::unique_ptr<Graph> from_edge_list(cudf::column_view const &source_indices,
                                               cudf::column_view const &destination_indices,
                                               size_t number_of_vertices,
                                               size_t number_of_edges);

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
  static std::unique_ptr<Graph> from_edge_list(cudf::column_view const &source_indices,
                                               cudf::column_view const &destination_indices,
                                               cudf::column_view const &edge_data,
                                               size_t number_of_vertices,
                                               size_t number_of_edges);

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
  static std::unique_ptr<Graph> from_adj_list(cudf::column_view const &offsets,
                                              cudf::column_view const &indices,
                                              size_t number_of_vertices,
                                              size_t number_of_edges);

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
  static std::unique_ptr<Graph> from_adj_list(cudf::column_view const &offsets,
                                              cudf::column_view const &indices,
                                              cudf::column_view const &edge_data,
                                              size_t number_of_vertices,
                                              size_t number_of_edges);

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
  static std::unique_ptr<Graph> from_transposed_adj_list(cudf::column_view const &offsets,
                                                         cudf::column_view const &indices,
                                                         size_t number_of_vertices,
                                                         size_t number_of_edges);

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
  static std::unique_ptr<Graph> from_transposed_adj_list(cudf::column_view const &offsets,
                                                         cudf::column_view const &indices,
                                                         cudf::column_view const &edge_data,
                                                         size_t number_of_vertices,
                                                         size_t number_of_edges);

 private:
  Graph(EdgeList edgeList_,
        AdjList adjList_,
        AdjList transposedAdjList_,
        size_t numberOfVertices_,
        size_t numberOfEdges_) : edgeList(edgeList_),
                                 adjList(adjList_),
                                 transposedAdjList(transposedAdjList_),
                                 prop(),
                                 numberOfVertices(numberOfVertices_),
                                 numberOfEdges(numberOfEdges_) {
    }
};

} //namespace experimental
} //namespace cugraph
