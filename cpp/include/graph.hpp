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

#include <utilities/error_utils.h>
#include "utilities/graph_utils.cuh"

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

//
// TODO: edge_data should go to the graph, it's the same no matter what format the graph structure is respresented in.
//
template <typename VT = int, typename WT = float>
class EdgeList {
public:
  VT const *src_indices;   ///< rowInd
  VT const *dst_indices;   ///< colInd
  WT const *edge_data;     ///< val
  
  EdgeList(): src_indices(nullptr), dst_indices(nullptr), edge_data(nullptr) {}
  
  EdgeList(VT const *src_indices_, VT const *dst_indices_, WT const *edge_data_):
    src_indices(src_indices_),
    dst_indices(dst_indices_),
    edge_data(edge_data_) {}
};

template <typename VT = int, typename WT = float>
class AdjList {
public:
  VT const *offsets;       ///< CSR/CSC offset range
  VT const *indices;       ///< CSR/CSC indices
  WT const *edge_data;     ///< val
  size_t    offsets_size;  ///< Number of vertices + 1

  void get_vertex_identifiers(VT *identifiers) {
    CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
    cugraph::detail::sequence<VT>(offsets_size, identifiers);
  }
  
  void get_source_indices(VT *indices) {
    CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
    cugraph::detail::offsets_to_indices<VT>(offsets, offsets_size, indices);
  }

  AdjList(): offsets(nullptr), indices(nullptr), edge_data(nullptr), offsets_size(0) {}

  AdjList(VT const *offsets_, VT const *indices_, WT const *edge_data_, size_t offsets_size_):
    offsets(offsets_),
    indices(indices_),
    edge_data(edge_data_),
    offsets_size(offsets_size_) {}
};

template <typename VT = int, typename WT = float>
class Graph {
public:
  EdgeList<VT,WT>          edgeList;          ///< COO
  AdjList<VT,WT>           adjList;           ///< CSR
  AdjList<VT,WT>           transposedAdjList; ///< CSC
  GraphProperties          prop;

  size_t                   numberOfVertices;
  size_t                   numberOfEdges;
  
  /**
   * @Synopsis   Wrap existing arrays representing an edge list in a Graph.
   *             cuGRAPH does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param[in]  source_indices        This array of size E (number of edges) contains the index of the source for each edge.
   *                                   Indices must be in the range [0, V-1].
   * @Param[in]  destination_indices   This array of size E (number of edges) contains the index of the destination for each edge.
   *                                   Indices must be in the range [0, V-1].
   * @Param[in]  edge_data             This array size E (number of edges) contains the weight for each edge.  This array can be null
   *                                   in which case the graph is considered unweighted.
   * @Param[in]  number_of_vertices    The number of vertices in the graph
   * @Param[in]  number_of_edges       The number of edges in the graph
   *
   * @Returns    unique pointer to a Graph object
   *
   * @throws     cugraph::logic_error when an error occurs.
   */
  static std::unique_ptr<Graph> from_edge_list(VT const *source_indices,
                                               VT const *destination_indices,
                                               WT const *edge_data,
                                               size_t number_of_vertices,
                                               size_t number_of_edges) {
    return std::unique_ptr<Graph>(new Graph(EdgeList<VT,WT>(source_indices, destination_indices, edge_data),
                                            AdjList<VT,WT>(),
                                            AdjList<VT,WT>(),
                                            number_of_vertices, number_of_edges));
  }

  /**
   * @Synopsis   Wrap existing arrays representing adjacency lists in a Graph.
   *             cuGRAPH does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param[in]  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                                   Offsets must be in the range [0, E] (number of edges).
   * @Param[in]  indices               This array of size E contains the index of the destination for each edge.
   *                                   Indices must be in the range [0, V-1].
   * @Param[in]  edge_data             This array of size E (number of edges) contains the weight for each edge.  This
   *                                   array can be null in which case the graph is considered unweighted.
   * @Param[in]  number_of_vertices    The number of vertices in the graph
   * @Param[in]  number_of_edges       The number of edges in the graph
   *
   * @Returns    unique pointer to a Graph object
   *
   * @throws     cugraph::logic_error when an error occurs.
   */
  static std::unique_ptr<Graph> from_adj_list(VT const *offsets,
                                              VT const *indices,
                                              WT const *edge_data,
                                              size_t number_of_vertices,
                                              size_t number_of_edges) {
    return std::unique_ptr<Graph>(new Graph(EdgeList<VT,WT>(),
                                            AdjList<VT,WT>(offsets, indices, edge_data, number_of_vertices+1),
                                            AdjList<VT,WT>(),
                                            number_of_vertices, number_of_edges));
  }

  /**
   * @Synopsis   Wrap existing arrays representing transposed adjacency lists in a Graph.
   *             cuGRAPH does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param[in]  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                                   Offsets must be in the range [0, E] (number of edges).
   * @Param[in]  indices               This array of size E contains the index of the destination for each edge.
   *                                   Indices must be in the range [0, V-1].
   * @Param[in]  edge_data             This array of size E (number of edges) contains the weight for each edge.  This array
   *                                   can be null in which case the graph is considered unweighted.
   * @Param[in]  number_of_vertices    The number of vertices in the graph
   * @Param[in]  number_of_edges       The number of edges in the graph
   *
   * @Returns    unique pointer to a Graph object
   *
   * @throws     cugraph::logic_error when an error occurs.
   */
  static std::unique_ptr<Graph> from_transposed_adj_list(VT const *offsets,
                                                         VT const *indices,
                                                         WT const *edge_data,
                                                         size_t number_of_vertices,
                                                         size_t number_of_edges) {
    return std::unique_ptr<Graph>(new Graph(EdgeList<VT,WT>(),
                                            AdjList<VT,WT>(),
                                            AdjList<VT,WT>(offsets, indices, edge_data, number_of_vertices+1),
                                            number_of_vertices, number_of_edges));
  }

 private:
  Graph(EdgeList<VT,WT> edgeList_,
        AdjList<VT,WT> adjList_,
        AdjList<VT,WT> transposedAdjList_,
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
