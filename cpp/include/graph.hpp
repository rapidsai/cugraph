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

  // TODO:
  //    New idea... get rid of EdgeList and AdjList
  //    Create three classes:
  //        GraphCOO<VT,WT>
  //        GraphCSR<VT,WT>
  //        GraphCSC<VT,WT>
  //
  //    Then checks to be sure we're using the correct graph type can be done at compile time.
  //

struct GraphProperties {
  bool directed;
  bool weighted;
  bool multigraph;
  bool bipartite;
  bool tree;
  PropType has_negative_edges;
  GraphProperties() : directed(false), weighted(false), multigraph(false), bipartite(false), tree(false), has_negative_edges(PROP_UNDEF){}
};

/**
 * @Synopsis    A graph stored in COO (COOrdinate) format.
 *
 * @tparam VT   Type of vertex (defaults to int)
 * @tparam WT   Type of weight (defaults to float)
 */
template <typename VT = int, typename WT = float>
class GraphCOO {
public:
  VT const *src_indices;   ///< rowInd
  VT const *dst_indices;   ///< colInd
  WT const *edge_data;     ///< edge weight

  GraphProperties          prop;

  size_t                   number_of_vertices;
  size_t                   number_of_edges;

  /**
   * @Synopsis   Wrap existing arrays representing an edge list in a Graph.
   *             GraphCOO does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param  source_indices        This array of size E (number of edges) contains the index of the source for each edge.
   *                               Indices must be in the range [0, V-1].
   * @Param  destination_indices   This array of size E (number of edges) contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @Param  edge_data             This array size E (number of edges) contains the weight for each edge.  This array can be null
   *                               in which case the graph is considered unweighted.
   * @Param  number_of_vertices    The number of vertices in the graph
   * @Param  number_of_edges       The number of edges in the graph
   */
  GraphCOO(VT const *src_indices_, VT const *dst_indices_, WT const *edge_data_,
           size_t number_of_vertices_, size_t number_of_edges_):
    src_indices(src_indices_),
    dst_indices(dst_indices_),
    prop(),
    edge_data(edge_data_),
    number_of_vertices(number_of_vertices_),
    number_of_edges(number_of_edges_)
  {}
};

/**
 * @Synopsis    A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam VT   Type of vertex (defaults to int)
 * @tparam WT   Type of weight (defaults to float)
 */
template <typename VT = int, typename WT = float>
class GraphCSR {
public:
  VT const *offsets;       ///< CSR offsets
  VT const *indices;       ///< CSR indices
  WT const *edge_data;     ///< edge weight

  GraphProperties          prop;

  size_t                   number_of_vertices;
  size_t                   number_of_edges;

  /**
   * @Synopsis    Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifier      Pointer to device memory to store the vertex identifiers
   */
  void get_vertex_identifiers(VT *identifiers) const {
    CUGRAPH_EXPECTS( offsets != nullptr , "No graph specified");
    cugraph::detail::sequence<VT>(number_of_vertices, identifiers);
  }
  
  /**
   * @Synopsis    Fill the identifiers in the array with the source vertex identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the source vertex identifiers
   */
  void get_source_indices(VT *src_indices) const {
    CUGRAPH_EXPECTS( offsets != nullptr , "No graph specified");
    cugraph::detail::offsets_to_indices<VT>(offsets, number_of_vertices+1, src_indices);
  }

  /**
   * @Synopsis   Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSR does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                               Offsets must be in the range [0, E] (number of edges).
   * @Param  indices               This array of size E contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @Param  edge_data             This array of size E (number of edges) contains the weight for each edge.  This
   *                               array can be null in which case the graph is considered unweighted.
   * @Param  number_of_vertices    The number of vertices in the graph
   * @Param  number_of_edges       The number of edges in the graph
   */
  GraphCSR(VT const *offsets_, VT const *indices_, WT const *edge_data_,
           size_t number_of_vertices_, size_t number_of_edges_):
    offsets(offsets_),
    indices(indices_),
    prop(),
    edge_data(edge_data_),
    number_of_vertices(number_of_vertices_),
    number_of_edges(number_of_edges_)
  {}
};

/**
 * @Synopsis    A graph stored in CSC (Compressed Sparse Column) format.
 *
 * @tparam VT   Type of vertex (defaults to int)
 * @tparam WT   Type of weight (defaults to float)
 */
template <typename VT = int, typename WT = float>
class GraphCSC {
public:
  VT const *offsets;       ///< CSC offsets
  VT const *indices;       ///< CSC indices
  WT const *edge_data;     ///< edge weight

  GraphProperties          prop;

  size_t                   number_of_vertices;
  size_t                   number_of_edges;

  /**
   * @Synopsis    Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifier      Pointer to device memory to store the vertex identifiers
   */
  void get_vertex_identifiers(VT *identifiers) const {
    CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
    cugraph::detail::sequence<VT>(number_of_vertices, identifiers);
  }
  
  /**
   * @Synopsis    Fill the identifiers in the array with the source vertex identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the source vertex identifiers
   */
  //  TODO:  Should this be get_destination_indices, since we're CSC?  The source
  //         indices will be in the indices array and the offsets are by destination, aren't they?
  void get_source_indices(VT *src_indices) const {
    CUGRAPH_EXPECTS( offsets != nullptr , "Invalid API parameter");
    cugraph::detail::offsets_to_indices<VT>(offsets, number_of_vertices+1, src_indices);
  }

  /**
   * @Synopsis   Wrap existing arrays representing transposed adjacency lists in a Graph.
   *             GraphCSC does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @Param  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                               Offsets must be in the range [0, E] (number of edges).
   * @Param  indices               This array of size E contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @Param  edge_data             This array of size E (number of edges) contains the weight for each edge.  This array
   *                               can be null in which case the graph is considered unweighted.
   * @Param  number_of_vertices    The number of vertices in the graph
   * @Param  number_of_edges       The number of edges in the graph
   */
  GraphCSC(VT const *offsets_, VT const *indices_, WT const *edge_data_,
           size_t number_of_vertices_, size_t number_of_edges_):
    offsets(offsets_),
    indices(indices_),
    prop(),
    edge_data(edge_data_),
    number_of_vertices(number_of_vertices_),
    number_of_edges(number_of_edges_)
  {}
};

} //namespace experimental
} //namespace cugraph
