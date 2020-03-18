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

namespace cugraph {
namespace experimental {

enum class PropType{PROP_UNDEF, PROP_FALSE, PROP_TRUE};

struct GraphProperties {
  bool directed{false};
  bool weighted{false};
  bool multigraph{multigraph};
  bool bipartite{false};
  bool tree{false};
  PropType has_negative_edges{PropType::PROP_UNDEF};
  GraphProperties() = default;
};

/**
 * @brief       Base class graphs, all but vertices and edges
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphBase {
public:
  WT const *edge_data;     ///< edge weight

  GraphProperties          prop;

  VT                       number_of_vertices;
  ET                       number_of_edges;

  GraphBase(WT const *edge_data_, VT number_of_vertices_, ET number_of_edges_):
    edge_data(edge_data_),
    prop(),
    number_of_vertices(number_of_vertices_),
    number_of_edges(number_of_edges_)
  {}
};

/**
 * @brief       A graph stored in COO (COOrdinate) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCOO: public GraphBase<VT, ET, WT> {
public:
  VT const *src_indices{nullptr};   ///< rowInd
  VT const *dst_indices{nullptr};   ///< colInd

  /**
   * @brief      Default constructor
   */
  GraphCOO(): GraphBase<VT,ET,WT>(nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing an edge list in a Graph.
   *
   *             GraphCOO does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @param  source_indices        This array of size E (number of edges) contains the index of the source for each edge.
   *                               Indices must be in the range [0, V-1].
   * @param  destination_indices   This array of size E (number of edges) contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @param  edge_data             This array size E (number of edges) contains the weight for each edge.  This array can be null
   *                               in which case the graph is considered unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCOO(VT const *src_indices_, VT const *dst_indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphBase<VT,ET,WT>(edge_data_, number_of_vertices_, number_of_edges_),
    src_indices(src_indices_), dst_indices(dst_indices_)
  {}
};

/**
 * @brief       Base class for graph stored in CSR (Compressed Sparse Row) format or CSC (Compressed Sparse Column) format
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCompressedSparseBase: public GraphBase<VT,ET,WT> {
public:
  ET const *offsets{nullptr};       ///< CSR offsets
  VT const *indices{nullptr};       ///< CSR indices

  /**
   * @brief      Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifier      Pointer to device memory to store the vertex identifiers
   */
  void get_vertex_identifiers(VT *identifiers) const;
  
  /**
   * @brief      Fill the identifiers in the array with the source vertex identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the source vertex identifiers
   */
  void get_source_indices(VT *src_indices) const;

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSR does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                               Offsets must be in the range [0, E] (number of edges).
   * @param  indices               This array of size E contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges) contains the weight for each edge.  This
   *                               array can be null in which case the graph is considered unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCompressedSparseBase(ET const *offsets_, VT const *indices_, WT const *edge_data_,
                            VT number_of_vertices_, ET number_of_edges_):
    GraphBase<VT,ET,WT>(edge_data_, number_of_vertices_, number_of_edges_),
    offsets{offsets_},
    indices{indices_}
  {}
};

/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSR: public GraphCompressedSparseBase<VT,ET,WT> {
public:
  /**
   * @brief      Default constructor
   */
  GraphCSR(): GraphCompressedSparseBase<VT,ET,WT>(nullptr, nullptr, nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSR does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                               Offsets must be in the range [0, E] (number of edges).
   * @param  indices               This array of size E contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges) contains the weight for each edge.  This
   *                               array can be null in which case the graph is considered unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSR(ET const *offsets_, VT const *indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphCompressedSparseBase<VT,ET,WT>(offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {}
};

/**
 * @brief       A graph stored in CSC (Compressed Sparse Column) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSC: public GraphCompressedSparseBase<VT,ET,WT> {
public:
  /**
   * @brief      Default constructor
   */
  GraphCSC(): GraphCompressedSparseBase<VT,ET,WT>(nullptr, nullptr, nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing transposed adjacency lists in a Graph.
   *             GraphCSC does not own the memory used to represent this graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of vertices) contains the offset of adjacency lists of every vertex.
   *                               Offsets must be in the range [0, E] (number of edges).
   * @param  indices               This array of size E contains the index of the destination for each edge.
   *                               Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges) contains the weight for each edge.  This array
   *                               can be null in which case the graph is considered unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSC(ET const *offsets_, VT const *indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphCompressedSparseBase<VT,ET,WT>(offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {}
};

} //namespace experimental
} //namespace cugraph
