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
#include <rmm/device_buffer.hpp>

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
class GraphViewBase {
public:
  WT const *edge_data;     ///< edge weight

  GraphProperties          prop;

  VT                       number_of_vertices;
  ET                       number_of_edges;

  GraphViewBase(WT const *edge_data_, VT number_of_vertices_, ET number_of_edges_):
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
class GraphCOOView: public GraphViewBase<VT, ET, WT> {
public:
  VT const *src_indices{nullptr};   ///< rowInd
  VT const *dst_indices{nullptr};   ///< colInd

  /**
   * @brief      Default constructor
   */
  GraphCOOView(): GraphViewBase<VT,ET,WT>(nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing an edge list in a Graph.
   *
   *             GraphCOOView does not own the memory used to represent this graph. This
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
  GraphCOOView(VT const *src_indices_, VT const *dst_indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphViewBase<VT,ET,WT>(edge_data_, number_of_vertices_, number_of_edges_),
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
class GraphCompressedSparseViewBase: public GraphViewBase<VT,ET,WT> {
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
  GraphCompressedSparseViewBase(ET const *offsets_, VT const *indices_, WT const *edge_data_,
                            VT number_of_vertices_, ET number_of_edges_):
    GraphViewBase<VT,ET,WT>(edge_data_, number_of_vertices_, number_of_edges_),
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
class GraphCSRView: public GraphCompressedSparseViewBase<VT,ET,WT> {
public:
  /**
   * @brief      Default constructor
   */
  GraphCSRView(): GraphCompressedSparseViewBase<VT,ET,WT>(nullptr, nullptr, nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this graph. This
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
  GraphCSRView(ET const *offsets_, VT const *indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphCompressedSparseViewBase<VT,ET,WT>(offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
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
class GraphCSCView: public GraphCompressedSparseViewBase<VT,ET,WT> {
public:
  /**
   * @brief      Default constructor
   */
  GraphCSCView(): GraphCompressedSparseViewBase<VT,ET,WT>(nullptr, nullptr, nullptr, 0, 0) {}
  
  /**
   * @brief      Wrap existing arrays representing transposed adjacency lists in a Graph.
   *             GraphCSCView does not own the memory used to represent this graph. This
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
  GraphCSCView(ET const *offsets_, VT const *indices_, WT const *edge_data_,
           VT number_of_vertices_, ET number_of_edges_):
    GraphCompressedSparseViewBase<VT,ET,WT>(offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {}
};

/**
 * @brief       A constructed graph stored in COO (COOrdinate) format.
 *
 * This class will src_indices and dst_indicies (until moved)
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCOO {
  VT number_of_vertices_;
  ET number_of_edges_;
  rmm::device_buffer src_indices_{};   ///< rowInd
  rmm::device_buffer dst_indices_{};   ///< colInd
  rmm::device_buffer edge_data_{}; ///< CSR data

public:

  /**
   * @brief      Take ownership of the provided graph arrays in COO format
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
  GraphCOO(VT number_of_vertices,
           ET number_of_edges,
           bool has_data = false):
    number_of_vertices_(number_of_vertices),
    number_of_edges_(number_of_edges_),
    src_indices_(sizeof(VT)*number_of_edges),
    dst_indices_(sizeof(VT)*number_of_edges)
    edge_data_(has_data? sizeof(WT)*number_of_edges : 0)
  {}

  VT* src_indices(void) { return static_cast<VT*>(src_indices_.data()); }
  VT* dst_indices(void) { return static_cast<VT*>(dst_indices_.data()); }
  WT* edge_data(void) { return static_cast<WT*>(edge_data_.data()); }

  struct contents {
    std::unique_ptr<rmm::device_buffer> src_indices;
    std::unique_ptr<rmm::device_buffer> dst_indices;
    std::unique_ptr<rmm::device_buffer> edge_data;
  };

  contents release() noexcept;

};

/**
 * @brief       Base class for constructted graphs stored in CSR (Compressed Sparse Row) format or CSC (Compressed Sparse Column) format
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCompressedSparseBase {
  VT number_of_vertices_;
  ET number_of_edges_;
  rmm::device_buffer offsets_{};   ///< CSR offsets
  rmm::device_buffer indices_{};   ///< CSR indices
  rmm::device_buffer edge_data_{}; ///< CSR data

public:

  /**
   * @brief      Take ownership of the provided graph arrays in CSR/CSC format
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
  GraphCompressedSparseBase(VT number_of_vertices,
                            ET number_of_edges,
                            bool has_data):
    number_of_vertices_(number_of_vertices),
    number_of_edges_(number_of_edges_),
    offsets_(sizeof(ET)*(number_of_vertices + 1)),
    indices_(sizeof(VT)*number_of_edges),
    edge_data_(has_data? sizeof(WT)*number_of_edges : 0)
  {}

  ET* offsets(void) { return static_cast<ET*>(offsets_.data()); }
  VT* indices(void) { return static_cast<VT*>(indices_.data()); }
  WT* edge_data(void) { return static_cast<WT*>(edge_data_.data()); }

  struct contents {
    std::unique_ptr<rmm::device_buffer> offsets;
    std::unique_ptr<rmm::device_buffer> indices;
    std::unique_ptr<rmm::device_buffer> edge_data;
  };

  contents release() noexcept;

};

/**
 * @brief       A constructed graph stored in CSR (Compressed Sparse Row) format.
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
  GraphCSR(): GraphCompressedSparseBase<VT,ET,WT>() {}
  
  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
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
  GraphCSR(VT number_of_vertices_,
           ET number_of_edges_,
           bool has_data = false):
    GraphCompressedSparseBase<VT,ET,WT>(number_of_vertices_, number_of_edges_, has_data_)
};

/**
 * @brief       A constructed graph stored in CSC (Compressed Sparse Column) format.
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
  GraphCSC(): GraphCompressedSparseBase<VT,ET,WT>() {}
  
  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
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
  GraphCSC(VT number_of_vertices_,
           ET number_of_edges_,
           bool has_data = false):
    GraphCompressedSparseBase<VT,ET,WT>(number_of_vertices_, number_of_edges_, has_data_)
  {}
};

} //namespace experimental
} //namespace cugraph
} //namespace experimental
} //namespace cugraph
