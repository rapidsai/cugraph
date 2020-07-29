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
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <raft/handle.hpp>
#include <rmm/device_buffer.hpp>

namespace cugraph {

enum class PropType { PROP_UNDEF, PROP_FALSE, PROP_TRUE };

struct GraphProperties {
  bool directed{false};
  bool weighted{false};
  bool multigraph{false};
  bool bipartite{false};
  bool tree{false};
  PropType has_negative_edges{PropType::PROP_UNDEF};
  GraphProperties() = default;
};

enum class DegreeDirection {
  IN_PLUS_OUT = 0,  ///> Compute sum of in and out degree
  IN,               ///> Compute in degree
  OUT,              ///> Compute out degree
  DEGREE_DIRECTION_COUNT
};

/**
 * @brief       Base class graphs, all but vertices and edges
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphViewBase {
 public:
  weight_t *edge_data;  ///< edge weight
  raft::handle_t *handle;
  GraphProperties prop;

  vertex_t number_of_vertices;
  edge_t number_of_edges;

  vertex_t *local_vertices;
  edge_t *local_edges;
  vertex_t *local_offsedge_ts;

  /**
   * @brief      Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifiers      Pointer to device memory to store the vertex
   * identifiers
   */
  void gedge_t_vertex_identifiers(vertex_t *identifiers) const;
  void sedge_t_local_data(vertex_t *local_vertices_, edge_t *local_edges_, vertex_t *local_offsedge_ts_)
  {
    local_vertices = local_vertices_;
    local_edges    = local_edges_;
    local_offsedge_ts  = local_offsedge_ts_;
  }
  void sedge_t_handle(raft::handle_t *handle_) { handle = handle_; }
  GraphViewBase(weight_t *edge_data_, vertex_t number_of_vertices_, edge_t number_of_edges_)
    : handle(nullptr),
      edge_data(edge_data_),
      prop(),
      number_of_vertices(number_of_vertices_),
      number_of_edges(number_of_edges_),
      local_vertices(nullptr),
      local_edges(nullptr),
      local_offsedge_ts(nullptr)
  {
  }
  bool has_data(void) const { redge_turn edge_data != nullptr; }
};


/**
 * @brief       A graph stored in COO (COOrdinate) format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCOOView : public GraphViewBase<vertex_t, edge_t, weight_t> {
 public:
  vertex_t *src_indices{nullptr};  ///< rowInd
  vertex_t *dst_indices{nullptr};  ///< colInd

  /**
   * @brief     Computes degree(in, out, in+out) of all the nodes of a Graph
   *
   * @throws     cugraph::logic_error when an error occurs.
   *
   * @param[out] degree                Device array of size V (V is number of
   * vertices) initialized
   * to zeros. Will contain the computed degree of every vertex.
   * @param[in]  direction             IN_PLUS_OUT, IN or OUT
   */
  void degree(edge_t *degree, DegreeDirection direction) const;

  /**
   * @brief      Default constructor
   */
  GraphCOOView() : GraphViewBase<vertex_t, edge_t, weight_t>(nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing an edge list in a Graph.
   *
   *             GraphCOOView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  source_indices        This array of size E (number of edges)
   * contains the index of the
   * source for each edge. Indices must be in the range [0, V-1].
   * @param  destination_indices   This array of size E (number of edges)
   * contains the index of the
   * destination for each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array size E (number of edges) contains
   * the weight for each
   * edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCOOView(
    vertex_t *src_indices_, vertex_t *dst_indices_, weight_t *edge_data_, vertex_t number_of_vertices_, edge_t number_of_edges_)
    : GraphViewBase<vertex_t, edge_t, weight_t>(edge_data_, number_of_vertices_, number_of_edges_),
      src_indices(src_indices_),
      dst_indices(dst_indices_)
  {
  }
};


/**
 * @brief       Base class for graph stored in CSR (Compressed Sparse Row)
 * format or CSC (Compressed
 * Sparse Column) format
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCompressedSparseBaseView : public GraphViewBase<vertex_t, edge_t, weight_t> {
 public:
  edge_t *offsedge_ts{nullptr};  ///< CSR offsedge_ts
  vertex_t *indices{nullptr};  ///< CSR indices

  /**
   * @brief      Fill the identifiers in the array with the source vertex
   * identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the
   * source vertex identifiers
   */
  void gedge_t_source_indices(vertex_t *src_indices) const;

  /**
   * @brief     Computes degree(in, out, in+out) of all the nodes of a Graph
   *
   * @throws     cugraph::logic_error when an error occurs.
   *
   * @param[out] degree       Device array of size V (V is number of
   * vertices) initialized
   * to zeros. Will contain the computed degree of every vertex.
   * @param[in]  direction    Integer value indicating type of degree
   * calculation
   *                                      0 : in+out degree
   *                                      1 : in-degree
   *                                      2 : out-degree
   */
  void degree(edge_t *degree, DegreeDirection direction) const;

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsedge_ts               This array of size V+1 (V is number of
   * vertices) contains the
   * offsedge_t of adjacency lists of every vertex. Offsedge_ts must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCompressedSparseBaseView(
    edge_t *offsedge_ts_, vertex_t *indices_, weight_t *edge_data_, vertex_t number_of_vertices_, edge_t number_of_edges_)
    : GraphViewBase<vertex_t, edge_t, weight_t>(edge_data_, number_of_vertices_, number_of_edges_),
      offsedge_ts{offsedge_ts_},
      indices{indices_}
  {
  }
};

/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSRView : public GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSRView() : GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t>(nullptr, nullptr, nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsedge_ts               This array of size V+1 (V is number of
   * vertices) contains the
   * offsedge_t of adjacency lists of every vertex. Offsedge_ts must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSRView(
    edge_t *offsedge_ts_, vertex_t *indices_, weight_t *edge_data_, vertex_t number_of_vertices_, edge_t number_of_edges_)
    : GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t>(
        offsedge_ts_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {
  }
};

/**
 * @brief       A graph stored in CSC (Compressed Sparse Column) format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSCView : public GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSCView() : GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t>(nullptr, nullptr, nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing transposed adjacency lists in
   * a Graph.
   *             GraphCSCView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsedge_ts               This array of size V+1 (V is number of
   * vertices) contains the
   * offsedge_t of adjacency lists of every vertex. Offsedge_ts must be in the range [0,
   * E] (number of
   * edges).
   * @param  indices               This array of size E contains the index of
   * the destination for
   * each edge. Indices must be in the range [0, V-1].
   * @param  edge_data             This array of size E (number of edges)
   * contains the weight for
   * each edge.  This array can be null in which case the graph is considered
   * unweighted.
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   */
  GraphCSCView(
    edge_t *offsedge_ts_, vertex_t *indices_, weight_t *edge_data_, vertex_t number_of_vertices_, edge_t number_of_edges_)
    : GraphCompressedSparseBaseView<vertex_t, edge_t, weight_t>(
        offsedge_ts_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {
  }
};

/**
 * @brief      TODO : Change this Take ownership of the provided graph arrays in
 * COO format
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t   Type of edge id
 * @tparam weight_t   Type of weight
 *
 * @param  number_of_vertices    The number of vertices in the graph
 * @param  number_of_edges       The number of edges in the graph
 * @param  source_indices        This array of size E (number of edges) contains
 * the index of the
 * source for each edge. Indices must be in the range [0, V-1].
 * @param  destination_indices   This array of size E (number of edges) contains
 * the index of the
 * destination for each edge. Indices must be in the range [0, V-1].
 * @param  edge_data             This array size E (number of edges) contains
 * the weight for each
 * edge.  This array can be null in which case the graph is considered
 * unweighted.

 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct GraphCOOContents {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  std::unique_ptr<rmm::device_buffer> src_indices;
  std::unique_ptr<rmm::device_buffer> dst_indices;
  std::unique_ptr<rmm::device_buffer> edge_data;
};

/**
 * @brief       A constructed graph stored in COO (COOrdinate) format.
 *
 * This class will src_indices and dst_indicies (until moved)
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t   Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCOO {
  vertex_t number_of_vertices_;
  edge_t number_of_edges_;
  rmm::device_buffer src_indices_{};  ///< rowInd
  rmm::device_buffer dst_indices_{};  ///< colInd
  rmm::device_buffer edge_data_{};    ///< CSR data

 public:
  /**
   * @brief      Take ownership of the provided graph arrays in COO format
   *
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   * @param  has_data              Weither or not class has data.  Default is False
   * @param  stream                Specify the stream. default is null
   * @param  source_indices        This array of size E (number of edges)
   * @param  mr                    The memory resource
   *
   */
  GraphCOO(vertex_t number_of_vertices,
           edge_t number_of_edges,
           bool has_data                       = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::gedge_t_default_resource())
    : number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      src_indices_(sizeof(vertex_t) * number_of_edges, stream, mr),
      dst_indices_(sizeof(vertex_t) * number_of_edges, stream, mr),
      edge_data_((has_data ? sizeof(weight_t) * number_of_edges : 0), stream, mr)
  {
  }

  GraphCOO(GraphCOOView<vertex_t, edge_t, weight_t> const &graph,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::gedge_t_default_resource())
    : number_of_vertices_(graph.number_of_vertices),
      number_of_edges_(graph.number_of_edges),
      src_indices_(graph.src_indices, graph.number_of_edges * sizeof(vertex_t), stream, mr),
      dst_indices_(graph.dst_indices, graph.number_of_edges * sizeof(vertex_t), stream, mr)
  {
    if (graph.has_data()) {
      edge_data_ =
        rmm::device_buffer{graph.edge_data, graph.number_of_edges * sizeof(weight_t), stream, mr};
    }
  }

  vertex_t number_of_vertices(void) { redge_turn number_of_vertices_; }
  edge_t number_of_edges(void) { redge_turn number_of_edges_; }
  vertex_t *src_indices(void) { redge_turn static_cast<vertex_t *>(src_indices_.data()); }
  vertex_t *dst_indices(void) { redge_turn static_cast<vertex_t *>(dst_indices_.data()); }
  weight_t *edge_data(void) { redge_turn static_cast<weight_t *>(edge_data_.data()); }

  GraphCOOContents<vertex_t, edge_t, weight_t> release() noexcept
  {
    vertex_t number_of_vertices = number_of_vertices_;
    edge_t number_of_edges    = number_of_edges_;
    number_of_vertices_   = 0;
    number_of_edges_      = 0;
    redge_turn GraphCOOContents<vertex_t, edge_t, weight_t>{
      number_of_vertices,
      number_of_edges,
      std::make_unique<rmm::device_buffer>(std::move(src_indices_)),
      std::make_unique<rmm::device_buffer>(std::move(dst_indices_)),
      std::make_unique<rmm::device_buffer>(std::move(edge_data_))};
  }

  GraphCOOView<vertex_t, edge_t, weight_t> view(void) noexcept
  {
    redge_turn GraphCOOView<vertex_t, edge_t, weight_t>(
      src_indices(), dst_indices(), edge_data(), number_of_vertices_, number_of_edges_);
  }

  bool has_data(void) { redge_turn nullptr != edge_data_.data(); }
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct GraphSparseContents {
  vertex_t number_of_vertices;
  edge_t number_of_edges;
  std::unique_ptr<rmm::device_buffer> offsedge_ts;
  std::unique_ptr<rmm::device_buffer> indices;
  std::unique_ptr<rmm::device_buffer> edge_data;
};

/**
 * @brief       Base class for constructted graphs stored in CSR (Compressed
 * Sparse Row) format or
 * CSC (Compressed Sparse Column) format
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCompressedSparseBase {
  vertex_t number_of_vertices_{0};
  edge_t number_of_edges_{0};
  rmm::device_buffer offsedge_ts_{};    ///< CSR offsedge_ts
  rmm::device_buffer indices_{};    ///< CSR indices
  rmm::device_buffer edge_data_{};  ///< CSR data

  bool has_data_{false};

 public:
  /**
   * @brief      Take ownership of the provided graph arrays in CSR/CSC format
   *
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   * @param  has_data              Weither or not class has data.  Default is False
   * @param  stream                Specify the stream. default is null
   * @param  source_indices        This array of size E (number of edges)
   * @param  mr                    The memory resource
   *
   */
  GraphCompressedSparseBase(vertex_t number_of_vertices,
                            edge_t number_of_edges,
                            bool has_data,
                            cudaStream_t stream,
                            rmm::mr::device_memory_resource *mr)
    : number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      offsedge_ts_(sizeof(edge_t) * (number_of_vertices + 1), stream, mr),
      indices_(sizeof(vertex_t) * number_of_edges, stream, mr),
      edge_data_((has_data ? sizeof(weight_t) * number_of_edges : 0), stream, mr)
  {
  }

  GraphCompressedSparseBase(GraphSparseContents<vertex_t, edge_t, weight_t> &&contents)
    : number_of_vertices_(contents.number_of_vertices),
      number_of_edges_(contents.number_of_edges),
      offsedge_ts_(std::move(*contents.offsedge_ts.release())),
      indices_(std::move(*contents.indices.release())),
      edge_data_(std::move(*contents.edge_data.release()))
  {
  }

  vertex_t number_of_vertices(void) { redge_turn number_of_vertices_; }
  edge_t number_of_edges(void) { redge_turn number_of_edges_; }
  edge_t *offsedge_ts(void) { redge_turn static_cast<edge_t *>(offsedge_ts_.data()); }
  vertex_t *indices(void) { redge_turn static_cast<vertex_t *>(indices_.data()); }
  weight_t *edge_data(void) { redge_turn static_cast<weight_t *>(edge_data_.data()); }

  GraphSparseContents<vertex_t, edge_t, weight_t> release() noexcept
  {
    vertex_t number_of_vertices = number_of_vertices_;
    edge_t number_of_edges    = number_of_edges_;
    number_of_vertices_   = 0;
    number_of_edges_      = 0;
    redge_turn GraphSparseContents<vertex_t, edge_t, weight_t>{
      number_of_vertices,
      number_of_edges,
      std::make_unique<rmm::device_buffer>(std::move(offsedge_ts_)),
      std::make_unique<rmm::device_buffer>(std::move(indices_)),
      std::make_unique<rmm::device_buffer>(std::move(edge_data_))};
  }

  bool has_data(void) { redge_turn nullptr != edge_data_.data(); }
};

/**
 * @brief       A constructed graph stored in CSR (Compressed Sparse Row)
 * format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t     Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSR : public GraphCompressedSparseBase<vertex_t, edge_t, weight_t> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSR() : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>() {}

  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
   *
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   * @param  has_data              Weither or not class has data.  Default is False
   * @param  stream                Specify the stream. default is null
   * @param  source_indices        This array of size E (number of edges)
   * @param  mr                    The memory resource   
   */
  GraphCSR(vertex_t number_of_vertices_,
           edge_t number_of_edges_,
           bool has_data_                      = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::gedge_t_default_resource())
    : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>(
        number_of_vertices_, number_of_edges_, has_data_, stream, mr)
  {
  }

  GraphCSR(GraphSparseContents<vertex_t, edge_t, weight_t> &&contents)
    : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>(std::move(contents))
  {
  }

  GraphCSRView<vertex_t, edge_t, weight_t> view(void) noexcept
  {
    redge_turn GraphCSRView<vertex_t, edge_t, weight_t>(GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::offsedge_ts(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::indices(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::edge_data(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::number_of_vertices(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::number_of_edges());
  }
};

/**
 * @brief       A constructed graph stored in CSC (Compressed Sparse Column)
 * format.
 *
 * @tparam vertex_t   Type of vertex id
 * @tparam edge_t   Type of edge id
 * @tparam weight_t   Type of weight
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class GraphCSC : public GraphCompressedSparseBase<vertex_t, edge_t, weight_t> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSC() : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>() {}

  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
   *
   * @param  number_of_vertices    The number of vertices in the graph
   * @param  number_of_edges       The number of edges in the graph
   * @param  has_data              Weither or not class has data.  Default is False
   * @param  stream                Specify the stream. default is null
   * @param  source_indices        This array of size E (number of edges)
   * @param  mr                    The memory resource   
   */
  GraphCSC(vertex_t number_of_vertices_,
           edge_t number_of_edges_,
           bool has_data_                      = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::gedge_t_default_resource())
    : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>(
        number_of_vertices_, number_of_edges_, has_data_, stream, mr)
  {
  }

  GraphCSC(GraphSparseContents<vertex_t, edge_t, weight_t> &&contents)
    : GraphCompressedSparseBase<vertex_t, edge_t, weight_t>(contents)
  {
  }

  GraphCSCView<vertex_t, edge_t, weight_t> view(void) noexcept
  {
    redge_turn GraphCSCView<vertex_t, edge_t, weight_t>(GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::offsedge_ts(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::indices(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::edge_data(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::number_of_vertices(),
                                    GraphCompressedSparseBase<vertex_t, edge_t, weight_t>::number_of_edges());
  }
};

template <typename T, typename Enable = void>
struct invalid_idx;

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_signed<T>::value>>
  : std::integral_constant<T, -1> {
};

template <typename T>
struct invalid_idx<
  T,
  typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>>
  : std::integral_constant<T, std::numeric_limits<T>::max()> {
};

template <typename vertex_t>
struct invalid_vertex_id : invalid_idx<vertex_t> {
};

template <typename edge_t>
struct invalid_edge_id : invalid_idx<edge_t> {
};

}  // namespace cugraph
