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
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <raft/handle.hpp>
#include <rmm/device_buffer.hpp>

namespace cugraph {
namespace experimental {

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
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphViewBase {
 public:
  WT *edge_data;  ///< edge weight
  raft::handle_t *handle;
  GraphProperties prop;

  VT number_of_vertices;
  ET number_of_edges;

  /**
   * @brief      Fill the identifiers array with the vertex identifiers.
   *
   * @param[out]    identifier      Pointer to device memory to store the vertex
   * identifiers
   */
  void get_vertex_identifiers(VT *identifiers) const;
  void set_handle(raft::handle_t *handle_) { handle = handle_; }
  GraphViewBase(WT *edge_data_, VT number_of_vertices_, ET number_of_edges_)
    : edge_data(edge_data_),
      prop(),
      number_of_vertices(number_of_vertices_),
      number_of_edges(number_of_edges_)
  {
    handle = new raft::handle_t;
  }
  bool has_data(void) const { return edge_data != nullptr; }
};

/**
 * @brief       A graph stored in COO (COOrdinate) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCOOView : public GraphViewBase<VT, ET, WT> {
 public:
  VT *src_indices{nullptr};  ///< rowInd
  VT *dst_indices{nullptr};  ///< colInd

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
  void degree(ET *degree, DegreeDirection direction) const;

  /**
   * @brief      Default constructor
   */
  GraphCOOView() : GraphViewBase<VT, ET, WT>(nullptr, 0, 0) {}

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
    VT *src_indices_, VT *dst_indices_, WT *edge_data_, VT number_of_vertices_, ET number_of_edges_)
    : GraphViewBase<VT, ET, WT>(edge_data_, number_of_vertices_, number_of_edges_),
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
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCompressedSparseBaseView : public GraphViewBase<VT, ET, WT> {
 public:
  ET *offsets{nullptr};  ///< CSR offsets
  VT *indices{nullptr};  ///< CSR indices

  /**
   * @brief      Fill the identifiers in the array with the source vertex
   * identifiers
   *
   * @param[out]    src_indices      Pointer to device memory to store the
   * source vertex identifiers
   */
  void get_source_indices(VT *src_indices) const;

  /**
   * @brief     Computes degree(in, out, in+out) of all the nodes of a Graph
   *
   * @throws     cugraph::logic_error when an error occurs.
   *
   * @param[out] degree                Device array of size V (V is number of
   * vertices) initialized
   * to zeros. Will contain the computed degree of every vertex.
   * @param[in]  x                     Integer value indicating type of degree
   * calculation
   *                                      0 : in+out degree
   *                                      1 : in-degree
   *                                      2 : out-degree
   */
  void degree(ET *degree, DegreeDirection direction) const;

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
    ET *offsets_, VT *indices_, WT *edge_data_, VT number_of_vertices_, ET number_of_edges_)
    : GraphViewBase<VT, ET, WT>(edge_data_, number_of_vertices_, number_of_edges_),
      offsets{offsets_},
      indices{indices_}
  {
  }
};

/**
 * @brief       A graph stored in CSR (Compressed Sparse Row) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSRView : public GraphCompressedSparseBaseView<VT, ET, WT> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSRView() : GraphCompressedSparseBaseView<VT, ET, WT>(nullptr, nullptr, nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing adjacency lists in a Graph.
   *             GraphCSRView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
    ET *offsets_, VT *indices_, WT *edge_data_, VT number_of_vertices_, ET number_of_edges_)
    : GraphCompressedSparseBaseView<VT, ET, WT>(
        offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {
  }
};

/**
 * @brief       A graph stored in CSC (Compressed Sparse Column) format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSCView : public GraphCompressedSparseBaseView<VT, ET, WT> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSCView() : GraphCompressedSparseBaseView<VT, ET, WT>(nullptr, nullptr, nullptr, 0, 0) {}

  /**
   * @brief      Wrap existing arrays representing transposed adjacency lists in
   * a Graph.
   *             GraphCSCView does not own the memory used to represent this
   * graph. This
   *             function does not allocate memory.
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
    ET *offsets_, VT *indices_, WT *edge_data_, VT number_of_vertices_, ET number_of_edges_)
    : GraphCompressedSparseBaseView<VT, ET, WT>(
        offsets_, indices_, edge_data_, number_of_vertices_, number_of_edges_)
  {
  }
};

/**
 * @brief      TODO : Change this Take ownership of the provided graph arrays in
 * COO format
 *
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
 * @param  number_of_vertices    The number of vertices in the graph
 * @param  number_of_edges       The number of edges in the graph
 */
template <typename VT, typename ET, typename WT>
struct GraphCOOContents {
  VT number_of_vertices;
  ET number_of_edges;
  std::unique_ptr<rmm::device_buffer> src_indices;
  std::unique_ptr<rmm::device_buffer> dst_indices;
  std::unique_ptr<rmm::device_buffer> edge_data;
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
  rmm::device_buffer src_indices_{};  ///< rowInd
  rmm::device_buffer dst_indices_{};  ///< colInd
  rmm::device_buffer edge_data_{};    ///< CSR data

 public:
  /**
   * @brief      Take ownership of the provided graph arrays in COO format
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
  GraphCOO(VT number_of_vertices,
           ET number_of_edges,
           bool has_data                       = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
    : number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      src_indices_(sizeof(VT) * number_of_edges, stream, mr),
      dst_indices_(sizeof(VT) * number_of_edges, stream, mr),
      edge_data_((has_data ? sizeof(WT) * number_of_edges : 0), stream, mr)
  {
  }

  GraphCOO(GraphCOOView<VT, ET, WT> const &graph,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
    : number_of_vertices_(graph.number_of_vertices),
      number_of_edges_(graph.number_of_edges),
      src_indices_(graph.src_indices, graph.number_of_edges * sizeof(VT), stream, mr),
      dst_indices_(graph.dst_indices, graph.number_of_edges * sizeof(VT), stream, mr)
  {
    if (graph.has_data()) {
      edge_data_ =
        rmm::device_buffer{graph.edge_data, graph.number_of_edges * sizeof(WT), stream, mr};
    }
  }

  VT number_of_vertices(void) { return number_of_vertices_; }
  ET number_of_edges(void) { return number_of_edges_; }
  VT *src_indices(void) { return static_cast<VT *>(src_indices_.data()); }
  VT *dst_indices(void) { return static_cast<VT *>(dst_indices_.data()); }
  WT *edge_data(void) { return static_cast<WT *>(edge_data_.data()); }

  GraphCOOContents<VT, ET, WT> release() noexcept
  {
    VT number_of_vertices = number_of_vertices_;
    ET number_of_edges    = number_of_edges_;
    number_of_vertices_   = 0;
    number_of_edges_      = 0;
    return GraphCOOContents<VT, ET, WT>{
      number_of_vertices,
      number_of_edges,
      std::make_unique<rmm::device_buffer>(std::move(src_indices_)),
      std::make_unique<rmm::device_buffer>(std::move(dst_indices_)),
      std::make_unique<rmm::device_buffer>(std::move(edge_data_))};
  }

  GraphCOOView<VT, ET, WT> view(void) noexcept
  {
    return GraphCOOView<VT, ET, WT>(
      src_indices(), dst_indices(), edge_data(), number_of_vertices_, number_of_edges_);
  }

  bool has_data(void) { return nullptr != edge_data_.data(); }
};

template <typename VT, typename ET, typename WT>
struct GraphSparseContents {
  VT number_of_vertices;
  ET number_of_edges;
  std::unique_ptr<rmm::device_buffer> offsets;
  std::unique_ptr<rmm::device_buffer> indices;
  std::unique_ptr<rmm::device_buffer> edge_data;
};

/**
 * @brief       Base class for constructted graphs stored in CSR (Compressed
 * Sparse Row) format or
 * CSC (Compressed Sparse Column) format
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCompressedSparseBase {
  VT number_of_vertices_{0};
  ET number_of_edges_{0};
  rmm::device_buffer offsets_{};    ///< CSR offsets
  rmm::device_buffer indices_{};    ///< CSR indices
  rmm::device_buffer edge_data_{};  ///< CSR data

  bool has_data_{false};

 public:
  /**
   * @brief      Take ownership of the provided graph arrays in CSR/CSC format
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
  GraphCompressedSparseBase(VT number_of_vertices,
                            ET number_of_edges,
                            bool has_data,
                            cudaStream_t stream,
                            rmm::mr::device_memory_resource *mr)
    : number_of_vertices_(number_of_vertices),
      number_of_edges_(number_of_edges),
      offsets_(sizeof(ET) * (number_of_vertices + 1), stream, mr),
      indices_(sizeof(VT) * number_of_edges, stream, mr),
      edge_data_((has_data ? sizeof(WT) * number_of_edges : 0), stream, mr)
  {
  }

  GraphCompressedSparseBase(GraphSparseContents<VT, ET, WT> &&contents)
    : number_of_vertices_(contents.number_of_vertices),
      number_of_edges_(contents.number_of_edges),
      offsets_(std::move(*contents.offsets.release())),
      indices_(std::move(*contents.indices.release())),
      edge_data_(std::move(*contents.edge_data.release()))
  {
  }

  VT number_of_vertices(void) { return number_of_vertices_; }
  ET number_of_edges(void) { return number_of_edges_; }
  ET *offsets(void) { return static_cast<ET *>(offsets_.data()); }
  VT *indices(void) { return static_cast<VT *>(indices_.data()); }
  WT *edge_data(void) { return static_cast<WT *>(edge_data_.data()); }

  GraphSparseContents<VT, ET, WT> release() noexcept
  {
    VT number_of_vertices = number_of_vertices_;
    ET number_of_edges    = number_of_edges_;
    number_of_vertices_   = 0;
    number_of_edges_      = 0;
    return GraphSparseContents<VT, ET, WT>{
      number_of_vertices,
      number_of_edges,
      std::make_unique<rmm::device_buffer>(std::move(offsets_)),
      std::make_unique<rmm::device_buffer>(std::move(indices_)),
      std::make_unique<rmm::device_buffer>(std::move(edge_data_))};
  }

  bool has_data(void) { return nullptr != edge_data_.data(); }
};

/**
 * @brief       A constructed graph stored in CSR (Compressed Sparse Row)
 * format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSR : public GraphCompressedSparseBase<VT, ET, WT> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSR() : GraphCompressedSparseBase<VT, ET, WT>() {}

  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
  GraphCSR(VT number_of_vertices_,
           ET number_of_edges_,
           bool has_data_                      = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
    : GraphCompressedSparseBase<VT, ET, WT>(
        number_of_vertices_, number_of_edges_, has_data_, stream, mr)
  {
  }

  GraphCSR(GraphSparseContents<VT, ET, WT> &&contents)
    : GraphCompressedSparseBase<VT, ET, WT>(std::move(contents))
  {
  }

  GraphCSRView<VT, ET, WT> view(void) noexcept
  {
    return GraphCSRView<VT, ET, WT>(GraphCompressedSparseBase<VT, ET, WT>::offsets(),
                                    GraphCompressedSparseBase<VT, ET, WT>::indices(),
                                    GraphCompressedSparseBase<VT, ET, WT>::edge_data(),
                                    GraphCompressedSparseBase<VT, ET, WT>::number_of_vertices(),
                                    GraphCompressedSparseBase<VT, ET, WT>::number_of_edges());
  }
};

/**
 * @brief       A constructed graph stored in CSC (Compressed Sparse Column)
 * format.
 *
 * @tparam VT   Type of vertex id
 * @tparam ET   Type of edge id
 * @tparam WT   Type of weight
 */
template <typename VT, typename ET, typename WT>
class GraphCSC : public GraphCompressedSparseBase<VT, ET, WT> {
 public:
  /**
   * @brief      Default constructor
   */
  GraphCSC() : GraphCompressedSparseBase<VT, ET, WT>() {}

  /**
   * @brief      Take ownership of the provided graph arrays in CSR format
   *
   * @param  offsets               This array of size V+1 (V is number of
   * vertices) contains the
   * offset of adjacency lists of every vertex. Offsets must be in the range [0,
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
  GraphCSC(VT number_of_vertices_,
           ET number_of_edges_,
           bool has_data_                      = false,
           cudaStream_t stream                 = nullptr,
           rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
    : GraphCompressedSparseBase<VT, ET, WT>(
        number_of_vertices_, number_of_edges_, has_data_, stream, mr)
  {
  }

  GraphCSC(GraphSparseContents<VT, ET, WT> &&contents)
    : GraphCompressedSparseBase<VT, ET, WT>(contents)
  {
  }

  GraphCSCView<VT, ET, WT> view(void) noexcept
  {
    return GraphCSCView<VT, ET, WT>(GraphCompressedSparseBase<VT, ET, WT>::offsets(),
                                    GraphCompressedSparseBase<VT, ET, WT>::indices(),
                                    GraphCompressedSparseBase<VT, ET, WT>::edge_data(),
                                    GraphCompressedSparseBase<VT, ET, WT>::number_of_vertices(),
                                    GraphCompressedSparseBase<VT, ET, WT>::number_of_edges());
  }
};

}  // namespace experimental
}  // namespace cugraph
