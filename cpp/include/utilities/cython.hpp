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

#include <experimental/graph.hpp>
#include <graph.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace cython {

enum class numberTypeEnum : int { int32Type, int64Type, floatType, doubleType };

enum class graphTypeEnum : int {
  // represents unintiialized or NULL ptr
  null,
  // represents some legacy Cxx type. This and other LegacyCxx values are not
  // used for the unique_ptr in a graph_container_t, but instead for when this
  // enum is used for determining high-level code paths to take to prevent
  // needing to expose each legacy enum value to cython.
  LegacyCSR,
  LegacyCSC,
  LegacyCOO,
  // represents that a GraphCxxView* unique_ptr type is present in a
  // graph_container_t.
  GraphCSRViewFloat,
  GraphCSRViewDouble,
  GraphCSCViewFloat,
  GraphCSCViewDouble,
  GraphCOOViewFloat,
  GraphCOOViewDouble,
  // represents values present in the graph_container_t to construct a graph_t,
  // but unlike legacy classes does not mean a graph_t unique_ptr is present in
  // the container.
  graph_t,
};

// "container" for a graph type instance which insulates the owner from the
// specifics of the actual graph type. This is intended to be used in Cython
// code that only needs to pass a graph object to another wrapped C++ API. This
// greatly simplifies the Cython code since the Cython definition only needs to
// define the container and not the various individual graph types in Cython.
struct graph_container_t {
  // FIXME: This union is in place only to support legacy calls, remove when
  // migration to graph_t types is complete, or when legacy graph objects are
  // constructed in the call_<<algo> wrappers instead of the
  // populate_graph_container_legacy() function.
  union graphPtrUnion {
    ~graphPtrUnion() {}

    void* null;
    std::unique_ptr<GraphCSRView<int32_t, int32_t, float>> GraphCSRViewFloatPtr;
    std::unique_ptr<GraphCSRView<int32_t, int32_t, double>> GraphCSRViewDoublePtr;
    std::unique_ptr<GraphCSCView<int32_t, int32_t, float>> GraphCSCViewFloatPtr;
    std::unique_ptr<GraphCSCView<int32_t, int32_t, double>> GraphCSCViewDoublePtr;
    std::unique_ptr<GraphCOOView<int32_t, int32_t, float>> GraphCOOViewFloatPtr;
    std::unique_ptr<GraphCOOView<int32_t, int32_t, double>> GraphCOOViewDoublePtr;
  };

  graph_container_t() : graph_ptr_union{nullptr}, graph_type{graphTypeEnum::null} {}
  ~graph_container_t() {}

  // The expected usage of a graph_container_t is for it to be created as part
  // of a cython wrapper simply for passing a templated instantiation of a
  // particular graph class from one call to another, and not to exist outside
  // of the individual wrapper function (deleted when the instance goes out of
  // scope once the wrapper function returns). Therefore, copys and assignments
  // to an instance are not supported and these methods are deleted.
  graph_container_t(const graph_container_t&) = delete;
  graph_container_t& operator=(const graph_container_t&) = delete;

  graphPtrUnion graph_ptr_union;
  graphTypeEnum graph_type;

  // primitive data used for constructing graph_t instances.
  void* src_vertices;
  void* dst_vertices;
  void* weights;
  void* vertex_partition_offsets;

  size_t num_partition_edges;
  size_t num_global_vertices;
  size_t num_global_edges;
  numberTypeEnum vertexType;
  numberTypeEnum edgeType;
  numberTypeEnum weightType;
  bool transposed;
  bool is_multi_gpu;
  bool sorted_by_degree;
  bool do_expensive_check;
  bool hypergraph_partitioned;
  int row_comm_size;
  int col_comm_size;
  int row_comm_rank;
  int col_comm_rank;
  experimental::graph_properties_t graph_props;
};

// replacement for std::tuple<,,>, since std::tuple is not
// supported in cython
//
template <typename vertex_t, typename weight_t>
struct major_minor_weights_t {
  explicit major_minor_weights_t(raft::handle_t const& handle)
    : shuffled_major_vertices_(0, handle.get_stream()),
      shuffled_minor_vertices_(0, handle.get_stream()),
      shuffled_weights_(0, handle.get_stream())
  {
  }
  rmm::device_uvector<vertex_t>& get_major(void) { return shuffled_major_vertices_; }

  rmm::device_uvector<vertex_t>& get_minor(void) { return shuffled_minor_vertices_; }

  rmm::device_uvector<weight_t>& get_weights(void) { return shuffled_weights_; }

 private:
  rmm::device_uvector<vertex_t> shuffled_major_vertices_;
  rmm::device_uvector<vertex_t> shuffled_minor_vertices_;
  rmm::device_uvector<weight_t> shuffled_weights_;
};

// wrapper for renumber_edgelist() return
// (unrenumbering maps, etc.)
//
template <typename vertex_t, typename edge_t>
struct renum_quad_t {
  explicit renum_quad_t(raft::handle_t const& handle)
    : dv_(0, handle.get_stream()), part_(std::vector<vertex_t>(), false, 0, 0, 0, 0)
  {
  }

  rmm::device_uvector<vertex_t>& get_dv(void) { return dv_; }
  cugraph::experimental::partition_t<vertex_t>& get_partition(void)
  {
    return part_;
  }  // requires a `pass` Cython exposure, at least, to `partition_t`
  vertex_t& get_num_vertices(void) { return nv_; }
  edge_t& get_num_edges(void) { return ne_; }

 private:
  rmm::device_uvector<vertex_t> dv_;
  cugraph::experimental::partition_t<vertex_t> part_;
  vertex_t nv_;
  edge_t ne_;
};

// FIXME: finish description for vertex_partition_offsets
//
// Factory function for populating an empty graph container with a new graph
// object from basic types, and sets the corresponding meta-data. Args are:
//
// graph_container_t& graph_container
//   Reference to the graph_container_t instance to
//   populate. populate_graph_container() can only be called on an "empty"
//   container (ie. a container that has not been previously populated by
//   populate_graph_container())
//
// graphTypeEnum legacyType
//   Specifies the type of graph when instantiating a legacy graph type
//   (GraphCSRViewFloat, etc.).
//   NOTE: this parameter will be removed when the transition to exclusinve use
//   of the new 2D graph classes is complete.
//
// raft::handle_t const& handle
//   Raft handle to be set on the new graph instance in the container
//
// void* src_vertices, dst_vertices, weights
//   Pointer to an array of values representing source and destination vertices,
//   and edge weights respectively. The value types of the array are specified
//   using numberTypeEnum values separately (see below). offsets should be size
//   num_vertices+1, indices should be size num_edges, weights should also be
//   size num_edges
//
// void* vertex_partition_offsets
//   Pointer to an array of vertexType values representing offsets into the
//   individual partitions for a multi-GPU paritioned graph. The offsets are used for ...
//
// numberTypeEnum vertexType, edgeType, weightType
//   numberTypeEnum enum value describing the data type for the vertices,
//   offsets, and weights arrays respectively. These enum values are used to
//   instantiate the proper templated graph type and for casting the arrays
//   accordingly.
//
// int num_vertices, num_edges
//   The number of vertices and edges respectively in the graph represented by
//   the above arrays.
//
// bool transposed
//   true if the resulting graph object should store a transposed adjacency
//   matrix
//
// bool multi_gpu
//   true if the resulting graph object is to be used for a multi-gpu
//   application
void populate_graph_container(graph_container_t& graph_container,
                              raft::handle_t& handle,
                              void* src_vertices,
                              void* dst_vertices,
                              void* weights,
                              void* vertex_partition_offsets,
                              numberTypeEnum vertexType,
                              numberTypeEnum edgeType,
                              numberTypeEnum weightType,
                              size_t num_partition_edges,
                              size_t num_global_vertices,
                              size_t num_global_edges,
                              bool sorted_by_degree,
                              bool transposed,
                              bool multi_gpu);

// FIXME: comment this function
// FIXME: Should local_* values be void* as well?
void populate_graph_container_legacy(graph_container_t& graph_container,
                                     graphTypeEnum legacyType,
                                     raft::handle_t const& handle,
                                     void* offsets,
                                     void* indices,
                                     void* weights,
                                     numberTypeEnum offsetType,
                                     numberTypeEnum indexType,
                                     numberTypeEnum weightType,
                                     size_t num_global_vertices,
                                     size_t num_global_edges,
                                     int* local_vertices,
                                     int* local_edges,
                                     int* local_offsets);

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
std::pair<size_t, weight_t> call_louvain(raft::handle_t const& handle,
                                         graph_container_t const& graph_container,
                                         void* identifiers,
                                         void* parts,
                                         size_t max_level,
                                         weight_t resolution);

// Wrapper for calling Pagerank using a graph container
template <typename vertex_t, typename weight_t>
void call_pagerank(raft::handle_t const& handle,
                   graph_container_t const& graph_container,
                   vertex_t* identifiers,
                   weight_t* pagerank,
                   vertex_t personalization_subset_size,
                   vertex_t* personalization_subset,
                   weight_t* personalization_values,
                   double alpha,
                   double tolerance,
                   int64_t max_iter,
                   bool has_guess);

// Wrapper for calling Katz centrality using a graph container
template <typename vertex_t, typename weight_t>
void call_katz_centrality(raft::handle_t const& handle,
                          graph_container_t const& graph_container,
                          vertex_t* identifiers,
                          weight_t* katz_centrality,
                          double alpha,
                          double beta,
                          double tolerance,
                          int64_t max_iter,
                          bool normalized,
                          bool has_guess);

// Wrapper for calling BFS through a graph container
template <typename vertex_t, typename weight_t>
void call_bfs(raft::handle_t const& handle,
              graph_container_t const& graph_container,
              vertex_t* identifiers,
              vertex_t* distances,
              vertex_t* predecessors,
              double* sp_counters,
              const vertex_t start_vertex,
              bool directed);

// Wrapper for calling SSSP through a graph container
template <typename vertex_t, typename weight_t>
void call_sssp(raft::handle_t const& handle,
               graph_container_t const& graph_container,
               vertex_t* identifiers,
               weight_t* distances,
               vertex_t* predecessors,
               const vertex_t source_vertex);

// wrapper for shuffling:
//
template <typename vertex_t, typename edge_t, typename weight_t>
major_minor_weights_t<vertex_t, weight_t> call_shuffle(
  raft::handle_t const& handle,
  vertex_t* edgelist_major_vertices /* [IN] */,  // make_zip_iterator does not accept const !
  vertex_t* edgelist_minor_vertices /* [IN] */,
  weight_t* edegelist_weights,
  edge_t num_edgelist_edges,
  bool is_hypergraph_partitioned);  // = false

// Wrapper for calling renumber_edeglist() inplace:
//
template <typename vertex_t, typename edge_t>
renum_quad_t<vertex_t, edge_t> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_major_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool is_hypergraph_partitioned,
  bool do_expensive_check,
  bool multi_gpu);

// Helper for setting up subcommunicators, typically called as part of the
// user-initiated comms initialization in Python.
//
// raft::handle_t& handle
//   Raft handle for which the new subcommunicators will be created. The
//   subcommunicators will then be accessible from the handle passed to the
//   parallel processes.
//
// size_t row_comm_size
//   Number of items in a partition row (ie. pcols), needed for creating the
//   appropriate number of subcommunicator instances.
void init_subcomms(raft::handle_t& handle, size_t row_comm_size);

}  // namespace cython
}  // namespace cugraph
