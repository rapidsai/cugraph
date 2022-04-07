/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/graph_generators.hpp>
#include <cugraph/legacy/graph.hpp>
#include <cugraph/utilities/graph_traits.hpp>

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
    std::unique_ptr<legacy::GraphCSRView<int32_t, int32_t, float>> GraphCSRViewFloatPtr;
    std::unique_ptr<legacy::GraphCSRView<int32_t, int32_t, double>> GraphCSRViewDoublePtr;
    std::unique_ptr<legacy::GraphCSCView<int32_t, int32_t, float>> GraphCSCViewFloatPtr;
    std::unique_ptr<legacy::GraphCSCView<int32_t, int32_t, double>> GraphCSCViewDoublePtr;
    std::unique_ptr<legacy::GraphCOOView<int32_t, int32_t, float>> GraphCOOViewFloatPtr;
    std::unique_ptr<legacy::GraphCOOView<int32_t, int32_t, double>> GraphCOOViewDoublePtr;
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
  bool is_weighted;
  void* vertex_partition_offsets;
  void* segment_offsets;
  size_t num_segments;

  size_t num_local_edges;
  size_t num_global_vertices;
  size_t num_global_edges;
  numberTypeEnum vertexType;
  numberTypeEnum edgeType;
  numberTypeEnum weightType;
  bool transposed;
  bool is_multi_gpu;
  bool do_expensive_check;
  int row_comm_size;
  int col_comm_size;
  int row_comm_rank;
  int col_comm_rank;
  graph_properties_t graph_props;
};

/**
 * @brief     Owning struct. Allows returning multiple edge lists and edge offsets.
 *            cython only
 *
 * @param  number_of_vertices    The total number of vertices
 * @param  number_of_edges       The total number of edges (number of elements in src_indices,
 dst_indices and edge_data)
 * @param  number_of_subgraph    The number of subgraphs, number of elements in subgraph_offsets - 1
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
 * @param  subgraph_offsets            This array size number_of_subgraph + 1 contains edge offsets
 for each subgraph


 */
struct cy_multi_edgelists_t {
  size_t number_of_vertices;
  size_t number_of_edges;
  size_t number_of_subgraph;
  std::unique_ptr<rmm::device_buffer> src_indices;
  std::unique_ptr<rmm::device_buffer> dst_indices;
  std::unique_ptr<rmm::device_buffer> edge_data;
  std::unique_ptr<rmm::device_buffer> subgraph_offsets;
};

// replacement for std::tuple<,,>, since std::tuple is not
// supported in cython
//
template <typename vertex_t, typename edge_t, typename weight_t>
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

  std::vector<edge_t>& get_edge_counts(void) { return edge_counts_; }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_major_wrap(
    void)  // const: triggers errors in Cython autogen-ed C++
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_major_vertices_.release()),
                          sizeof(vertex_t));
  }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_minor_wrap(void)  // const
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_minor_vertices_.release()),
                          sizeof(vertex_t));
  }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_weights_wrap(void)  // const
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(shuffled_weights_.release()),
                          sizeof(weight_t));
  }

  std::unique_ptr<std::vector<edge_t>> get_edge_counts_wrap(void)  // const
  {
    return std::make_unique<std::vector<edge_t>>(edge_counts_);
  }

 private:
  rmm::device_uvector<vertex_t> shuffled_major_vertices_;
  rmm::device_uvector<vertex_t> shuffled_minor_vertices_;
  rmm::device_uvector<weight_t> shuffled_weights_;
  std::vector<edge_t> edge_counts_{};
};

// aggregate for random_walks() return type
// to be exposed to cython:
//
struct random_walk_ret_t {
  size_t coalesced_sz_v_;
  size_t coalesced_sz_w_;
  size_t num_paths_;
  size_t max_depth_;
  std::unique_ptr<rmm::device_buffer> d_coalesced_v_;
  std::unique_ptr<rmm::device_buffer> d_coalesced_w_;
  std::unique_ptr<rmm::device_buffer> d_sizes_;
};

struct random_walk_path_t {
  std::unique_ptr<rmm::device_buffer> d_v_offsets;
  std::unique_ptr<rmm::device_buffer> d_w_sizes;
  std::unique_ptr<rmm::device_buffer> d_w_offsets;
};

struct graph_generator_t {
  std::unique_ptr<rmm::device_buffer> d_source;
  std::unique_ptr<rmm::device_buffer> d_destination;
};

// enum class generator_distribution_t { POWER_LAW = 0, UNIFORM };
// aggregate for random_walks() COO return type
// to be exposed to cython:
//
struct random_walk_coo_t {
  size_t num_edges_;    // total number of COO triplets (for all paths)
  size_t num_offsets_;  // offsets of where each COO set starts for each path;
                        // NOTE: this can differ than num_paths_,
                        // because paths with 0 edges (one vertex)
                        // don't participate to the COO

  std::unique_ptr<rmm::device_buffer>
    d_src_;  // coalesced set of COO source vertices; |d_src_| = num_edges_
  std::unique_ptr<rmm::device_buffer>
    d_dst_;  // coalesced set of COO destination vertices; |d_dst_| = num_edges_
  std::unique_ptr<rmm::device_buffer>
    d_weights_;  // coalesced set of COO edge weights; |d_weights_| = num_edges_
  std::unique_ptr<rmm::device_buffer>
    d_offsets_;  // offsets where each COO subset for each path starts; |d_offsets_| = num_offsets_
};

// wrapper for renumber_edgelist() return
// (unrenumbering maps, etc.)
//
template <typename vertex_t, typename edge_t>
struct renum_tuple_t {
  explicit renum_tuple_t(raft::handle_t const& handle) : dv_(0, handle.get_stream()), part_() {}

  rmm::device_uvector<vertex_t>& get_dv(void) { return dv_; }

  std::pair<std::unique_ptr<rmm::device_buffer>, size_t> get_dv_wrap(
    void)  // const: see above explanation
  {
    return std::make_pair(std::make_unique<rmm::device_buffer>(dv_.release()), sizeof(vertex_t));
  }

  cugraph::partition_t<vertex_t>& get_partition(void) { return part_; }
  vertex_t& get_num_vertices(void) { return nv_; }
  edge_t& get_num_edges(void) { return ne_; }

  std::vector<vertex_t>& get_segment_offsets(void) { return segment_offsets_; }

  std::unique_ptr<std::vector<vertex_t>> get_segment_offsets_wrap()
  {  // const
    return std::make_unique<std::vector<vertex_t>>(segment_offsets_);
  }

  // `partition_t` pass-through getters
  //
  int get_part_row_size() const { return part_.row_comm_size(); }

  int get_part_col_size() const { return part_.col_comm_size(); }

  int get_part_comm_rank() const { return part_.comm_rank(); }

  // FIXME: part_.vertex_partition_offsets() returns a std::vector
  //
  std::unique_ptr<std::vector<vertex_t>> get_partition_offsets_wrap(void)  // const
  {
    return std::make_unique<std::vector<vertex_t>>(part_.vertex_partition_offsets());
  }

  std::pair<vertex_t, vertex_t> get_part_local_vertex_range() const
  {
    auto tpl_v = part_.local_vertex_partition_range();
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_local_vertex_first() const
  {
    return part_.local_vertex_partition_range_first();
  }

  vertex_t get_part_local_vertex_last() const { return part_.local_vertex_partition_range_last(); }

  std::pair<vertex_t, vertex_t> get_part_vertex_partition_range(size_t vertex_partition_idx) const
  {
    auto tpl_v = part_.vertex_partition_range(vertex_partition_idx);
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_vertex_partition_first(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_first(vertex_partition_idx);
  }

  vertex_t get_part_vertex_partition_last(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_last(vertex_partition_idx);
  }

  vertex_t get_part_vertex_partition_size(size_t vertex_partition_idx) const
  {
    return part_.vertex_partition_range_size(vertex_partition_idx);
  }

  size_t get_part_number_of_matrix_partitions() const
  {
    return part_.number_of_local_edgex_partitions();
  }

  std::pair<vertex_t, vertex_t> get_part_matrix_partition_major_range(size_t partition_idx) const
  {
    auto tpl_v = part_.local_edgex_partition_major_range(partition_idx);
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_matrix_partition_major_first(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_first(partition_idx);
  }

  vertex_t get_part_matrix_partition_major_last(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_range_last(partition_idx);
  }

  vertex_t get_part_matrix_partition_major_value_start_offset(size_t partition_idx) const
  {
    return part_.local_edge_partition_major_value_start_offset(partition_idx);
  }

  std::pair<vertex_t, vertex_t> get_part_matrix_partition_minor_range() const
  {
    auto tpl_v = part_.local_edge_partition_minor_range();
    return std::make_pair(std::get<0>(tpl_v), std::get<1>(tpl_v));
  }

  vertex_t get_part_matrix_partition_minor_first() const
  {
    return part_.local_edge_partition_minor_range_first();
  }

  vertex_t get_part_matrix_partition_minor_last() const
  {
    return part_.local_edge_partition_minor_range_last();
  }

 private:
  rmm::device_uvector<vertex_t> dv_;
  cugraph::partition_t<vertex_t> part_;
  vertex_t nv_{0};
  edge_t ne_{0};
  std::vector<vertex_t> segment_offsets_;
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
// bool is_weighted
//   true if the resulting graph object should store edge weights
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
                              void* segment_offsets,
                              size_t num_segments,
                              numberTypeEnum vertexType,
                              numberTypeEnum edgeType,
                              numberTypeEnum weightType,
                              size_t num_local_edges,
                              size_t num_global_vertices,
                              size_t num_global_edges,
                              bool is_weighted,
                              bool is_symmetric,
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
              vertex_t depth_limit,
              vertex_t* sources,
              size_t n_sources,
              bool direction_optimizing);

// Wrapper for calling SSSP through a graph container
template <typename vertex_t, typename weight_t>
void call_sssp(raft::handle_t const& handle,
               graph_container_t const& graph_container,
               vertex_t* identifiers,
               weight_t* distances,
               vertex_t* predecessors,
               const vertex_t source_vertex);

// Wrapper for calling egonet through a graph container
template <typename vertex_t, typename weight_t>
std::unique_ptr<cy_multi_edgelists_t> call_egonet(raft::handle_t const& handle,
                                                  graph_container_t const& graph_container,
                                                  vertex_t* source_vertex,
                                                  vertex_t n_subgraphs,
                                                  vertex_t radius);

// Wrapper for calling WCC through a graph container
template <typename vertex_t, typename weight_t>
void call_wcc(raft::handle_t const& handle,
              graph_container_t const& graph_container,
              vertex_t* components);

// Wrapper for calling HITS through a graph container
template <typename vertex_t, typename weight_t>
void call_hits(raft::handle_t const& handle,
               graph_container_t const& graph_container,
               weight_t* hubs,
               weight_t* authorities,
               size_t max_iter,
               weight_t tolerance,
               const weight_t* starting_value,
               bool normalized);

// Wrapper for calling graph generator
template <typename vertex_t>
std::unique_ptr<graph_generator_t> call_generate_rmat_edgelist(raft::handle_t const& handle,
                                                               size_t scale,
                                                               size_t num_edges,
                                                               double a,
                                                               double b,
                                                               double c,
                                                               uint64_t seed,
                                                               bool clip_and_flip,
                                                               bool scramble_vertex_ids);
template <typename vertex_t>
std::vector<std::pair<std::unique_ptr<rmm::device_buffer>, std::unique_ptr<rmm::device_buffer>>>
call_generate_rmat_edgelists(raft::handle_t const& handle,
                             size_t n_edgelists,
                             size_t min_scale,
                             size_t max_scale,
                             size_t edge_factor,
                             cugraph::generator_distribution_t size_distribution,
                             cugraph::generator_distribution_t edge_distribution,
                             uint64_t seed,
                             bool clip_and_flip,
                             bool scramble_vertex_ids);

// wrapper for random_walks.
//
template <typename vertex_t, typename edge_t>
std::enable_if_t<cugraph::is_vertex_edge_combo<vertex_t, edge_t>::value,
                 std::unique_ptr<random_walk_ret_t>>
call_random_walks(raft::handle_t const& handle,
                  graph_container_t const& graph_container,
                  vertex_t const* ptr_start_set,
                  edge_t num_paths,
                  edge_t max_depth,
                  bool use_padding);

template <typename index_t>
std::unique_ptr<random_walk_path_t> call_rw_paths(raft::handle_t const& handle,
                                                  index_t num_paths,
                                                  index_t const* vertex_path_sizes);

// convertor from random_walks return type to COO:
//
template <typename vertex_t, typename index_t>
std::unique_ptr<random_walk_coo_t> random_walks_to_coo(raft::handle_t const& handle,
                                                       random_walk_ret_t& rw_ret);

// wrapper for shuffling:
//
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<major_minor_weights_t<vertex_t, edge_t, weight_t>> call_shuffle(
  raft::handle_t const& handle,
  vertex_t*
    edgelist_major_vertices,  // [IN / OUT]: groupby_gpu_id_and_shuffle_values() sorts in-place
  vertex_t* edgelist_minor_vertices,  // [IN / OUT]
  weight_t* edgelist_weights,         // [IN / OUT]
  edge_t num_edgelist_edges);

// Wrapper for calling renumber_edeglist() inplace:
//
template <typename vertex_t, typename edge_t>
std::unique_ptr<renum_tuple_t<vertex_t, edge_t>> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_src_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_dst_vertices /* [INOUT] */,
  std::vector<edge_t> const& edge_counts,
  bool store_transposed,
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
