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

#include <experimental/graph_view.hpp>
#include <graph.hpp>
#include <raft/handle.hpp>

namespace cugraph {
namespace cython {

enum class numberTypeEnum : int { intType, floatType, doubleType };

// FIXME: The GraphC??View* types will not be used in the near future. Those are
// left in place as cython wrappers transition from the GraphC* classes to
// graph_* classes. Remove GraphC* classes once the transition is complete.
enum class graphTypeEnum : int {
  null,
  GraphCSRViewFloat,
  GraphCSRViewDouble,
  GraphCSCViewFloat,
  GraphCSCViewDouble,
  GraphCOOViewFloat,
  GraphCOOViewDouble,
  graph_view_t_float,
  graph_view_t_double,
  graph_view_t_float_mg,
  graph_view_t_double_mg,
  graph_view_t_float_transposed,
  graph_view_t_double_transposed,
  graph_view_t_float_mg_transposed,
  graph_view_t_double_mg_transposed
};

// Enum for the high-level type of GraphC??View* class to instantiate.
enum class legacyGraphTypeEnum : int { CSR, CSC, COO };

// "container" for a graph type instance which insulates the owner from the
// specifics of the actual graph type. This is intended to be used in Cython
// code that only needs to pass a graph object to another wrapped C++ API. This
// greatly simplifies the Cython code since the Cython definition only needs to
// define the container and not the various individual graph types in Cython.
struct graph_container_t {
  // FIXME: use std::variant (or a better alternative, ie. type erasure?) instead
  //        of a union if possible
  union graphPtrUnion {
    ~graphPtrUnion() {}

    void* null;
    std::unique_ptr<GraphCSRView<int, int, float>> GraphCSRViewFloatPtr;
    std::unique_ptr<GraphCSRView<int, int, double>> GraphCSRViewDoublePtr;
    std::unique_ptr<GraphCSCView<int, int, float>> GraphCSCViewFloatPtr;
    std::unique_ptr<GraphCSCView<int, int, double>> GraphCSCViewDoublePtr;
    std::unique_ptr<GraphCOOView<int, int, float>> GraphCOOViewFloatPtr;
    std::unique_ptr<GraphCOOView<int, int, double>> GraphCOOViewDoublePtr;
    std::unique_ptr<experimental::graph_view_t<int, int, float, false, false>>
      graph_view_t_float_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, double, false, false>>
      graph_view_t_double_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, float, false, true>>
      graph_view_t_float_mg_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, double, false, true>>
      graph_view_t_double_mg_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, float, true, false>>
      graph_view_t_float_transposed_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, double, true, false>>
      graph_view_t_double_transposed_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, float, true, true>>
      graph_view_t_float_mg_transposed_ptr;
    std::unique_ptr<experimental::graph_view_t<int, int, double, true, true>>
      graph_view_t_double_mg_transposed_ptr;
  };

  graph_container_t() : graph_ptr_union{nullptr}, graph_ptr_type{graphTypeEnum::null} {}
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
  graphTypeEnum graph_ptr_type;
};

// Factory function for populating an empty graph container with a new graph
// object from basic types, and sets the corresponding meta-data. Args are:
//
// graph_container_t& graph_container
//   Reference to the graph_container_t instance to
//   populate. populate_graph_container() can only be called on an "empty"
//   container (ie. a container that has not been previously populated by
//   populate_graph_container())
//
// legacyGraphTypeEnum legacyType
//   Specifies the type of graph when instantiating a legacy graph type
//   (GraphCSRViewFloat, etc.).
//   NOTE: this parameter will be removed when the transition to exclusinve use
//   of the new 2D graph classes is complete.
//
// raft::handle_t const& handle
//   Raft handle to be set on the new graph instance in the container
//
// void* offsets, indices, weights
//   Pointer to an array of values representing offsets, indices, and weights
//   respectively. The value types of the array are specified using
//   numberTypeEnum values separately (see below). offsets should be size
//   num_vertices+1, indices should be size num_edges, weights should also be
//   size num_edges
//
// numberTypeEnum offsetType, indexType, weightType
//   numberTypeEnum enum value describing the data type for the offsets,
//   indices, and weights arrays respectively. These enum values are used to
//   instantiate the proper templated graph type and for casting the arrays
//   accordingly.
//
// int num_vertices, num_edges
//   The number of vertices and edges respectively in the graph represented by
//   the above arrays.
//
// int* local_vertices, local_edges
//   Arrays containing the number of vertices and number of edges,
//   respectively. For example, if there are a total of 7 vertices, 16 edges,
//   and the algorithm is distributed over 3 GPUs, the local_vertices may contain
//   [2,2,3] and local_edges may contain [5,5,6].
//   NOTE: these parameters are only needed for legacy GraphC??View* classes and
//   may not be present in future versions.
//
// int* local_offsets
//   Array containing the offsets between the local_* arrays and those for the
//   global graph, allowing the array to start at position zero yet still be
//   mapped to a position in the global array.
//   NOTE: this parameter is only needed for legacy GraphC??View* classes and
//   may not be present in future versions.
//
// bool transposed
//   true if the resulting graph object should store a transposed adjacency
//   matrix
//
// bool multi_gpu
//   true if the resulting graph object is to be used for a multi-gpu
//   application
//
// FIXME: Should local_* values be void* as well?
void populate_graph_container(graph_container_t& graph_container,
                              legacyGraphTypeEnum legacyType,
                              raft::handle_t const& handle,
                              void* offsets,
                              void* indices,
                              void* weights,
                              numberTypeEnum offsetType,
                              numberTypeEnum indexType,
                              numberTypeEnum weightType,
                              int num_vertices,
                              int num_edges,
                              int* local_vertices,
                              int* local_edges,
                              int* local_offsets,
                              bool transposed,
                              bool multi_gpu);

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
weight_t call_louvain(raft::handle_t const& handle,
                      graph_container_t const& graph_container,
                      void* parts,
                      size_t max_level,
                      weight_t resolution);

}  // namespace cython
}  // namespace cugraph
