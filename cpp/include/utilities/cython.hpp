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

#include <graph.hpp>
#include <experimental/graph_view.hpp>
#include <raft/handle.hpp>

namespace cugraph {
namespace cython {

// FIXME: use std::variant (or a better alternative, ie. type erasure?) instead
//        of a union if possible
union graphPtrUnion {
  void* null;
  GraphCSRView<int, int, float>* GraphCSRViewFloatPtr;
  GraphCSRView<int, int, double>* GraphCSRViewDoublePtr;
  experimental::graph_view_t<int, int, float, false, false>* graph_view_t_float_ptr;
  experimental::graph_view_t<int, int, double, false, false>* graph_view_t_double_ptr;
  experimental::graph_view_t<int, int, float, false, true>* graph_view_t_float_mg_ptr;
  experimental::graph_view_t<int, int, double, false, true>* graph_view_t_double_mg_ptr;
  experimental::graph_view_t<int, int, float, true, false>* graph_view_t_float_transposed_ptr;
  experimental::graph_view_t<int, int, double, true, false>* graph_view_t_double_transposed_ptr;
  experimental::graph_view_t<int, int, float, true, true>* graph_view_t_float_mg_transposed_ptr;
  experimental::graph_view_t<int, int, double, true, true>* graph_view_t_double_mg_transposed_ptr;
};

enum class numberTypeEnum : int { intType,
                                  floatType,
                                  doubleType
};
enum class graphTypeEnum : int { null,
                                 GraphCSRViewFloat,
                                 GraphCSRViewDouble,
                                 graph_view_t_float,
                                 graph_view_t_double,
                                 graph_view_t_float_mg,
                                 graph_view_t_double_mg,
                                 graph_view_t_float_transposed,
                                 graph_view_t_double_transposed,
                                 graph_view_t_float_mg_transposed,
                                 graph_view_t_double_mg_transposed
};

// "container" for a graph type instance which insulates the owner from the
// specifics of the actual graph type. This is intended to be used in Cython
// code that only needs to pass a graph object to another wrapped C++ API. This
// simplifies the Cython code greatly since it only needs to define the
// container and not the various individual graph types in Cython.
struct graph_container_t {
  inline graph_container_t() :
     graph_ptr{nullptr},
     graph_ptr_type{graphTypeEnum::null} {}
   /*
  inline ~graph_container_t() {
    switch(graph_ptr_type) {
      case graphTypeEnum::GraphCSRViewFloat :
        delete graph_ptr.GraphCSRViewFloatPtr;
        std::cout << "DELETED GraphCSRViewFloatPtr" << std::endl;
        break;
      case graphTypeEnum::GraphCSRViewDouble :
        delete graph_ptr.GraphCSRViewDoublePtr;
        break;
      case graphTypeEnum::graph_view_t_float :
        delete graph_ptr.graph_view_t_float_ptr;
        break;
      case graphTypeEnum::graph_view_t_double :
        delete graph_ptr.graph_view_t_double_ptr;
        break;
      case graphTypeEnum::graph_view_t_float_mg :
        delete graph_ptr.graph_view_t_float_mg_ptr;
        break;
      case graphTypeEnum::graph_view_t_double_mg :
        delete graph_ptr.graph_view_t_double_mg_ptr;
        break;
      case graphTypeEnum::graph_view_t_float_transposed :
        delete graph_ptr.graph_view_t_float_transposed_ptr;
        break;
      case graphTypeEnum::graph_view_t_double_transposed :
        delete graph_ptr.graph_view_t_double_transposed_ptr;
        break;
      case graphTypeEnum::graph_view_t_float_mg_transposed :
        delete graph_ptr.graph_view_t_float_mg_transposed_ptr;
        break;
      case graphTypeEnum::graph_view_t_double_mg_transposed :
        delete graph_ptr.graph_view_t_double_mg_transposed_ptr;
        break;
      default :
        break;
    }
    graph_ptr_type = graphTypeEnum::null;
  }
   */
  graphPtrUnion graph_ptr;
  graphTypeEnum graph_ptr_type;
};

// Factory function for creating graph containers from basic types
// FIXME: Should local_* values be void* as well?
void create_graph_t(graph_container_t& graph_container,
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
                      graph_container_t graph_container,
                      int* parts,
                      size_t max_level,
                      weight_t resolution);

}  // namespace cython
}  // namespace cugraph
