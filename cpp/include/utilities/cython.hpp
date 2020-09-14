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
// #include <experimental/graph_view.hpp>
#include <raft/handle.hpp>

namespace cugraph {
namespace cython {

// FIXME: use std::variant instead of a union if possible
// FIXME: add both CSRView and graph_type_t objects for easier testing during
//        the transition
union graphUnion {
  graphUnion() {}
  GraphCSRView<int, int, float> GraphCSRViewFloat;
  GraphCSRView<int, int, double> GraphCSRViewDouble;
};

enum weightTypeEnum { floatType = 0, doubleType = 1 };

// FIXME: Add comments describing this struct, where it's used, etc.
struct graph_container_t {
  graph_container_t() {}
  graphUnion graph;
  // FIXME: cython issues using an enum so just using an int for now.
  // weightTypeEnum wType;
  int wType;
};

// Factory function for creating graph containers from basic types
// FIXME: This should accept void* for offsets and indices as well and take a
//        dtype directly for each instead of the enum/int.
graph_container_t create_graph_t(raft::handle_t const& handle,
                                 int* offsets,
                                 int* indices,
                                 void* weights,
                                 int weightType,
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
