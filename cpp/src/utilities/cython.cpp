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

#include <algorithms.hpp>
#include <utilities/cython.hpp>

#include <raft/handle.hpp>

namespace cugraph {
namespace cython {

// Factory function for creating graph containers from basic types
// FIXME: This should accept void* for offsets and indices as well and take a
//        dtype directly for each instead of the enum/int.
graph_container_t create_graph_t(raft::handle_t const& handle,
                                 int* offsets,
                                 int* indices,
                                 void* weights,
                                 weightTypeEnum weightType,
                                 int num_vertices,
                                 int num_edges,
                                 int* local_vertices,
                                 int* local_edges,
                                 int* local_offsets,
                                 bool transposed,
                                 bool multi_gpu)
{
  graph_container_t graph_container{};
  graph_container.wType = weightType;

  // FIXME: instantiate graph_type_t instead when ready, add conditionals for
  // properly instantiating MG or not based on multi_gpu, etc.

  if (weightType == weightTypeEnum::floatType) {
    graph_container.graph.GraphCSRViewFloat = GraphCSRView<int, int, float>(
      offsets, indices, reinterpret_cast<float*>(weights), num_vertices, num_edges);
    graph_container.graph.GraphCSRViewFloat.set_local_data(
      local_vertices, local_edges, local_offsets);
    graph_container.graph.GraphCSRViewFloat.set_handle(const_cast<raft::handle_t*>(&handle));

  } else {
    graph_container.graph.GraphCSRViewDouble = GraphCSRView<int, int, double>(
      offsets, indices, reinterpret_cast<double*>(weights), num_vertices, num_edges);
    graph_container.graph.GraphCSRViewDouble.set_local_data(
      local_vertices, local_edges, local_offsets);
    graph_container.graph.GraphCSRViewDouble.set_handle(const_cast<raft::handle_t*>(&handle));
  }

  return graph_container;
}

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
weight_t call_louvain(raft::handle_t const& handle,
                      graph_container_t graph_container,
                      int* parts,
                      size_t max_level,
                      weight_t resolution)
{
  weight_t final_modularity;

  if (graph_container.wType == weightTypeEnum::floatType) {
    std::pair<size_t, float> results = louvain(handle,
                                               graph_container.graph.GraphCSRViewFloat,
                                               parts,
                                               max_level,
                                               static_cast<float>(resolution));
    final_modularity                 = results.second;
  } else {
    std::pair<size_t, double> results = louvain(handle,
                                                graph_container.graph.GraphCSRViewDouble,
                                                parts,
                                                max_level,
                                                static_cast<double>(resolution));
    final_modularity                  = results.second;
  }

  return final_modularity;
}

// Explicit instantiations
template float call_louvain(raft::handle_t const& handle,
                            graph_container_t graph_container,
                            int* parts,
                            size_t max_level,
                            float resolution);

template double call_louvain(raft::handle_t const& handle,
                             graph_container_t graph_container,
                             int* parts,
                             size_t max_level,
                             double resolution);

}  // namespace cython
}  // namespace cugraph
