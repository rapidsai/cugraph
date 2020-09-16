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

// Populates a graph_container_t with a pointer to a new graph object and sets
// the meta-data accordingly.  The graph container owns the pointer and it is
// assumed it will delete it on destruction.
//
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
                    bool multi_gpu)
{

  if (weightType == numberTypeEnum::floatType) {
    graph_container.graph_ptr.GraphCSRViewFloatPtr = new GraphCSRView<int, int, float>(
      reinterpret_cast<int*>(offsets),
      reinterpret_cast<int*>(indices),
      reinterpret_cast<float*>(weights),
      num_vertices,
      num_edges);
    graph_container.graph_ptr.GraphCSRViewFloatPtr->set_local_data(
      local_vertices, local_edges, local_offsets);
    graph_container.graph_ptr.GraphCSRViewFloatPtr->set_handle(const_cast<raft::handle_t*>(&handle));
    graph_container.graph_ptr_type = graphTypeEnum::GraphCSRViewFloat;

  } else {
    graph_container.graph_ptr.GraphCSRViewDoublePtr = new GraphCSRView<int, int, double>(
      reinterpret_cast<int*>(offsets),
      reinterpret_cast<int*>(indices),
      reinterpret_cast<double*>(weights),
      num_vertices,
      num_edges);
    graph_container.graph_ptr.GraphCSRViewDoublePtr->set_local_data(
      local_vertices, local_edges, local_offsets);
    graph_container.graph_ptr.GraphCSRViewDoublePtr->set_handle(const_cast<raft::handle_t*>(&handle));
    graph_container.graph_ptr_type = graphTypeEnum::GraphCSRViewDouble;
  }
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

  if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewFloat) {
    std::pair<size_t, float> results = louvain(handle,
                                               *(graph_container.graph_ptr.GraphCSRViewFloatPtr),
                                               parts,
                                               max_level,
                                               static_cast<float>(resolution));
    final_modularity                 = results.second;
  } else {
    std::pair<size_t, double> results = louvain(handle,
                                                *(graph_container.graph_ptr.GraphCSRViewDoublePtr),
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
