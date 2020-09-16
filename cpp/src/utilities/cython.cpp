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
#include <experimental/graph_view.hpp>
#include <graph.hpp>
#include <raft/handle.hpp>
#include <utilities/cython.hpp>

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
  // FIXME: This is soon-to-be legacy code left in place until the new graph_t
  // class is supported everywhere else. Remove everything down to the comment
  // line after the return stmnt.
  // Keep new code below return stmnt enabled to ensure it builds.
  if (weightType == numberTypeEnum::floatType) {
    auto g = new GraphCSRView<int, int, float>(reinterpret_cast<int*>(offsets),
                                               reinterpret_cast<int*>(indices),
                                               reinterpret_cast<float*>(weights),
                                               num_vertices,
                                               num_edges);
    graph_container.graph_ptr_union.GraphCSRViewFloatPtr = g;
    graph_container.graph_ptr_type                       = graphTypeEnum::GraphCSRViewFloat;
    g->set_local_data(local_vertices, local_edges, local_offsets);
    g->set_handle(const_cast<raft::handle_t*>(&handle));

  } else {
    auto g = new GraphCSRView<int, int, double>(reinterpret_cast<int*>(offsets),
                                                reinterpret_cast<int*>(indices),
                                                reinterpret_cast<double*>(weights),
                                                num_vertices,
                                                num_edges);
    graph_container.graph_ptr_union.GraphCSRViewDoublePtr = g;
    graph_container.graph_ptr_type                        = graphTypeEnum::GraphCSRViewDouble;
    g->set_local_data(local_vertices, local_edges, local_offsets);
    g->set_handle(const_cast<raft::handle_t*>(&handle));
  }

  return;
  ////////////////////////////////////////////////////////////////////////////////////

  bool do_expensive_check{false};
  bool sorted_by_global_degree_within_vertex_partition{false};
  experimental::graph_properties_t graph_props{.is_symmetric = false, .is_multigraph = false};

  if (multi_gpu) {
    std::vector<int const*> adjmatrix_partition_offsets_vect;
    std::vector<int const*> adjmatrix_partition_indices_vect;
    std::vector<int> vertex_partition_segment_offsets_vect;
    std::vector<int> vertex_partition_offsets;
    experimental::partition_t<int> partition(vertex_partition_offsets, false, 0, 0, 0, 0);

    if (weightType == numberTypeEnum::floatType) {
      std::vector<float const*> adjmatrix_partition_weights_vect;
      auto g = new experimental::graph_view_t<int, int, float, false, true>(
        handle,
        adjmatrix_partition_offsets_vect,
        adjmatrix_partition_indices_vect,
        adjmatrix_partition_weights_vect,
        vertex_partition_segment_offsets_vect,
        partition,
        num_vertices,
        num_edges,
        graph_props,
        sorted_by_global_degree_within_vertex_partition,
        do_expensive_check);
      graph_container.graph_ptr_union.graph_view_t_float_mg_ptr = g;
      graph_container.graph_ptr_type = graphTypeEnum::graph_view_t_float_mg;

    } else {
      std::vector<double const*> adjmatrix_partition_weights_vect;
      auto g = new experimental::graph_view_t<int, int, double, false, true>(
        handle,
        adjmatrix_partition_offsets_vect,
        adjmatrix_partition_indices_vect,
        adjmatrix_partition_weights_vect,
        vertex_partition_segment_offsets_vect,
        partition,
        num_vertices,
        num_edges,
        graph_props,
        sorted_by_global_degree_within_vertex_partition,
        do_expensive_check);
      graph_container.graph_ptr_union.graph_view_t_double_mg_ptr = g;
      graph_container.graph_ptr_type = graphTypeEnum::graph_view_t_double_mg;
    }

  } else {
    auto offsets_array = reinterpret_cast<int const*>(offsets);
    auto indices_array = reinterpret_cast<int const*>(indices);
    std::vector<int> segment_offsets;

    if (weightType == numberTypeEnum::floatType) {
      auto weights_array = reinterpret_cast<float const*>(weights);
      auto g             = new experimental::graph_view_t<int, int, float, false, false>(
        handle,
        offsets_array,
        indices_array,
        weights_array,
        segment_offsets,
        num_vertices,
        num_edges,
        graph_props,
        sorted_by_global_degree_within_vertex_partition,
        do_expensive_check);
      graph_container.graph_ptr_union.graph_view_t_float_ptr = g;
      graph_container.graph_ptr_type                         = graphTypeEnum::graph_view_t_float;

    } else {
      auto weights_array = reinterpret_cast<double const*>(weights);
      auto g             = new experimental::graph_view_t<int, int, double, false, false>(
        handle,
        offsets_array,
        indices_array,
        weights_array,
        segment_offsets,
        num_vertices,
        num_edges,
        graph_props,
        sorted_by_global_degree_within_vertex_partition,
        do_expensive_check);
      graph_container.graph_ptr_union.graph_view_t_double_ptr = g;
      graph_container.graph_ptr_type                          = graphTypeEnum::graph_view_t_double;
    }
  }
}

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
weight_t call_louvain(raft::handle_t const& handle,
                      graph_container_t& graph_container,
                      int* parts,
                      size_t max_level,
                      weight_t resolution)
{
  weight_t final_modularity;

  if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewFloat) {
    std::pair<size_t, float> results =
      louvain(handle,
              *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
              parts,
              max_level,
              static_cast<float>(resolution));
    final_modularity = results.second;
  } else {
    std::pair<size_t, double> results =
      louvain(handle,
              *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
              parts,
              max_level,
              static_cast<double>(resolution));
    final_modularity = results.second;
  }

  return final_modularity;
}

// Explicit instantiations
template float call_louvain(raft::handle_t const& handle,
                            graph_container_t& graph_container,
                            int* parts,
                            size_t max_level,
                            float resolution);

template double call_louvain(raft::handle_t const& handle,
                             graph_container_t& graph_container,
                             int* parts,
                             size_t max_level,
                             double resolution);

}  // namespace cython
}  // namespace cugraph
