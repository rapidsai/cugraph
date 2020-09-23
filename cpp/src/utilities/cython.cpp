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
#include <utilities/error.hpp>

namespace cugraph {
namespace cython {

// Populates a graph_container_t with a pointer to a new graph object and sets
// the meta-data accordingly.  The graph container owns the pointer and it is
// assumed it will delete it on destruction.
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
                              bool multi_gpu)
{
  CUGRAPH_EXPECTS(graph_container.graph_ptr_type == graphTypeEnum::null,
                  "populate_graph_container() can only be called on an empty container.");

  // FIXME: This is soon-to-be legacy code left in place until the new graph_t
  // class is supported everywhere else. Remove everything down to the comment
  // line after the return stmnt.
  // Keep new code below return stmnt enabled to ensure it builds.
  if (weightType == numberTypeEnum::floatType) {
    switch (legacyType) {
      case legacyGraphTypeEnum::CSR: {
        graph_container.graph_ptr_union.GraphCSRViewFloatPtr =
          std::make_unique<GraphCSRView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_vertices,
                                                          num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCSRViewFloat;
        (graph_container.graph_ptr_union.GraphCSRViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSRViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case legacyGraphTypeEnum::CSC: {
        graph_container.graph_ptr_union.GraphCSCViewFloatPtr =
          std::make_unique<GraphCSCView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_vertices,
                                                          num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCSCViewFloat;
        (graph_container.graph_ptr_union.GraphCSCViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSCViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case legacyGraphTypeEnum::COO: {
        graph_container.graph_ptr_union.GraphCOOViewFloatPtr =
          std::make_unique<GraphCOOView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_vertices,
                                                          num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCOOViewFloat;
        (graph_container.graph_ptr_union.GraphCOOViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCOOViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
    }

  } else {
    switch (legacyType) {
      case legacyGraphTypeEnum::CSR: {
        graph_container.graph_ptr_union.GraphCSRViewDoublePtr =
          std::make_unique<GraphCSRView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_vertices,
                                                           num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCSRViewDouble;
        (graph_container.graph_ptr_union.GraphCSRViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSRViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case legacyGraphTypeEnum::CSC: {
        graph_container.graph_ptr_union.GraphCSCViewDoublePtr =
          std::make_unique<GraphCSCView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_vertices,
                                                           num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCSCViewDouble;
        (graph_container.graph_ptr_union.GraphCSCViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSCViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case legacyGraphTypeEnum::COO: {
        graph_container.graph_ptr_union.GraphCOOViewDoublePtr =
          std::make_unique<GraphCOOView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_vertices,
                                                           num_edges);
        graph_container.graph_ptr_type = graphTypeEnum::GraphCOOViewDouble;
        (graph_container.graph_ptr_union.GraphCOOViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCOOViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
    }
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
      graph_container.graph_ptr_union.graph_view_t_float_mg_ptr =
        std::unique_ptr<experimental::graph_view_t<int, int, float, false, true>>(g);
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
      graph_container.graph_ptr_union.graph_view_t_double_mg_ptr =
        std::unique_ptr<experimental::graph_view_t<int, int, double, false, true>>(g);
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
      graph_container.graph_ptr_union.graph_view_t_float_ptr =
        std::unique_ptr<experimental::graph_view_t<int, int, float, false, false>>(g);
      graph_container.graph_ptr_type = graphTypeEnum::graph_view_t_float;

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
      graph_container.graph_ptr_union.graph_view_t_double_ptr =
        std::unique_ptr<experimental::graph_view_t<int, int, double, false, false>>(g);
      graph_container.graph_ptr_type = graphTypeEnum::graph_view_t_double;
    }
  }
}

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
weight_t call_louvain(raft::handle_t const& handle,
                      graph_container_t const& graph_container,
                      void* parts,
                      size_t max_level,
                      weight_t resolution)
{
  weight_t final_modularity;

  // FIXME: the only graph types currently in the container have ints for
  // vertex_t and edge_t types. In the future, additional types for vertices and
  // edges will be available, and when that happens, additional castings will be
  // needed for the 'parts' arg in particular. For now, it is hardcoded to int.
  if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewFloat) {
    std::pair<size_t, float> results =
      louvain(handle,
              *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
              reinterpret_cast<int*>(parts),
              max_level,
              static_cast<float>(resolution));
    final_modularity = results.second;
  } else {
    std::pair<size_t, double> results =
      louvain(handle,
              *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
              reinterpret_cast<int*>(parts),
              max_level,
              static_cast<double>(resolution));
    final_modularity = results.second;
  }

  return final_modularity;
}

// Explicit instantiations
template float call_louvain(raft::handle_t const& handle,
                            graph_container_t const& graph_container,
                            void* parts,
                            size_t max_level,
                            float resolution);

template double call_louvain(raft::handle_t const& handle,
                             graph_container_t const& graph_container,
                             void* parts,
                             size_t max_level,
                             double resolution);

}  // namespace cython
}  // namespace cugraph
