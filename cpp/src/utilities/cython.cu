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
#include <partition_manager.hpp>
#include <raft/handle.hpp>
#include <utilities/cython.hpp>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>

namespace cugraph {
namespace cython {

namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<multi_gpu>* = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t& handle,
             void* src_vertices,
             void* dst_vertices,
             void* weights,
             experimental::partition_t<int> const& partition,
             size_t num_vertices,
             size_t num_edges,
             experimental::graph_properties_t& graph_props,
             bool sorted_by_global_degree_within_vertex_partition,
             bool do_expensive_check)
{
  std::vector<experimental::edgelist_t<vertex_t, edge_t, weight_t>> edgelist(
    {{reinterpret_cast<vertex_t*>(src_vertices),
      reinterpret_cast<vertex_t*>(dst_vertices),
      reinterpret_cast<weight_t*>(weights),
      static_cast<edge_t>(num_edges)}});

  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    partition,
    static_cast<vertex_t>(num_vertices),
    static_cast<edge_t>(num_edges),
    graph_props,
    sorted_by_global_degree_within_vertex_partition,
    do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<!multi_gpu>* = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t& handle,
             void* src_vertices,
             void* dst_vertices,
             void* weights,
             experimental::partition_t<int> const& partition,
             size_t num_vertices,
             size_t num_edges,
             experimental::graph_properties_t& graph_props,
             bool sorted_by_global_degree_within_vertex_partition,
             bool do_expensive_check)
{
  experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    reinterpret_cast<vertex_t*>(src_vertices),
    reinterpret_cast<vertex_t*>(dst_vertices),
    reinterpret_cast<weight_t*>(weights),
    static_cast<edge_t>(num_edges)};

  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    static_cast<vertex_t>(num_vertices),
    graph_props,
    sorted_by_global_degree_within_vertex_partition,
    do_expensive_check);
}

}  // namespace detail

// Populates a graph_container_t with a pointer to a new graph object and sets
// the meta-data accordingly.  The graph container owns the pointer and it is
// assumed it will delete it on destruction.
void populate_graph_container(graph_container_t& graph_container,
                              raft::handle_t& handle,
                              void* src_vertices,
                              void* dst_vertices,
                              void* weights,
                              void* vertex_partition_offsets,
                              numberTypeEnum vertexType,
                              numberTypeEnum edgeType,
                              numberTypeEnum weightType,
                              size_t num_vertices,
                              size_t num_edges,
                              int partition_row_size,  // pcols
                              int partition_col_size,  // prows
                              bool transposed,
                              bool multi_gpu)
{
  CUGRAPH_EXPECTS(graph_container.graph_ptr_type == graphTypeEnum::null,
                  "populate_graph_container() can only be called on an empty container.");

  bool do_expensive_check{false};
  bool hypergraph_partitioned{false};

  raft::comms::comms_t const& communicator = handle.get_comms();
  int const rank                           = communicator.get_rank();
  int partition_row_rank                   = rank / partition_row_size;
  int partition_col_rank                   = rank % partition_row_size;

  // Setup the subcommunicators needed for this partition on the handle
  partition_2d::subcomm_factory_t<partition_2d::key_naming_t, int> subcomm_factory(
    handle, partition_row_size);

  // Copy the contents of the vertex_partition_offsets (host array) to a vector
  // as needed by the partition_t ctor.
  int* vertex_partition_offsets_array = reinterpret_cast<int*>(vertex_partition_offsets);
  std::vector<int> vertex_partition_offsets_vect;  // vertex_t

  for (int32_t i = 0; i < (partition_row_size * partition_col_size); ++i) {
    vertex_partition_offsets_vect.push_back(vertex_partition_offsets_array[i]);
  }
  experimental::partition_t<int> partition(vertex_partition_offsets_vect,
                                           hypergraph_partitioned,
                                           partition_row_size,
                                           partition_col_size,
                                           partition_row_rank,
                                           partition_col_rank);

  experimental::graph_properties_t graph_props{.is_symmetric = false, .is_multigraph = false};

  auto src_vertices_array = reinterpret_cast<int*>(src_vertices);
  auto dst_vertices_array = reinterpret_cast<int*>(dst_vertices);

  if (multi_gpu) {
    bool sorted_by_global_degree_within_vertex_partition{false};

    if (vertexType == numberTypeEnum::int32Type) {
      if (edgeType == numberTypeEnum::int32Type) {
        if (weightType == numberTypeEnum::floatType) {
          if (transposed) {
            graph_container.graph_ptr_union.graph_t_float_mg_transposed_ptr =
              detail::create_graph<int32_t, int32_t, float, true, true>(
                handle,
                src_vertices,
                dst_vertices,
                weights,
                partition,
                num_vertices,
                num_edges,
                graph_props,
                sorted_by_global_degree_within_vertex_partition,
                do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_float_mg_transposed;
          } else {
            graph_container.graph_ptr_union.graph_t_float_mg_ptr =
              detail::create_graph<int32_t, int32_t, float, false, true>(
                handle,
                src_vertices,
                dst_vertices,
                weights,
                partition,
                num_vertices,
                num_edges,
                graph_props,
                sorted_by_global_degree_within_vertex_partition,
                do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_float_mg;
          }
        } else {
          if (transposed) {
            graph_container.graph_ptr_union.graph_t_double_mg_transposed_ptr =
              detail::create_graph<int32_t, int32_t, double, true, true>(
                handle,
                src_vertices,
                dst_vertices,
                weights,
                partition,
                num_vertices,
                num_edges,
                graph_props,
                sorted_by_global_degree_within_vertex_partition,
                do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_double_mg_transposed;
          } else {
            graph_container.graph_ptr_union.graph_t_double_mg_ptr =
              detail::create_graph<int32_t, int32_t, double, false, true>(
                handle,
                src_vertices,
                dst_vertices,
                weights,
                partition,
                num_vertices,
                num_edges,
                graph_props,
                sorted_by_global_degree_within_vertex_partition,
                do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_double_mg;
          }
        }
      } else {
        CUGRAPH_FAIL("int64_t not supported yet");
      }
    } else {
      CUGRAPH_FAIL("int64_t not supported yet");
    }

  } else {
    bool sorted_by_degree{false};

    if (vertexType == numberTypeEnum::int32Type) {
      if (edgeType == numberTypeEnum::int32Type) {
        if (weightType == numberTypeEnum::floatType) {
          if (transposed) {
            graph_container.graph_ptr_union.graph_t_float_transposed_ptr =
              detail::create_graph<int32_t, int32_t, float, true, false>(handle,
                                                                         src_vertices,
                                                                         dst_vertices,
                                                                         weights,
                                                                         partition,
                                                                         num_vertices,
                                                                         num_edges,
                                                                         graph_props,
                                                                         sorted_by_degree,
                                                                         do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_float_transposed;
          } else {
            graph_container.graph_ptr_union.graph_t_float_ptr =
              detail::create_graph<int32_t, int32_t, float, false, false>(handle,
                                                                          src_vertices,
                                                                          dst_vertices,
                                                                          weights,
                                                                          partition,
                                                                          num_vertices,
                                                                          num_edges,
                                                                          graph_props,
                                                                          sorted_by_degree,
                                                                          do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_float;
          }
        } else {
          if (transposed) {
            graph_container.graph_ptr_union.graph_t_double_transposed_ptr =
              detail::create_graph<int32_t, int32_t, double, true, false>(handle,
                                                                          src_vertices,
                                                                          dst_vertices,
                                                                          weights,
                                                                          partition,
                                                                          num_vertices,
                                                                          num_edges,
                                                                          graph_props,
                                                                          sorted_by_degree,
                                                                          do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_double_transposed;
          } else {
            graph_container.graph_ptr_union.graph_t_double_ptr =
              detail::create_graph<int32_t, int32_t, double, false, false>(handle,
                                                                           src_vertices,
                                                                           dst_vertices,
                                                                           weights,
                                                                           partition,
                                                                           num_vertices,
                                                                           num_edges,
                                                                           graph_props,
                                                                           sorted_by_degree,
                                                                           do_expensive_check);

            graph_container.graph_ptr_type = graphTypeEnum::graph_t_double;
          }
        }
      } else {
        CUGRAPH_FAIL("int64_t not supported yet");
      }
    } else {
      CUGRAPH_FAIL("int64_t not supported yet");
    }
  }
}

void populate_graph_container_legacy(graph_container_t& graph_container,
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
                                     int* local_offsets)
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
}

////////////////////////////////////////////////////////////////////////////////

namespace detail {
template <typename graph_view_t, typename weight_t>
std::pair<size_t, weight_t> call_louvain(raft::handle_t const& handle,
                                         graph_view_t const& graph_view,
                                         void* identifiers,
                                         void* parts,
                                         size_t max_level,
                                         weight_t resolution)
{
  thrust::copy(//rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
               thrust::device,
               thrust::make_counting_iterator(graph_view.get_local_vertex_first()),
               thrust::make_counting_iterator(graph_view.get_local_vertex_last()),
               reinterpret_cast<typename graph_view_t::vertex_type*>(identifiers));

  return louvain(handle,
                 graph_view,
                 reinterpret_cast<typename graph_view_t::vertex_type*>(parts),
                 max_level,
                 static_cast<weight_t>(resolution));
}

}  // namespace detail

// Wrapper for calling Louvain using a graph container
template <typename weight_t>
std::pair<size_t, weight_t> call_louvain(raft::handle_t const& handle,
                                         graph_container_t const& graph_container,
                                         void* identifiers,
                                         void* parts,
                                         size_t max_level,
                                         weight_t resolution)
{
  std::pair<size_t, weight_t> results;

  // FIXME: the only graph types currently in the container have ints for
  // vertex_t and edge_t types. In the future, additional types for vertices and
  // edges will be available, and when that happens, additional castings will be
  // needed for the 'parts' arg in particular. For now, it is hardcoded to int.
  if (graph_container.graph_ptr_type == graphTypeEnum::graph_t_float_mg) {
    return detail::call_louvain(handle,
                                graph_container.graph_ptr_union.graph_t_float_mg_ptr->view(),
                                identifiers,
                                parts,
                                max_level,
                                resolution);
  } else if (graph_container.graph_ptr_type == graphTypeEnum::graph_t_double_mg) {
    return detail::call_louvain(handle,
                                graph_container.graph_ptr_union.graph_t_double_mg_ptr->view(),
                                identifiers,
                                parts,
                                max_level,
                                resolution);
  } else if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewFloat) {
    //     if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewFloat) {
    graph_container.graph_ptr_union.GraphCSCViewFloatPtr->get_vertex_identifiers(
      static_cast<int32_t*>(identifiers));
    results = louvain(handle,
                      *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
                      reinterpret_cast<int*>(parts),
                      max_level,
                      static_cast<float>(resolution));
  } else if (graph_container.graph_ptr_type == graphTypeEnum::GraphCSRViewDouble) {
    graph_container.graph_ptr_union.GraphCSCViewDoublePtr->get_vertex_identifiers(
      static_cast<int32_t*>(identifiers));
    results = louvain(handle,
                      *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
                      reinterpret_cast<int*>(parts),
                      max_level,
                      static_cast<double>(resolution));
  }

  return results;
}

// Explicit instantiations
template std::pair<size_t, float> call_louvain(raft::handle_t const& handle,
                                               graph_container_t const& graph_container,
                                               void* identifiers,
                                               void* parts,
                                               size_t max_level,
                                               float resolution);

template std::pair<size_t, double> call_louvain(raft::handle_t const& handle,
                                                graph_container_t const& graph_container,
                                                void* identifiers,
                                                void* parts,
                                                size_t max_level,
                                                double resolution);

}  // namespace cython
}  // namespace cugraph
