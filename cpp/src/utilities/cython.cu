/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <experimental/detail/graph_utils.cuh>
#include <experimental/graph_functions.hpp>
#include <experimental/graph_view.hpp>
#include <graph.hpp>
#include <partition_manager.hpp>
#include <raft/handle.hpp>
#include <utilities/cython.hpp>
#include <utilities/error.hpp>
#include <utilities/shuffle_comm.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

namespace cugraph {
namespace cython {

namespace detail {

// FIXME: Add description of this function
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<multi_gpu>* = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t const& handle, graph_container_t const& graph_container)
{
  std::vector<experimental::edgelist_t<vertex_t, edge_t, weight_t>> edgelist(
    {{reinterpret_cast<vertex_t*>(graph_container.src_vertices),
      reinterpret_cast<vertex_t*>(graph_container.dst_vertices),
      reinterpret_cast<weight_t*>(graph_container.weights),
      static_cast<edge_t>(graph_container.num_partition_edges)}});

  std::vector<vertex_t> partition_offsets_vector(
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets),
    reinterpret_cast<vertex_t*>(graph_container.vertex_partition_offsets) +
      (graph_container.row_comm_size * graph_container.col_comm_size) + 1);

  experimental::partition_t<vertex_t> partition(partition_offsets_vector,
                                                graph_container.hypergraph_partitioned,
                                                graph_container.row_comm_size,
                                                graph_container.col_comm_size,
                                                graph_container.row_comm_rank,
                                                graph_container.col_comm_rank);

  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    partition,
    static_cast<vertex_t>(graph_container.num_global_vertices),
    static_cast<edge_t>(graph_container.num_global_edges),
    graph_container.graph_props,
    // FIXME:  This currently fails if sorted_by_degree is true...
    // graph_container.sorted_by_degree,
    false,
    graph_container.do_expensive_check);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool transposed,
          bool multi_gpu,
          std::enable_if_t<!multi_gpu>* = nullptr>
std::unique_ptr<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>
create_graph(raft::handle_t const& handle, graph_container_t const& graph_container)
{
  experimental::edgelist_t<vertex_t, edge_t, weight_t> edgelist{
    reinterpret_cast<vertex_t*>(graph_container.src_vertices),
    reinterpret_cast<vertex_t*>(graph_container.dst_vertices),
    reinterpret_cast<weight_t*>(graph_container.weights),
    static_cast<edge_t>(graph_container.num_partition_edges)};
  return std::make_unique<experimental::graph_t<vertex_t, edge_t, weight_t, transposed, multi_gpu>>(
    handle,
    edgelist,
    static_cast<vertex_t>(graph_container.num_global_vertices),
    graph_container.graph_props,
    graph_container.sorted_by_degree,
    graph_container.do_expensive_check);
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
                              size_t num_partition_edges,
                              size_t num_global_vertices,
                              size_t num_global_edges,
                              bool sorted_by_degree,
                              bool transposed,
                              bool multi_gpu)
{
  CUGRAPH_EXPECTS(graph_container.graph_type == graphTypeEnum::null,
                  "populate_graph_container() can only be called on an empty container.");

  bool do_expensive_check{true};
  bool hypergraph_partitioned{false};

  if (multi_gpu) {
    auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
    auto const row_comm_rank = row_comm.get_rank();
    auto const row_comm_size = row_comm.get_size();  // pcols
    auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
    auto const col_comm_rank = col_comm.get_rank();
    auto const col_comm_size = col_comm.get_size();  // prows
    graph_container.row_comm_size = row_comm_size;
    graph_container.col_comm_size = col_comm_size;
    graph_container.row_comm_rank = row_comm_rank;
    graph_container.col_comm_rank = col_comm_rank;
  }

  graph_container.vertex_partition_offsets = vertex_partition_offsets;
  graph_container.src_vertices             = src_vertices;
  graph_container.dst_vertices             = dst_vertices;
  graph_container.weights                  = weights;
  graph_container.num_partition_edges      = num_partition_edges;
  graph_container.num_global_vertices      = num_global_vertices;
  graph_container.num_global_edges         = num_global_edges;
  graph_container.vertexType               = vertexType;
  graph_container.edgeType                 = edgeType;
  graph_container.weightType               = weightType;
  graph_container.transposed               = transposed;
  graph_container.is_multi_gpu             = multi_gpu;
  graph_container.hypergraph_partitioned   = hypergraph_partitioned;
  graph_container.sorted_by_degree         = sorted_by_degree;
  graph_container.do_expensive_check       = do_expensive_check;

  experimental::graph_properties_t graph_props{.is_symmetric = false, .is_multigraph = false};
  graph_container.graph_props = graph_props;

  graph_container.graph_type = graphTypeEnum::graph_t;
}

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
                                     int* local_offsets)
{
  CUGRAPH_EXPECTS(graph_container.graph_type == graphTypeEnum::null,
                  "populate_graph_container() can only be called on an empty container.");

  // FIXME: This is soon-to-be legacy code left in place until the new graph_t
  // class is supported everywhere else. Remove everything down to the comment
  // line after the return stmnt.
  // Keep new code below return stmnt enabled to ensure it builds.
  if (weightType == numberTypeEnum::floatType) {
    switch (legacyType) {
      case graphTypeEnum::LegacyCSR: {
        graph_container.graph_ptr_union.GraphCSRViewFloatPtr =
          std::make_unique<GraphCSRView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_global_vertices,
                                                          num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCSRViewFloat;
        (graph_container.graph_ptr_union.GraphCSRViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSRViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case graphTypeEnum::LegacyCSC: {
        graph_container.graph_ptr_union.GraphCSCViewFloatPtr =
          std::make_unique<GraphCSCView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_global_vertices,
                                                          num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCSCViewFloat;
        (graph_container.graph_ptr_union.GraphCSCViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSCViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case graphTypeEnum::LegacyCOO: {
        graph_container.graph_ptr_union.GraphCOOViewFloatPtr =
          std::make_unique<GraphCOOView<int, int, float>>(reinterpret_cast<int*>(offsets),
                                                          reinterpret_cast<int*>(indices),
                                                          reinterpret_cast<float*>(weights),
                                                          num_global_vertices,
                                                          num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCOOViewFloat;
        (graph_container.graph_ptr_union.GraphCOOViewFloatPtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCOOViewFloatPtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      default: CUGRAPH_FAIL("unsupported graphTypeEnum value"); break;
    }

  } else {
    switch (legacyType) {
      case graphTypeEnum::LegacyCSR: {
        graph_container.graph_ptr_union.GraphCSRViewDoublePtr =
          std::make_unique<GraphCSRView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_global_vertices,
                                                           num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCSRViewDouble;
        (graph_container.graph_ptr_union.GraphCSRViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSRViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case graphTypeEnum::LegacyCSC: {
        graph_container.graph_ptr_union.GraphCSCViewDoublePtr =
          std::make_unique<GraphCSCView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_global_vertices,
                                                           num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCSCViewDouble;
        (graph_container.graph_ptr_union.GraphCSCViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCSCViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      case graphTypeEnum::LegacyCOO: {
        graph_container.graph_ptr_union.GraphCOOViewDoublePtr =
          std::make_unique<GraphCOOView<int, int, double>>(reinterpret_cast<int*>(offsets),
                                                           reinterpret_cast<int*>(indices),
                                                           reinterpret_cast<double*>(weights),
                                                           num_global_vertices,
                                                           num_global_edges);
        graph_container.graph_type = graphTypeEnum::GraphCOOViewDouble;
        (graph_container.graph_ptr_union.GraphCOOViewDoublePtr)
          ->set_local_data(local_vertices, local_edges, local_offsets);
        (graph_container.graph_ptr_union.GraphCOOViewDoublePtr)
          ->set_handle(const_cast<raft::handle_t*>(&handle));
      } break;
      default: CUGRAPH_FAIL("unsupported graphTypeEnum value"); break;
    }
  }
  return;
}

////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Final, fully-templatized call.
template <bool transposed,
          typename return_t,
          typename function_t,
          typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool is_multi_gpu>
return_t call_function(raft::handle_t const& handle,
                       graph_container_t const& graph_container,
                       function_t function)
{
  auto graph =
    create_graph<vertex_t, edge_t, weight_t, transposed, is_multi_gpu>(handle, graph_container);

  return function(handle, graph->view());
}

// Makes another call based on vertex_t and edge_t
template <bool transposed,
          typename return_t,
          typename function_t,
          typename weight_t,
          bool is_multi_gpu>
return_t call_function(raft::handle_t const& handle,
                       graph_container_t const& graph_container,
                       function_t function)
{
  // Since only vertex/edge types (int32,int32), (int32,int64), and
  // (int64,int64) are being supported, explicitely check for those types and
  // ensure (int64,int32) is rejected as unsupported.
  if ((graph_container.vertexType == numberTypeEnum::int32Type) &&
      (graph_container.edgeType == numberTypeEnum::int32Type)) {
    return call_function<transposed,
                         return_t,
                         function_t,
                         int32_t,
                         int32_t,
                         weight_t,
                         is_multi_gpu>(handle, graph_container, function);
  } else if ((graph_container.vertexType == numberTypeEnum::int32Type) &&
             (graph_container.edgeType == numberTypeEnum::int64Type)) {
    return call_function<transposed,
                         return_t,
                         function_t,
                         int32_t,
                         int64_t,
                         weight_t,
                         is_multi_gpu>(handle, graph_container, function);
  } else if ((graph_container.vertexType == numberTypeEnum::int64Type) &&
             (graph_container.edgeType == numberTypeEnum::int64Type)) {
    return call_function<transposed,
                         return_t,
                         function_t,
                         int64_t,
                         int64_t,
                         weight_t,
                         is_multi_gpu>(handle, graph_container, function);
  } else {
    CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
  }
}

// Makes another call based on weight_t
template <bool transposed, typename return_t, typename function_t, bool is_multi_gpu>
return_t call_function(raft::handle_t const& handle,
                       graph_container_t const& graph_container,
                       function_t function)
{
  if (graph_container.weightType == numberTypeEnum::floatType) {
    return call_function<transposed, return_t, function_t, float, is_multi_gpu>(
      handle, graph_container, function);
  } else if (graph_container.weightType == numberTypeEnum::doubleType) {
    return call_function<transposed, return_t, function_t, double, is_multi_gpu>(
      handle, graph_container, function);
  } else {
    CUGRAPH_FAIL("weightType unsupported");
  }
}

// Makes another call based on multi_gpu
template <bool transposed, typename return_t, typename function_t>
return_t call_function(raft::handle_t const& handle,
                       graph_container_t const& graph_container,
                       function_t function)
{
  if (graph_container.is_multi_gpu) {
    return call_function<transposed, return_t, function_t, true>(handle, graph_container, function);
  } else {
    return call_function<transposed, return_t, function_t, false>(
      handle, graph_container, function);
  }
}

// Initial call_function() call starts here.
// This makes another call based on transposed
template <typename return_t, typename function_t>
return_t call_function(raft::handle_t const& handle,
                       graph_container_t const& graph_container,
                       function_t function)
{
  if (graph_container.transposed) {
    return call_function<true, return_t, function_t>(handle, graph_container, function);
  } else {
    return call_function<false, return_t, function_t>(handle, graph_container, function);
  }
}

template <typename weight_t>
class louvain_functor {
 public:
  louvain_functor(void* identifiers, void* parts, size_t max_level, weight_t resolution)
    : identifiers_(identifiers), parts_(parts), max_level_(max_level), resolution_(resolution)
  {
  }

  template <typename graph_view_t>
  std::pair<size_t, weight_t> operator()(raft::handle_t const& handle,
                                         graph_view_t const& graph_view)
  {
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 thrust::make_counting_iterator(graph_view.get_local_vertex_first()),
                 thrust::make_counting_iterator(graph_view.get_local_vertex_last()),
                 reinterpret_cast<typename graph_view_t::vertex_type*>(identifiers_));

    return cugraph::louvain(handle,
                            graph_view,
                            reinterpret_cast<typename graph_view_t::vertex_type*>(parts_),
                            max_level_,
                            resolution_);
  }

 private:
  void* identifiers_;  // FIXME: this will be used in a future PR
  void* parts_;
  size_t max_level_;
  weight_t resolution_;
};

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
  // LEGACY PATH - remove when migration to graph_t types complete
  if (graph_container.graph_type == graphTypeEnum::GraphCSRViewFloat) {
    graph_container.graph_ptr_union.GraphCSRViewFloatPtr->get_vertex_identifiers(
      static_cast<int32_t*>(identifiers));
    return louvain(handle,
                   *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
                   reinterpret_cast<int32_t*>(parts),
                   max_level,
                   static_cast<float>(resolution));
  } else if (graph_container.graph_type == graphTypeEnum::GraphCSRViewDouble) {
    graph_container.graph_ptr_union.GraphCSRViewDoublePtr->get_vertex_identifiers(
      static_cast<int32_t*>(identifiers));
    return louvain(handle,
                   *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
                   reinterpret_cast<int32_t*>(parts),
                   max_level,
                   static_cast<double>(resolution));
  }

  // NON-LEGACY PATH
  detail::louvain_functor<weight_t> functor{identifiers, parts, max_level, resolution};

  return detail::call_function<false, std::pair<size_t, weight_t>>(
    handle, graph_container, functor);
}

// Wrapper for calling Pagerank through a graph container
template <typename vertex_t, typename weight_t>
void call_pagerank(raft::handle_t const& handle,
                   graph_container_t const& graph_container,
                   vertex_t* identifiers,
                   weight_t* p_pagerank,
                   vertex_t personalization_subset_size,
                   vertex_t* personalization_subset,
                   weight_t* personalization_values,
                   double alpha,
                   double tolerance,
                   int64_t max_iter,
                   bool has_guess)
{
  if (graph_container.is_multi_gpu) {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, true, true>(handle, graph_container);
      cugraph::experimental::pagerank(handle,
                                      graph->view(),
                                      static_cast<weight_t*>(nullptr),
                                      reinterpret_cast<int32_t*>(personalization_subset),
                                      reinterpret_cast<weight_t*>(personalization_values),
                                      static_cast<int32_t>(personalization_subset_size),
                                      reinterpret_cast<weight_t*>(p_pagerank),
                                      static_cast<weight_t>(alpha),
                                      static_cast<weight_t>(tolerance),
                                      max_iter,
                                      has_guess,
                                      true);
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, true, true>(handle, graph_container);
      cugraph::experimental::pagerank(handle,
                                      graph->view(),
                                      static_cast<weight_t*>(nullptr),
                                      reinterpret_cast<vertex_t*>(personalization_subset),
                                      reinterpret_cast<weight_t*>(personalization_values),
                                      static_cast<vertex_t>(personalization_subset_size),
                                      reinterpret_cast<weight_t*>(p_pagerank),
                                      static_cast<weight_t>(alpha),
                                      static_cast<weight_t>(tolerance),
                                      max_iter,
                                      has_guess,
                                      true);
    }
  } else {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, true, false>(handle, graph_container);
      cugraph::experimental::pagerank(handle,
                                      graph->view(),
                                      static_cast<weight_t*>(nullptr),
                                      reinterpret_cast<int32_t*>(personalization_subset),
                                      reinterpret_cast<weight_t*>(personalization_values),
                                      static_cast<int32_t>(personalization_subset_size),
                                      reinterpret_cast<weight_t*>(p_pagerank),
                                      static_cast<weight_t>(alpha),
                                      static_cast<weight_t>(tolerance),
                                      max_iter,
                                      has_guess,
                                      true);
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, true, false>(handle, graph_container);
      cugraph::experimental::pagerank(handle,
                                      graph->view(),
                                      static_cast<weight_t*>(nullptr),
                                      reinterpret_cast<vertex_t*>(personalization_subset),
                                      reinterpret_cast<weight_t*>(personalization_values),
                                      static_cast<vertex_t>(personalization_subset_size),
                                      reinterpret_cast<weight_t*>(p_pagerank),
                                      static_cast<weight_t>(alpha),
                                      static_cast<weight_t>(tolerance),
                                      max_iter,
                                      has_guess,
                                      true);
    }
  }
}

// Wrapper for calling Katz centrality through a graph container
template <typename vertex_t, typename weight_t>
void call_katz_centrality(raft::handle_t const& handle,
                          graph_container_t const& graph_container,
                          vertex_t* identifiers,
                          weight_t* katz_centrality,
                          double alpha,
                          double beta,
                          double tolerance,
                          int64_t max_iter,
                          bool has_guess,
                          bool normalize)
{
  if (graph_container.graph_type == graphTypeEnum::GraphCSRViewFloat) {
    cugraph::katz_centrality(*(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
                             reinterpret_cast<double*>(katz_centrality),
                             alpha,
                             static_cast<int32_t>(max_iter),
                             tolerance,
                             has_guess,
                             normalize);
    graph_container.graph_ptr_union.GraphCSRViewFloatPtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
  } else if (graph_container.graph_type == graphTypeEnum::graph_t) {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, true, true>(handle, graph_container);
      cugraph::experimental::katz_centrality(handle,
                                             graph->view(),
                                             static_cast<weight_t*>(nullptr),
                                             reinterpret_cast<weight_t*>(katz_centrality),
                                             static_cast<weight_t>(alpha),
                                             static_cast<weight_t>(beta),
                                             static_cast<weight_t>(tolerance),
                                             static_cast<size_t>(max_iter),
                                             has_guess,
                                             normalize,
                                             false);
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, true, true>(handle, graph_container);
      cugraph::experimental::katz_centrality(handle,
                                             graph->view(),
                                             static_cast<weight_t*>(nullptr),
                                             reinterpret_cast<weight_t*>(katz_centrality),
                                             static_cast<weight_t>(alpha),
                                             static_cast<weight_t>(beta),
                                             static_cast<weight_t>(tolerance),
                                             static_cast<size_t>(max_iter),
                                             has_guess,
                                             normalize,
                                             false);
    } else {
      CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
    }
  }
}

// Wrapper for calling BFS through a graph container
template <typename vertex_t, typename weight_t>
void call_bfs(raft::handle_t const& handle,
              graph_container_t const& graph_container,
              vertex_t* identifiers,
              vertex_t* distances,
              vertex_t* predecessors,
              double* sp_counters,
              const vertex_t start_vertex,
              bool directed)
{
  if (graph_container.graph_type == graphTypeEnum::GraphCSRViewFloat) {
    graph_container.graph_ptr_union.GraphCSRViewFloatPtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
    bfs(handle,
        *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
        reinterpret_cast<int32_t*>(distances),
        reinterpret_cast<int32_t*>(predecessors),
        sp_counters,
        static_cast<int32_t>(start_vertex),
        directed);
  } else if (graph_container.graph_type == graphTypeEnum::GraphCSRViewDouble) {
    graph_container.graph_ptr_union.GraphCSRViewDoublePtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
    bfs(handle,
        *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
        reinterpret_cast<int32_t*>(distances),
        reinterpret_cast<int32_t*>(predecessors),
        sp_counters,
        static_cast<int32_t>(start_vertex),
        directed);
  } else if (graph_container.graph_type == graphTypeEnum::graph_t) {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, false, true>(handle, graph_container);
      cugraph::experimental::bfs(handle,
                                 graph->view(),
                                 reinterpret_cast<int32_t*>(distances),
                                 reinterpret_cast<int32_t*>(predecessors),
                                 static_cast<int32_t>(start_vertex));
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, false, true>(handle, graph_container);
      cugraph::experimental::bfs(handle,
                                 graph->view(),
                                 reinterpret_cast<vertex_t*>(distances),
                                 reinterpret_cast<vertex_t*>(predecessors),
                                 static_cast<vertex_t>(start_vertex));
    } else {
      CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
    }
  }
}

// Wrapper for calling extract_egonet through a graph container
// FIXME : this should not be a legacy COO and it is not clear how to handle C++ api return type as
// is.graph_container Need to figure out how to return edge lists
template <typename vertex_t, typename weight_t>
std::unique_ptr<cy_multi_edgelists_t> call_egonet(raft::handle_t const& handle,
                                                  graph_container_t const& graph_container,
                                                  vertex_t* source_vertex,
                                                  vertex_t n_subgraphs,
                                                  vertex_t radius)
{
  if (graph_container.edgeType == numberTypeEnum::int32Type) {
    auto graph =
      detail::create_graph<int32_t, int32_t, weight_t, false, false>(handle, graph_container);
    auto g = cugraph::experimental::extract_ego(handle,
                                                graph->view(),
                                                reinterpret_cast<int32_t*>(source_vertex),
                                                static_cast<int32_t>(n_subgraphs),
                                                static_cast<int32_t>(radius));
    cy_multi_edgelists_t coo_contents{
      0,  // not used
      std::get<0>(g).size(),
      static_cast<size_t>(n_subgraphs),
      std::make_unique<rmm::device_buffer>(std::get<0>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<1>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<2>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<3>(g).release())};
    return std::make_unique<cy_multi_edgelists_t>(std::move(coo_contents));
  } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
    auto graph =
      detail::create_graph<vertex_t, int64_t, weight_t, false, false>(handle, graph_container);
    auto g = cugraph::experimental::extract_ego(handle,
                                                graph->view(),
                                                reinterpret_cast<vertex_t*>(source_vertex),
                                                static_cast<vertex_t>(n_subgraphs),
                                                static_cast<vertex_t>(radius));
    cy_multi_edgelists_t coo_contents{
      0,  // not used
      std::get<0>(g).size(),
      static_cast<size_t>(n_subgraphs),
      std::make_unique<rmm::device_buffer>(std::get<0>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<1>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<2>(g).release()),
      std::make_unique<rmm::device_buffer>(std::get<3>(g).release())};
    return std::make_unique<cy_multi_edgelists_t>(std::move(coo_contents));
  } else {
    CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
  }
}

// Wrapper for calling SSSP through a graph container
template <typename vertex_t, typename weight_t>
void call_sssp(raft::handle_t const& handle,
               graph_container_t const& graph_container,
               vertex_t* identifiers,
               weight_t* distances,
               vertex_t* predecessors,
               const vertex_t source_vertex)
{
  if (graph_container.graph_type == graphTypeEnum::GraphCSRViewFloat) {
    graph_container.graph_ptr_union.GraphCSRViewFloatPtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
    sssp(  // handle, TODO: clarify: no raft_handle_t? why?
      *(graph_container.graph_ptr_union.GraphCSRViewFloatPtr),
      reinterpret_cast<float*>(distances),
      reinterpret_cast<int32_t*>(predecessors),
      static_cast<int32_t>(source_vertex));
  } else if (graph_container.graph_type == graphTypeEnum::GraphCSRViewDouble) {
    graph_container.graph_ptr_union.GraphCSRViewDoublePtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
    sssp(  // handle, TODO: clarify: no raft_handle_t? why?
      *(graph_container.graph_ptr_union.GraphCSRViewDoublePtr),
      reinterpret_cast<double*>(distances),
      reinterpret_cast<int32_t*>(predecessors),
      static_cast<int32_t>(source_vertex));
  } else if (graph_container.graph_type == graphTypeEnum::graph_t) {
    if (graph_container.edgeType == numberTypeEnum::int32Type) {
      auto graph =
        detail::create_graph<int32_t, int32_t, weight_t, false, true>(handle, graph_container);
      cugraph::experimental::sssp(handle,
                                  graph->view(),
                                  reinterpret_cast<weight_t*>(distances),
                                  reinterpret_cast<int32_t*>(predecessors),
                                  static_cast<int32_t>(source_vertex));
    } else if (graph_container.edgeType == numberTypeEnum::int64Type) {
      auto graph =
        detail::create_graph<vertex_t, int64_t, weight_t, false, true>(handle, graph_container);
      cugraph::experimental::sssp(handle,
                                  graph->view(),
                                  reinterpret_cast<weight_t*>(distances),
                                  reinterpret_cast<vertex_t*>(predecessors),
                                  static_cast<vertex_t>(source_vertex));
    } else {
      CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
    }
  }
}

// wrapper for shuffling:
//
template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<major_minor_weights_t<vertex_t, weight_t>> call_shuffle(
  raft::handle_t const& handle,
  vertex_t*
    edgelist_major_vertices,  // [IN / OUT]: groupby_gpuid_and_shuffle_values() sorts in-place
  vertex_t* edgelist_minor_vertices,  // [IN / OUT]
  weight_t* edgelist_weights,         // [IN / OUT]
  edge_t num_edgelist_edges,
  bool is_hypergraph_partitioned)  // = false
{
  auto& comm = handle.get_comms();

  auto& row_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());

  auto& col_comm = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());

  std::unique_ptr<major_minor_weights_t<vertex_t, weight_t>> ptr_ret =
    std::make_unique<major_minor_weights_t<vertex_t, weight_t>>(handle);

  if (edgelist_weights != nullptr) {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices, edgelist_weights));

    std::forward_as_tuple(
      std::tie(ptr_ret->get_major(), ptr_ret->get_minor(), ptr_ret->get_weights()),
      std::ignore) =
      cugraph::experimental::groupby_gpuid_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>{
             is_hypergraph_partitioned,
             comm.get_size(),
             row_comm.get_size(),
             col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  } else {
    auto zip_edge = thrust::make_zip_iterator(
      thrust::make_tuple(edgelist_major_vertices, edgelist_minor_vertices));

    std::forward_as_tuple(std::tie(ptr_ret->get_major(), ptr_ret->get_minor()),
                          std::ignore) =
      cugraph::experimental::groupby_gpuid_and_shuffle_values(
        comm,  // handle.get_comms(),
        zip_edge,
        zip_edge + num_edgelist_edges,
        [key_func =
           cugraph::experimental::detail::compute_gpu_id_from_edge_t<vertex_t>{
             is_hypergraph_partitioned,
             comm.get_size(),
             row_comm.get_size(),
             col_comm.get_size()}] __device__(auto val) {
          return key_func(thrust::get<0>(val), thrust::get<1>(val));
        },
        handle.get_stream());
  }

  return ptr_ret;  // RVO-ed
}

// Wrapper for calling renumber_edeglist() inplace:
// TODO: check if return type needs further handling...
//
template <typename vertex_t, typename edge_t>
std::unique_ptr<renum_quad_t<vertex_t, edge_t>> call_renumber(
  raft::handle_t const& handle,
  vertex_t* shuffled_edgelist_major_vertices /* [INOUT] */,
  vertex_t* shuffled_edgelist_minor_vertices /* [INOUT] */,
  edge_t num_edgelist_edges,
  bool is_hypergraph_partitioned,
  bool do_expensive_check,
  bool multi_gpu)  // bc. cython cannot take non-type template params
{
  // caveat: return values have different types on the 2 branches below:
  //
  std::unique_ptr<renum_quad_t<vertex_t, edge_t>> p_ret =
    std::make_unique<renum_quad_t<vertex_t, edge_t>>(handle);

  if (multi_gpu) {
    std::tie(
      p_ret->get_dv(), p_ret->get_partition(), p_ret->get_num_vertices(), p_ret->get_num_edges()) =
      cugraph::experimental::renumber_edgelist<vertex_t, edge_t, true>(
        handle,
        shuffled_edgelist_major_vertices,
        shuffled_edgelist_minor_vertices,
        num_edgelist_edges,
        is_hypergraph_partitioned,
        do_expensive_check);
  } else {
    auto ret_f = cugraph::experimental::renumber_edgelist<vertex_t, edge_t, false>(
      handle,
      shuffled_edgelist_major_vertices,
      shuffled_edgelist_minor_vertices,
      num_edgelist_edges,
      do_expensive_check);

    auto tot_vertices = static_cast<vertex_t>(ret_f.size());

    p_ret->get_dv() = std::move(ret_f);
    cugraph::experimental::partition_t<vertex_t> part_sg(
      std::vector<vertex_t>{0, tot_vertices}, false, 1, 1, 0, 0);

    p_ret->get_partition() = std::move(part_sg);

    p_ret->get_num_vertices() = tot_vertices;
    p_ret->get_num_edges()    = num_edgelist_edges;
  }

  return p_ret;  // RVO-ed (copy ellision)
}

// Helper for setting up subcommunicators
void init_subcomms(raft::handle_t& handle, size_t row_comm_size)
{
  partition_2d::subcomm_factory_t<partition_2d::key_naming_t, int> subcomm_factory(handle,
                                                                                   row_comm_size);
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

template void call_pagerank(raft::handle_t const& handle,
                            graph_container_t const& graph_container,
                            int* identifiers,
                            float* p_pagerank,
                            int32_t personalization_subset_size,
                            int32_t* personalization_subset,
                            float* personalization_values,
                            double alpha,
                            double tolerance,
                            int64_t max_iter,
                            bool has_guess);

template void call_pagerank(raft::handle_t const& handle,
                            graph_container_t const& graph_container,
                            int* identifiers,
                            double* p_pagerank,
                            int32_t personalization_subset_size,
                            int32_t* personalization_subset,
                            double* personalization_values,
                            double alpha,
                            double tolerance,
                            int64_t max_iter,
                            bool has_guess);

template void call_pagerank(raft::handle_t const& handle,
                            graph_container_t const& graph_container,
                            int64_t* identifiers,
                            float* p_pagerank,
                            int64_t personalization_subset_size,
                            int64_t* personalization_subset,
                            float* personalization_values,
                            double alpha,
                            double tolerance,
                            int64_t max_iter,
                            bool has_guess);

template void call_pagerank(raft::handle_t const& handle,
                            graph_container_t const& graph_container,
                            int64_t* identifiers,
                            double* p_pagerank,
                            int64_t personalization_subset_size,
                            int64_t* personalization_subset,
                            double* personalization_values,
                            double alpha,
                            double tolerance,
                            int64_t max_iter,
                            bool has_guess);

template void call_katz_centrality(raft::handle_t const& handle,
                                   graph_container_t const& graph_container,
                                   int* identifiers,
                                   float* katz_centrality,
                                   double alpha,
                                   double beta,
                                   double tolerance,
                                   int64_t max_iter,
                                   bool has_guess,
                                   bool normalize);

template void call_katz_centrality(raft::handle_t const& handle,
                                   graph_container_t const& graph_container,
                                   int* identifiers,
                                   double* katz_centrality,
                                   double alpha,
                                   double beta,
                                   double tolerance,
                                   int64_t max_iter,
                                   bool has_guess,
                                   bool normalize);

template void call_katz_centrality(raft::handle_t const& handle,
                                   graph_container_t const& graph_container,
                                   int64_t* identifiers,
                                   float* katz_centrality,
                                   double alpha,
                                   double beta,
                                   double tolerance,
                                   int64_t max_iter,
                                   bool has_guess,
                                   bool normalize);

template void call_katz_centrality(raft::handle_t const& handle,
                                   graph_container_t const& graph_container,
                                   int64_t* identifiers,
                                   double* katz_centrality,
                                   double alpha,
                                   double beta,
                                   double tolerance,
                                   int64_t max_iter,
                                   bool has_guess,
                                   bool normalize);

template void call_bfs<int32_t, float>(raft::handle_t const& handle,
                                       graph_container_t const& graph_container,
                                       int32_t* identifiers,
                                       int32_t* distances,
                                       int32_t* predecessors,
                                       double* sp_counters,
                                       const int32_t start_vertex,
                                       bool directed);

template void call_bfs<int32_t, double>(raft::handle_t const& handle,
                                        graph_container_t const& graph_container,
                                        int32_t* identifiers,
                                        int32_t* distances,
                                        int32_t* predecessors,
                                        double* sp_counters,
                                        const int32_t start_vertex,
                                        bool directed);

template void call_bfs<int64_t, float>(raft::handle_t const& handle,
                                       graph_container_t const& graph_container,
                                       int64_t* identifiers,
                                       int64_t* distances,
                                       int64_t* predecessors,
                                       double* sp_counters,
                                       const int64_t start_vertex,
                                       bool directed);

template void call_bfs<int64_t, double>(raft::handle_t const& handle,
                                        graph_container_t const& graph_container,
                                        int64_t* identifiers,
                                        int64_t* distances,
                                        int64_t* predecessors,
                                        double* sp_counters,
                                        const int64_t start_vertex,
                                        bool directed);
template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int32_t, float>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t* source_vertex,
  int32_t n_subgraphs,
  int32_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int32_t, double>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int32_t* source_vertex,
  int32_t n_subgraphs,
  int32_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int64_t, float>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int64_t* source_vertex,
  int64_t n_subgraphs,
  int64_t radius);

template std::unique_ptr<cy_multi_edgelists_t> call_egonet<int64_t, double>(
  raft::handle_t const& handle,
  graph_container_t const& graph_container,
  int64_t* source_vertex,
  int64_t n_subgraphs,
  int64_t radius);

template void call_sssp(raft::handle_t const& handle,
                        graph_container_t const& graph_container,
                        int32_t* identifiers,
                        float* distances,
                        int32_t* predecessors,
                        const int32_t source_vertex);

template void call_sssp(raft::handle_t const& handle,
                        graph_container_t const& graph_container,
                        int32_t* identifiers,
                        double* distances,
                        int32_t* predecessors,
                        const int32_t source_vertex);

template void call_sssp(raft::handle_t const& handle,
                        graph_container_t const& graph_container,
                        int64_t* identifiers,
                        float* distances,
                        int64_t* predecessors,
                        const int64_t source_vertex);

template void call_sssp(raft::handle_t const& handle,
                        graph_container_t const& graph_container,
                        int64_t* identifiers,
                        double* distances,
                        int64_t* predecessors,
                        const int64_t source_vertex);

template std::unique_ptr<major_minor_weights_t<int32_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

template std::unique_ptr<major_minor_weights_t<int32_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

template std::unique_ptr<major_minor_weights_t<int32_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int32_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

template std::unique_ptr<major_minor_weights_t<int32_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int32_t* edgelist_major_vertices,
  int32_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

template std::unique_ptr<major_minor_weights_t<int64_t, float>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  float* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

template std::unique_ptr<major_minor_weights_t<int64_t, double>> call_shuffle(
  raft::handle_t const& handle,
  int64_t* edgelist_major_vertices,
  int64_t* edgelist_minor_vertices,
  double* edgelist_weights,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned);

// TODO: add the remaining relevant EIDIr's:
//
template std::unique_ptr<renum_quad_t<int32_t, int32_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_major_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_minor_vertices /* [INOUT] */,
  int32_t num_edgelist_edges,
  bool is_hypergraph_partitioned,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_quad_t<int32_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int32_t* shuffled_edgelist_major_vertices /* [INOUT] */,
  int32_t* shuffled_edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned,
  bool do_expensive_check,
  bool multi_gpu);

template std::unique_ptr<renum_quad_t<int64_t, int64_t>> call_renumber(
  raft::handle_t const& handle,
  int64_t* shuffled_edgelist_major_vertices /* [INOUT] */,
  int64_t* shuffled_edgelist_minor_vertices /* [INOUT] */,
  int64_t num_edgelist_edges,
  bool is_hypergraph_partitioned,
  bool do_expensive_check,
  bool multi_gpu);

}  // namespace cython
}  // namespace cugraph
