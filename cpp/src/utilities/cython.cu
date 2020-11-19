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

  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();  // pcols
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();  // prows

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
  graph_container.row_comm_size            = row_comm_size;
  graph_container.col_comm_size            = col_comm_size;
  graph_container.row_comm_rank            = row_comm_rank;
  graph_container.col_comm_rank            = col_comm_rank;
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
  if (graph_container.graph_type == graphTypeEnum::GraphCSCViewFloat) {
    pagerank(handle,
             *(graph_container.graph_ptr_union.GraphCSCViewFloatPtr),
             reinterpret_cast<float*>(p_pagerank),
             static_cast<int32_t>(personalization_subset_size),
             reinterpret_cast<int32_t*>(personalization_subset),
             reinterpret_cast<float*>(personalization_values),
             alpha,
             tolerance,
             max_iter,
             has_guess);
    graph_container.graph_ptr_union.GraphCSCViewFloatPtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
  } else if (graph_container.graph_type == graphTypeEnum::GraphCSCViewDouble) {
    pagerank(handle,
             *(graph_container.graph_ptr_union.GraphCSCViewDoublePtr),
             reinterpret_cast<double*>(p_pagerank),
             static_cast<int32_t>(personalization_subset_size),
             reinterpret_cast<int32_t*>(personalization_subset),
             reinterpret_cast<double*>(personalization_values),
             alpha,
             tolerance,
             max_iter,
             has_guess);
    graph_container.graph_ptr_union.GraphCSCViewDoublePtr->get_vertex_identifiers(
      reinterpret_cast<int32_t*>(identifiers));
  } else if (graph_container.graph_type == graphTypeEnum::graph_t) {
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
                                      false);
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
                                      false);
    } else {
      CUGRAPH_FAIL("vertexType/edgeType combination unsupported");
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

}  // namespace cython
}  // namespace cugraph
