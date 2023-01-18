/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cugraph_c/algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/paths_result.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace cugraph {
namespace c_api {

struct bfs_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t* sources_;
  bool direction_optimizing_;
  size_t depth_limit_;
  bool compute_predecessors_;
  bool do_expensive_check_;
  cugraph_paths_result_t* result_{};

  bfs_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_graph_t* graph,
              ::cugraph_type_erased_device_array_view_t* sources,
              bool direction_optimizing,
              size_t depth_limit,
              bool compute_predecessors,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t*>(sources)),
      direction_optimizing_(direction_optimizing),
      depth_limit_(depth_limit),
      compute_predecessors_(compute_predecessors),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // BFS expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      rmm::device_uvector<vertex_t> sources(sources_->size_, handle_.get_stream());
      raft::copy(
        sources.data(), sources_->as_type<vertex_t>(), sources_->size_, handle_.get_stream());

      if constexpr (multi_gpu) {
        sources = detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
          handle_, std::move(sources));
      }

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 sources.data(),
                                                 sources.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 do_expensive_check_);

      cugraph::bfs<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        sources.data(),
        sources.size(),
        direction_optimizing_,
        static_cast<vertex_t>(depth_limit_),
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      if (compute_predecessors_) {
        std::vector<vertex_t> vertex_partition_range_lasts =
          graph_view.vertex_partition_range_lasts();

        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                     predecessors.data(),
                                                     predecessors.size(),
                                                     number_map->data(),
                                                     vertex_partition_range_lasts,
                                                     do_expensive_check_);
      }

      result_ = new cugraph_paths_result_t{
        new cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(distances, graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(predecessors, graph_->vertex_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_vertices(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_distances(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->distances_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_predecessors(
  cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->predecessors_->view());
}

extern "C" void cugraph_paths_result_free(cugraph_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_paths_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->distances_;
  delete internal_pointer->predecessors_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                            cugraph_graph_t* graph,
                                            cugraph_type_erased_device_array_view_t* sources,
                                            bool_t direction_optimizing,
                                            size_t depth_limit,
                                            bool_t compute_predecessors,
                                            bool_t do_expensive_check,
                                            cugraph_paths_result_t** result,
                                            cugraph_error_t** error)
{
  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and sources must match",
    *error);

  cugraph::c_api::bfs_functor functor(handle,
                                      graph,
                                      sources,
                                      direction_optimizing,
                                      depth_limit,
                                      compute_predecessors,
                                      do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
