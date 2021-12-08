/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>

#include <raft/handle.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_bfs_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* distances_;
  cugraph_type_erased_device_array_t* predecessors_;
};

struct bfs_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_t* sources_;
  bool direction_optimizing_;
  size_t depth_limit_;
  bool do_expensive_check_;
  bool compute_predecessors_;
  cugraph_bfs_result_t* result_{};

  bfs_functor(raft::handle_t const& handle,
              cugraph_graph_t* graph,
              cugraph_type_erased_device_array_t* sources,
              bool direction_optimizing,
              size_t depth_limit,
              bool do_expensive_check,
              bool compute_predecessors)
    : abstract_functor(),
      handle_(handle),
      graph_(graph),
      sources_(sources),
      direction_optimizing_(direction_optimizing),
      depth_limit_(depth_limit),
      do_expensive_check_(do_expensive_check),
      compute_predecessors_(compute_predecessors)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
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
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> vertex_ids(graph->get_number_of_vertices(),
                                               handle_.get_stream());
      rmm::device_uvector<vertex_t> distances(graph->get_number_of_vertices(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph->get_number_of_vertices(), handle_.get_stream());
      }

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 sources_->as_type<vertex_t>(),
                                                 sources_->size_,
                                                 number_map->data(),
                                                 graph_view.get_local_vertex_first(),
                                                 graph_view.get_local_vertex_last(),
                                                 do_expensive_check_);

      cugraph::bfs<vertex_t, edge_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        sources_->as_type<vertex_t>(),
        sources_->size_,
        direction_optimizing_,
        static_cast<vertex_t>(depth_limit_),
        do_expensive_check_);

      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      if (compute_predecessors_) {
        std::vector<vertex_t> vertex_partition_lasts = graph_view.get_vertex_partition_lasts();

        unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                     predecessors.data(),
                                                     predecessors.size(),
                                                     number_map->data(),
                                                     vertex_partition_lasts,
                                                     do_expensive_check_);
      }

      result_ = new cugraph_bfs_result_t{
        new cugraph_type_erased_device_array_t(std::move(vertex_ids), graph_->vertex_type_),
        new cugraph_type_erased_device_array_t(std::move(distances), graph_->weight_type_),
        new cugraph_type_erased_device_array_t(std::move(predecessors), graph_->weight_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_vertices(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_bfs_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_t*>(internal_pointer->vertex_ids_);
}

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_distances(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_bfs_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_t*>(internal_pointer->distances_);
}

extern "C" cugraph_type_erased_device_array_t* cugraph_bfs_result_get_predecessors(
  cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_bfs_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_t*>(internal_pointer->predecessors_);
}

extern "C" void cugraph_bfs_result_free(cugraph_bfs_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_bfs_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->distances_;
  delete internal_pointer->predecessors_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_bfs(const cugraph_resource_handle_t* handle,
                                            cugraph_graph_t* graph,
                                            cugraph_type_erased_device_array_t* sources,
                                            bool_t direction_optimizing,
                                            size_t depth_limit,
                                            bool_t do_expensive_check,
                                            bool_t compute_predecessors,
                                            cugraph_bfs_result_t** result,
                                            cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;

  try {
    auto p_handle  = reinterpret_cast<raft::handle_t const*>(handle);
    auto p_graph   = reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph);
    auto p_sources = reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_t*>(sources);

    cugraph::c_api::bfs_functor functor(*p_handle,
                                        p_graph,
                                        p_sources,
                                        direction_optimizing,
                                        depth_limit,
                                        do_expensive_check,
                                        compute_predecessors);

    // FIXME:  This seems like a recurring pattern.  Can I encapsulate
    //    The vertex_dispatcher and error handling calls into a reusable function?
    //    After all, we're in C++ here.
    cugraph::dispatch::vertex_dispatcher(cugraph::c_api::dtypes_mapping[p_graph->vertex_type_],
                                         cugraph::c_api::dtypes_mapping[p_graph->edge_type_],
                                         cugraph::c_api::dtypes_mapping[p_graph->weight_type_],
                                         p_graph->store_transposed_,
                                         p_graph->multi_gpu_,
                                         functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *result = reinterpret_cast<cugraph_bfs_result_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}
