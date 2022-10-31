/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace cugraph {
namespace c_api {

struct sssp_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  size_t source_;
  double cutoff_;
  bool compute_predecessors_;
  bool do_expensive_check_;
  cugraph_paths_result_t* result_{};

  sssp_functor(::cugraph_resource_handle_t const* handle,
               ::cugraph_graph_t* graph,
               size_t source,
               double cutoff,
               bool compute_predecessors,
               bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      source_(source),
      cutoff_(cutoff),
      compute_predecessors_(compute_predecessors),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_type_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // SSSP expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> source_ids(1, handle_.get_stream());
      rmm::device_uvector<weight_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle_.get_stream());
      rmm::device_uvector<vertex_t> predecessors(0, handle_.get_stream());

      if (compute_predecessors_) {
        predecessors.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      vertex_t src = static_cast<vertex_t>(source_);
      raft::update_device(source_ids.data(), &src, 1, handle_.get_stream());

      //
      // Need to renumber sources
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 source_ids.data(),
                                                 source_ids.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 do_expensive_check_);

      raft::update_host(&src, source_ids.data(), 1, handle_.get_stream());

      cugraph::sssp<vertex_t, edge_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        edge_weights->view(),
        distances.data(),
        compute_predecessors_ ? predecessors.data() : nullptr,
        src,
        static_cast<weight_t>(cutoff_),
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
        new cugraph_type_erased_device_array_t(distances, graph_->weight_type_),
        new cugraph_type_erased_device_array_t(predecessors, graph_->vertex_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" cugraph_error_code_t cugraph_sssp(const cugraph_resource_handle_t* handle,
                                             cugraph_graph_t* graph,
                                             size_t source,
                                             double cutoff,
                                             bool_t compute_predecessors,
                                             bool_t do_expensive_check,
                                             cugraph_paths_result_t** result,
                                             cugraph_error_t** error)
{
  cugraph::c_api::sssp_functor functor(
    handle, graph, source, cutoff, compute_predecessors, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
