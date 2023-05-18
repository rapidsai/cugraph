/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph_c/labeling_algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/labeling_result.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct scc_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_labeling_result_t* result_{};

  scc_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_graph_t* graph,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
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
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else if constexpr (multi_gpu) {
      error_code_ = CUGRAPH_NOT_IMPLEMENTED;
      error_->error_message_ =
        "strongly connected components not currently implemented for multi-GPU";

    } else if constexpr (!std::is_same_v<vertex_t, edge_t>) {
      unsupported();
    } else if constexpr (std::is_same_v<weight_t, double>) {
      unsupported();
    } else {
      // SCC expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        nullptr,
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      rmm::device_uvector<vertex_t> components(graph_view.number_of_vertices(),
                                               handle_.get_stream());

      cugraph::connected_components(
        legacy_graph_view, cugraph::cugraph_cc_t::CUGRAPH_STRONG, components.data());
      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_labeling_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(components, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_strongly_connected_components(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  bool_t do_expensive_check,
  cugraph_labeling_result_t** result,
  cugraph_error_t** error)
{
  scc_functor functor(handle, graph, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
