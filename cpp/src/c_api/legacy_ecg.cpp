/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/hierarchical_clustering_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct legacy_ecg_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  double min_weight_;
  size_t ensemble_size_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_hierarchical_clustering_result_t* result_{};

  legacy_ecg_functor(::cugraph_resource_handle_t const* handle,
                     ::cugraph_graph_t* graph,
                     double min_weight,
                     size_t ensemble_size,
                     bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      min_weight_(min_weight),
      ensemble_size_(ensemble_size),
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
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      // ecg expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view = graph->view();

      auto edge_partition_view = graph_view.local_edge_partition_view();

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

      rmm::device_uvector<vertex_t> clusters(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());

      // FIXME:  Need modularity..., although currently not used
      cugraph::ecg(handle_,
                   legacy_graph_view,
                   static_cast<weight_t>(min_weight_),
                   static_cast<vertex_t>(ensemble_size_),
                   clusters.data());

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_hierarchical_clustering_result_t{
        weight_t{0},
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(clusters, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_legacy_ecg(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  double min_weight,
  size_t ensemble_size,
  bool_t do_expensive_check,
  cugraph_hierarchical_clustering_result_t** result,
  cugraph_error_t** error)
{
  legacy_ecg_functor functor(handle, graph, min_weight, ensemble_size, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
