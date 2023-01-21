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
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_heirarchical_clustering_result_t {
  double modularity{0};
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* clusters_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct louvain_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  size_t max_level_;
  double resolution_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_heirarchical_clustering_result_t* result_{};

  louvain_functor(::cugraph_resource_handle_t const* handle,
                  ::cugraph_graph_t* graph,
                  size_t max_level,
                  double resolution,
                  bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      max_level_(max_level),
      resolution_(resolution),
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
    } else {
      // louvain expects store_transposed == false
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

      rmm::device_uvector<vertex_t> clusters(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());

      auto [level, modularity] = cugraph::louvain(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        clusters.data(),
        max_level_,
        static_cast<weight_t>(resolution_));

      rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                             handle_.get_stream());
      raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_heirarchical_clustering_result_t{
        modularity,
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(clusters, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_heirarchical_clustering_result_get_vertices(
  cugraph_heirarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_heirarchical_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t*
cugraph_heirarchical_clustering_result_get_clusters(
  cugraph_heirarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_heirarchical_clustering_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->clusters_->view());
}

extern "C" double cugraph_heirarchical_clustering_result_get_modularity(
  cugraph_heirarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_heirarchical_clustering_result_t*>(result);
  return internal_pointer->modularity;
}

extern "C" void cugraph_heirarchical_clustering_result_free(
  cugraph_heirarchical_clustering_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_heirarchical_clustering_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->clusters_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_louvain(const cugraph_resource_handle_t* handle,
                                                cugraph_graph_t* graph,
                                                size_t max_level,
                                                double resolution,
                                                bool_t do_expensive_check,
                                                cugraph_heirarchical_clustering_result_t** result,
                                                cugraph_error_t** error)
{
  louvain_functor functor(handle, graph, max_level, resolution, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
