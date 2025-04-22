/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "c_api/capi_helper.hpp"
#include "c_api/graph.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_layout_result_t {
  cugraph_type_erased_device_array_t* vertices_{nullptr};
  cugraph_type_erased_device_array_t* x_axis_{nullptr};
  cugraph_type_erased_device_array_t* y_axis_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct minimum_spanning_tree_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_layout_result_t* result_{};
  ;

  minimum_spanning_tree_functor(::cugraph_resource_handle_t const* handle,
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
      unsupported();
    } else if constexpr (!std::is_same_v<edge_t, int32_t>) {
      unsupported();
    } else {
      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, false>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, false>, weight_t>*>(
        graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view          = graph->view();
      auto edge_partition_view = graph_view.local_edge_partition_view();

      rmm::device_uvector<weight_t> tmp_weights(0, handle_.get_stream());
      if (edge_weights == nullptr) {
        tmp_weights.resize(edge_partition_view.indices().size(), handle_.get_stream());
        cugraph::detail::scalar_fill(handle_, tmp_weights.data(), tmp_weights.size(), weight_t{1});
      }

      cugraph::legacy::GraphCSRView<vertex_t, edge_t, weight_t> legacy_graph_view(
        const_cast<edge_t*>(edge_partition_view.offsets().data()),
        const_cast<vertex_t*>(edge_partition_view.indices().data()),
        (edge_weights == nullptr)
          ? tmp_weights.data()
          : const_cast<weight_t*>(edge_weights->view().value_firsts().front()),
        edge_partition_view.offsets().size() - 1,
        edge_partition_view.indices().size());

     
      

      auto x = cugraph::minimum_spanning_tree<vertex_t, edge_t, weight_t>(
        handle_,
        legacy_graph_view
        );
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_vertices(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_x_axis(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->x_axis_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_layout_result_get_y_axis(
  cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->y_axis_->view());
}

extern "C" void cugraph_layout_result_free(cugraph::c_api::cugraph_layout_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_layout_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->x_axis_;
  delete internal_pointer->y_axis_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_minimum_spanning_tree(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph::c_api::cugraph_layout_result_t** result,
  cugraph_error_t** error)
{

  minimum_spanning_tree_functor functor(handle,
                                        graph,
                                        do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
