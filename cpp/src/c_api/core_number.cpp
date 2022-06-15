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
#include <cugraph_c/core_algorithms.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_core_result_t {
  cugraph_type_erased_device_array_t* vertices_;
  cugraph_type_erased_device_array_t* core_numbers_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct core_number_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_k_core_degree_type_t degree_type_;
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_core_result_t* result_{};

  core_number_functor(cugraph_resource_handle_t const* handle,
                      cugraph_graph_t* graph,
                      cugraph_k_core_degree_type_t degree_type_,
                      bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      // try casting first, if not look for how to cast between enum types
      degree_type_(reinterpret_cast<cugraph::c_api::cugraph_k_core_degree_type_t>(degree_type)),
      do_expensive_check_(do_expensive_check)
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      if constexpr (!store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS)
          ;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<edge_t> core_numbers(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());

      cugraph::core_number<vertex_t, edge_t, weight_t, multi_gpu>(
        // cugraph::core_number(
        handle_,
        graph_view,
        core_numbers.data(),
        degree_type_,
        // k_first,
        // k_last,
        do_expensive_check_);

      // do some stuff here
      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_core_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(core_numbers, graph_->edge_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_core_result_get_vertices(
  cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_core_result_get_core_numbers(
  cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->core_numbers_->view());
}

extern "C" void cugraph_core_result_free(cugraph_core_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_core_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->core_numbers_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_core_number(const cugraph_resource_handle_t* handle,
                                                    cugraph_graph_t* graph,
                                                    const cugraph_k_core_degree_type_t degree_type,
                                                    bool_t do_expensive_check,
                                                    cugraph_core_result_t** result,
                                                    cugraph_error_t** error)
{
  core_number_functor functor(handle, graph, degree_type, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}