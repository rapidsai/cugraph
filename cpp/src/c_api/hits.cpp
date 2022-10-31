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

struct cugraph_hits_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* hubs_;
  cugraph_type_erased_device_array_t* authorities_;
  double hub_score_differences_;
  size_t number_of_iterations_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct hits_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  double epsilon_;
  size_t max_iterations_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* initial_hubs_guess_vertices_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* initial_hubs_guess_values_;
  bool normalize_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_hits_result_t* result_{};

  hits_functor(::cugraph_resource_handle_t const* handle,
               ::cugraph_graph_t* graph,
               double epsilon,
               size_t max_iterations,
               ::cugraph_type_erased_device_array_view_t const* initial_hubs_guess_vertices,
               ::cugraph_type_erased_device_array_view_t const* initial_hubs_guess_values,
               bool normalize,
               bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      epsilon_(epsilon),
      max_iterations_(max_iterations),
      initial_hubs_guess_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          initial_hubs_guess_vertices)),
      initial_hubs_guess_values_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          initial_hubs_guess_values)),
      normalize_(normalize),
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
      // HITS expects store_transposed == true
      if constexpr (!store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, true, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<weight_t> hubs(graph_view.local_vertex_partition_range_size(),
                                         handle_.get_stream());
      rmm::device_uvector<weight_t> authorities(graph_view.local_vertex_partition_range_size(),
                                                handle_.get_stream());
      weight_t hub_score_differences{0};
      size_t number_of_iterations{0};

      if (initial_hubs_guess_vertices_ != nullptr) {
        rmm::device_uvector<vertex_t> guess_vertices(initial_hubs_guess_vertices_->size_,
                                                     handle_.get_stream());
        rmm::device_uvector<weight_t> guess_values(initial_hubs_guess_values_->size_,
                                                   handle_.get_stream());

        raft::copy(guess_vertices.data(),
                   initial_hubs_guess_vertices_->as_type<vertex_t>(),
                   guess_vertices.size(),
                   handle_.get_stream());
        raft::copy(guess_values.data(),
                   initial_hubs_guess_values_->as_type<weight_t>(),
                   guess_values.size(),
                   handle_.get_stream());

        hubs = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, weight_t, multi_gpu>(
            handle_,
            std::move(guess_vertices),
            std::move(guess_values),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            weight_t{0},
            do_expensive_check_);
      }

      std::tie(hub_score_differences, number_of_iterations) =
        cugraph::hits<vertex_t, edge_t, weight_t, multi_gpu>(
          handle_,
          graph_view,
          hubs.data(),
          authorities.data(),
          epsilon_,
          max_iterations_,
          (initial_hubs_guess_vertices_ != nullptr),
          normalize_,
          do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_hits_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(hubs, graph_->weight_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(authorities, graph_->weight_type_),
        hub_score_differences,
        number_of_iterations};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_vertices(
  cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertex_ids_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_hubs(
  cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->hubs_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_hits_result_get_authorities(
  cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->authorities_->view());
}

extern "C" double cugraph_hits_result_get_hub_score_differences(cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  return internal_pointer->hub_score_differences_;
}

extern "C" size_t cugraph_hits_result_get_number_of_iterations(cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  return internal_pointer->number_of_iterations_;
}

extern "C" void cugraph_hits_result_free(cugraph_hits_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_hits_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->hubs_;
  delete internal_pointer->authorities_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_hits(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  double epsilon,
  size_t max_iterations,
  const cugraph_type_erased_device_array_view_t* initial_hubs_guess_vertices,
  const cugraph_type_erased_device_array_view_t* initial_hubs_guess_values,
  bool_t normalize,
  bool_t do_expensive_check,
  cugraph_hits_result_t** result,
  cugraph_error_t** error)
{
  hits_functor functor(handle,
                       graph,
                       epsilon,
                       max_iterations,
                       initial_hubs_guess_vertices,
                       initial_hubs_guess_values,
                       normalize,
                       do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
