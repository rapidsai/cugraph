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

#include <cugraph_c/graph_functions.h>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/graph_functions.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace {

struct create_vertex_pairs_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* first_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* second_;
  cugraph::c_api::cugraph_vertex_pairs_t* result_{};

  create_vertex_pairs_functor(::cugraph_resource_handle_t const* handle,
                              ::cugraph_graph_t* graph,
                              ::cugraph_type_erased_device_array_view_t const* first,
                              ::cugraph_type_erased_device_array_view_t const* second)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      first_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(first)),
      second_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(second))
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
      rmm::device_uvector<vertex_t> first_copy(first_->size_, handle_.get_stream());
      rmm::device_uvector<vertex_t> second_copy(second_->size_, handle_.get_stream());

      raft::copy(
        first_copy.data(), first_->as_type<vertex_t>(), first_->size_, handle_.get_stream());
      raft::copy(
        second_copy.data(), second_->as_type<vertex_t>(), second_->size_, handle_.get_stream());

      if constexpr (multi_gpu) {
        // FIXME: shuffle first_copy/second_copy
      }

      result_ = new cugraph::c_api::cugraph_vertex_pairs_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(first_copy, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(second_copy, graph_->vertex_type_)};
    }
  }
};

struct two_hop_neighbors_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t const* graph_;
  cugraph::c_api::cugraph_vertex_pairs_t* result_{};

  two_hop_neighbors_functor(::cugraph_resource_handle_t const* handle,
                            ::cugraph_graph_t const* graph)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph))
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
      auto graph = reinterpret_cast<
        cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      CUGRAPH_FAIL("Not implemented");
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_create_vertex_pairs(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* first,
  const cugraph_type_erased_device_array_view_t* second,
  cugraph_vertex_pairs_t** vertex_pairs,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(first)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and first must match",
    *error);

  CAPI_EXPECTS(
    reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
      reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(second)
        ->type_,
    CUGRAPH_INVALID_INPUT,
    "vertex type of graph and second must match",
    *error);

  create_vertex_pairs_functor functor(handle, graph, first, second);

  return cugraph::c_api::run_algorithm(graph, functor, vertex_pairs, error);
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_vertex_pairs_get_first(
  cugraph_vertex_pairs_t* vertex_pairs)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_vertex_pairs_t*>(vertex_pairs);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->first_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_vertex_pairs_get_second(
  cugraph_vertex_pairs_t* vertex_pairs)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_vertex_pairs_t*>(vertex_pairs);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->second_->view());
}

extern "C" void cugraph_vertex_pairs_free(cugraph_vertex_pairs_t* vertex_pairs)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_vertex_pairs_t*>(vertex_pairs);
  delete internal_pointer->first_;
  delete internal_pointer->second_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_two_hop_neighbors(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  cugraph_vertex_pairs_t** result,
  cugraph_error_t** error)
{
  two_hop_neighbors_functor functor(handle, graph);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
