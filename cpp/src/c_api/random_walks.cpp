/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace cugraph {
namespace c_api {

struct cugraph_random_walk_result_t {
  bool result_compressed_{false};
  size_t max_path_length_{0};
  cugraph_type_erased_device_array_t* paths_{nullptr};
  cugraph_type_erased_device_array_t* weights_{nullptr};
  cugraph_type_erased_device_array_t* sizes_{nullptr};
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct uniform_random_walks_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  size_t max_length_{0};
  cugraph::c_api::cugraph_random_walk_result_t* result_{nullptr};

  uniform_random_walks_functor(cugraph_resource_handle_t const* handle,
                               cugraph_rng_state_t* rng_state,
                               cugraph_graph_t* graph,
                               cugraph_type_erased_device_array_view_t const* start_vertices,
                               size_t max_length)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      max_length_(max_length)
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
      // random walks expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      auto [paths, weights] = cugraph::uniform_random_walks(
        handle_,
        rng_state_->rng_state_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
        max_length_);

      //
      // Need to unrenumber the vertices in the resulting paths
      //
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        paths.data(),
        paths.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        false);

      result_ = new cugraph::c_api::cugraph_random_walk_result_t{
        false,
        max_length_,
        new cugraph::c_api::cugraph_type_erased_device_array_t(paths, graph_->vertex_type_),
        weights
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*weights, graph_->weight_type_)
          : nullptr,
        nullptr};
    }
  }
};

struct biased_random_walks_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  size_t max_length_{0};
  cugraph::c_api::cugraph_random_walk_result_t* result_{nullptr};

  biased_random_walks_functor(cugraph_resource_handle_t const* handle,
                              cugraph_rng_state_t* rng_state,
                              cugraph_graph_t* graph,
                              cugraph_type_erased_device_array_view_t const* start_vertices,
                              size_t max_length)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      max_length_(max_length)
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
      // random walks expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      auto [paths, weights] = cugraph::biased_random_walks(
        handle_,
        rng_state_->rng_state_,
        graph_view,
        edge_weights->view(),
        raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
        max_length_);

      //
      // Need to unrenumber the vertices in the resulting paths
      //
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        paths.data(),
        paths.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        false);

      result_ = new cugraph::c_api::cugraph_random_walk_result_t{
        false,
        max_length_,
        new cugraph::c_api::cugraph_type_erased_device_array_t(paths, graph_->vertex_type_),
        weights
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*weights, graph_->weight_type_)
          : nullptr,
        nullptr};
    }
  }
};

struct node2vec_random_walks_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_rng_state_t* rng_state_{nullptr};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  size_t max_length_{0};
  double p_{0};
  double q_{0};
  cugraph::c_api::cugraph_random_walk_result_t* result_{nullptr};

  node2vec_random_walks_functor(cugraph_resource_handle_t const* handle,
                                cugraph_rng_state_t* rng_state,
                                cugraph_graph_t* graph,
                                cugraph_type_erased_device_array_view_t const* start_vertices,
                                size_t max_length,
                                double p,
                                double q)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      rng_state_(reinterpret_cast<cugraph::c_api::cugraph_rng_state_t*>(rng_state)),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
      max_length_(max_length),
      p_(p),
      q_(q)
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
      // random walks expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights =
        reinterpret_cast<cugraph::edge_property_t<edge_t, weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> start_vertices(start_vertices_->size_, handle_.get_stream());
      raft::copy(start_vertices.data(),
                 start_vertices_->as_type<vertex_t>(),
                 start_vertices.size(),
                 handle_.get_stream());

      //
      // Need to renumber start_vertices
      //
      cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        start_vertices.data(),
        start_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        false);

      auto [paths, weights] = cugraph::node2vec_random_walks(
        handle_,
        rng_state_->rng_state_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
        max_length_,
        static_cast<weight_t>(p_),
        static_cast<weight_t>(q_));

      // FIXME:  Need to fix invalid_vtx issue here.  We can't unrenumber max_vertex_id+1
      // properly...
      //   So if the result includes an invalid vertex we don't handle it properly.

      //
      // Need to unrenumber the vertices in the resulting paths
      //
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        paths.data(),
        paths.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        false);

      result_ = new cugraph::c_api::cugraph_random_walk_result_t{
        false,
        max_length_,
        new cugraph::c_api::cugraph_type_erased_device_array_t(paths, graph_->vertex_type_),
        weights
          ? new cugraph::c_api::cugraph_type_erased_device_array_t(*weights, graph_->weight_type_)
          : nullptr,
        nullptr};
    }
  }
};

}  // anonymous namespace

size_t cugraph_random_walk_result_get_max_path_length(cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return internal_pointer->max_path_length_;
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_paths(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->paths_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_weights(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->weights_->view());
}

cugraph_type_erased_device_array_view_t* cugraph_random_walk_result_get_path_sizes(
  cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->sizes_->view());
}

void cugraph_random_walk_result_free(cugraph_random_walk_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_random_walk_result_t*>(result);
  delete internal_pointer->paths_;
  delete internal_pointer->sizes_;
  delete internal_pointer->weights_;
  delete internal_pointer;
}

cugraph_error_code_t cugraph_uniform_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  uniform_random_walks_functor functor(handle, rng_state, graph, start_vertices, max_length);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_biased_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  biased_random_walks_functor functor(handle, rng_state, graph, start_vertices, max_length);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

cugraph_error_code_t cugraph_node2vec_random_walks(
  const cugraph_resource_handle_t* handle,
  cugraph_rng_state_t* rng_state,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  size_t max_length,
  double p,
  double q,
  cugraph_random_walk_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   start_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and start_vertices must match",
               *error);

  node2vec_random_walks_functor functor(handle, rng_state, graph, start_vertices, max_length, p, q);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
