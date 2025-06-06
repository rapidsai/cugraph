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
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/shuffle_functions.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_triangle_count_result_t {
  cugraph_type_erased_device_array_t* vertices_;
  cugraph_type_erased_device_array_t* counts_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {

struct triangle_count_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* vertices_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_triangle_count_result_t* result_{};

  triangle_count_functor(::cugraph_resource_handle_t const* handle,
                         ::cugraph_graph_t* graph,
                         ::cugraph_type_erased_device_array_view_t const* vertices,
                         bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(vertices)),
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
      // triangle counting expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> vertices(0, handle_.get_stream());
      rmm::device_uvector<edge_t> counts(0, handle_.get_stream());

      if (vertices_ != nullptr) {
        vertices.resize(vertices_->size_, handle_.get_stream());

        raft::copy(
          vertices.data(), vertices_->as_type<vertex_t>(), vertices.size(), handle_.get_stream());

        if constexpr (multi_gpu) {
          vertices = cugraph::shuffle_ext_vertices(handle_, std::move(vertices));
        }

        counts.resize(vertices.size(), handle_.get_stream());

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          vertices.data(),
          vertices.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);
      } else {
        counts.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
      }

      cugraph::triangle_count<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        vertices_ == nullptr
          ? std::nullopt
          : std::make_optional(raft::device_span<vertex_t>{vertices.data(), vertices.size()}),
        raft::device_span<edge_t>{counts.data(), counts.size()},
        do_expensive_check_);

      if (vertices_ == nullptr) {
        vertices.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
        raft::copy(vertices.data(), number_map->data(), vertices.size(), handle_.get_stream());
      } else {
        cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
          handle_,
          vertices.data(),
          vertices.size(),
          number_map->data(),
          graph_view.vertex_partition_range_lasts(),
          do_expensive_check_);
      }

      result_ = new cugraph::c_api::cugraph_triangle_count_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertices, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(counts, graph_->edge_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_triangle_count_result_get_vertices(
  cugraph_triangle_count_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_triangle_count_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->vertices_->view());
}

extern "C" cugraph_type_erased_device_array_view_t* cugraph_triangle_count_result_get_counts(
  cugraph_triangle_count_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_triangle_count_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->counts_->view());
}

extern "C" void cugraph_triangle_count_result_free(cugraph_triangle_count_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_triangle_count_result_t*>(result);
  delete internal_pointer->vertices_;
  delete internal_pointer->counts_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_triangle_count(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start,
  bool_t do_expensive_check,
  cugraph_triangle_count_result_t** result,
  cugraph_error_t** error)
{
  triangle_count_functor functor(handle, graph, start, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
