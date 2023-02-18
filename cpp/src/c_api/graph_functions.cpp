/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <c_api/graph_helper.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
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
            typename edge_type_type_t,
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
        std::tie(first_copy, second_copy, std::ignore, std::ignore) =
          cugraph::detail::shuffle_ext_vertex_pairs_to_local_gpu_by_edge_partitioning<
            vertex_t,
            edge_t,
            weight_t,
            edge_type_type_t>(
            handle_, std::move(first_copy), std::move(second_copy), std::nullopt, std::nullopt);
      }

      result_ = new cugraph::c_api::cugraph_vertex_pairs_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(first_copy, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(second_copy, graph_->vertex_type_)};
    }
  }
};

struct two_hop_neighbors_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_{};
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* start_vertices_{nullptr};
  cugraph::c_api::cugraph_vertex_pairs_t* result_{};
  bool do_expensive_check_{false};

  two_hop_neighbors_functor(::cugraph_resource_handle_t const* handle,
                            ::cugraph_graph_t* graph,
                            ::cugraph_type_erased_device_array_view_t const* start_vertices,
                            bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      start_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          start_vertices)),
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
      // k_hop_nbrs expects store_transposed == false
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

      rmm::device_uvector<vertex_t> start_vertices(0, handle_.get_stream());

      if (start_vertices_ != nullptr) {
        start_vertices.resize(start_vertices_->size_, handle_.get_stream());
        raft::copy(start_vertices.data(),
                   start_vertices_->as_type<vertex_t const>(),
                   start_vertices_->size_,
                   handle_.get_stream());

        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(
          handle_,
          start_vertices.data(),
          start_vertices.size(),
          number_map->data(),
          graph_view.local_vertex_partition_range_first(),
          graph_view.local_vertex_partition_range_last(),
          do_expensive_check_);

        if constexpr (multi_gpu) {
          start_vertices =
            cugraph::detail::shuffle_ext_vertices_to_local_gpu_by_vertex_partitioning(
              handle_, std::move(start_vertices));
        }
      } else {
        start_vertices.resize(graph_view.local_vertex_partition_range_size(), handle_.get_stream());
        cugraph::detail::sequence_fill(handle_.get_stream(),
                                       start_vertices.data(),
                                       start_vertices.size(),
                                       graph_view.local_vertex_partition_range_first());
      }

      auto [offsets, dst] = cugraph::k_hop_nbrs(
        handle_,
        graph_view,
        raft::device_span<vertex_t const>{start_vertices.data(), start_vertices.size()},
        size_t{2},
        do_expensive_check_);

      auto src = cugraph::c_api::expand_sparse_offsets(
        raft::device_span<size_t const>{offsets.data(), offsets.size()},
        vertex_t{0},
        handle_.get_stream());

      // convert ids back to srcs:  src[i] = start_vertices[src[i]]
      cugraph::unrenumber_local_int_vertices(handle_,
                                             src.data(),
                                             src.size(),
                                             start_vertices.data(),
                                             vertex_t{0},
                                             graph_view.local_vertex_partition_range_size(),
                                             do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        src.data(),
        src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        dst.data(),
        dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      result_ = new cugraph::c_api::cugraph_vertex_pairs_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dst, graph_->vertex_type_)};
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
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* start_vertices,
  bool_t do_expensive_check,
  cugraph_vertex_pairs_t** result,
  cugraph_error_t** error)
{
  two_hop_neighbors_functor functor(handle, graph, start_vertices, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
