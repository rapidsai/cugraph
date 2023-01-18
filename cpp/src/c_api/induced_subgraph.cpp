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
#include <c_api/induced_subgraph_helper.hpp>
#include <c_api/induced_subgraph_result.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

namespace {

struct induced_subgraph_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* subgraph_offsets_{};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* subgraph_vertices_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  induced_subgraph_functor(cugraph_resource_handle_t const* handle,
                           cugraph_graph_t* graph,
                           cugraph_type_erased_device_array_view_t const* subgraph_offsets,
                           cugraph_type_erased_device_array_view_t const* subgraph_vertices,
                           bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      subgraph_offsets_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          subgraph_offsets)),
      subgraph_vertices_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          subgraph_vertices)),
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
      // induced subgraph expects store_transposed == false
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

      rmm::device_uvector<size_t> subgraph_offsets(0, handle_.get_stream());
      rmm::device_uvector<vertex_t> subgraph_vertices(subgraph_vertices_->size_,
                                                      handle_.get_stream());

      raft::copy(subgraph_vertices.data(),
                 subgraph_vertices_->as_type<vertex_t>(),
                 subgraph_vertices_->size_,
                 handle_.get_stream());

      if constexpr (multi_gpu) {
        std::tie(subgraph_vertices, subgraph_offsets) =
          cugraph::c_api::detail::shuffle_vertex_ids_and_offsets(
            handle_,
            std::move(subgraph_vertices),
            raft::device_span<size_t const>{subgraph_offsets_->as_type<size_t>(),
                                            subgraph_offsets_->size_});
      } else {
        subgraph_offsets.resize(subgraph_offsets_->size_, handle_.get_stream());
        raft::copy(subgraph_offsets.data(),
                   subgraph_offsets_->as_type<size_t>(),
                   subgraph_offsets_->size_,
                   handle_.get_stream());
      }

      //
      // Need to renumber subgraph_vertices
      //
      cugraph::renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle_,
        subgraph_vertices.data(),
        subgraph_vertices.size(),
        number_map->data(),
        graph_view.local_vertex_partition_range_first(),
        graph_view.local_vertex_partition_range_last(),
        do_expensive_check_);

      auto [src, dst, wgt, graph_offsets] = cugraph::extract_induced_subgraphs(
        handle_,
        graph_view,
        (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
        raft::device_span<size_t const>{subgraph_offsets.data(), subgraph_offsets.size()},
        raft::device_span<vertex_t const>{subgraph_vertices.data(), subgraph_vertices.size()},
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

      result_ = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(dst, graph_->vertex_type_),
        wgt ? new cugraph::c_api::cugraph_type_erased_device_array_t(*wgt, graph_->weight_type_)
            : NULL,
        new cugraph::c_api::cugraph_type_erased_device_array_t(graph_offsets, SIZE_T)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_extract_induced_subgraph(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* subgraph_offsets,
  const cugraph_type_erased_device_array_view_t* subgraph_vertices,
  bool_t do_expensive_check,
  cugraph_induced_subgraph_result_t** result,
  cugraph_error_t** error)
{
  CAPI_EXPECTS(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)->vertex_type_ ==
                 reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
                   subgraph_vertices)
                   ->type_,
               CUGRAPH_INVALID_INPUT,
               "vertex type of graph and subgraph_vertices must match",
               *error);

  induced_subgraph_functor functor(
    handle, graph, subgraph_offsets, subgraph_vertices, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
