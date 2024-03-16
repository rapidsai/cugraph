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

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <c_api/abstract_functor.hpp>
#include <c_api/graph.hpp>
#include <c_api/induced_subgraph_result.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <optional>

namespace {

struct k_truss_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_;
  size_t k_;
  bool do_expensive_check_;
  cugraph::c_api::cugraph_induced_subgraph_result_t* result_{};

  k_truss_functor(::cugraph_resource_handle_t const* handle,
                  ::cugraph_graph_t* graph,
                  size_t k,
                  bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      k_(k),
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
    if constexpr (!cugraph::is_candidate_legacy<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else if constexpr (multi_gpu) {
      unsupported();
    } else {
      // k_truss expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view = graph->view();
      rmm::device_uvector<vertex_t> src(0, handle_.get_stream());
      rmm::device_uvector<vertex_t> dst(0, handle_.get_stream());
      std::optional<rmm::device_uvector<weight_t>> wgt{std::nullopt};

      auto [result_src, result_dst, result_wgt] =
        cugraph::k_truss<vertex_t, edge_t, weight_t, multi_gpu>(
          handle_,
          graph_view,
          edge_weights ? std::make_optional(edge_weights->view()) : std::nullopt,
          k_,
          do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        result_src.data(),
        result_src.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        handle_,
        result_dst.data(),
        result_dst.size(),
        number_map->data(),
        graph_view.vertex_partition_range_lasts(),
        do_expensive_check_);

      rmm::device_uvector<size_t> edge_offsets(2, handle_.get_stream());
      std::vector<size_t> h_edge_offsets{{0, result_src.size()}};
      raft::update_device(
        edge_offsets.data(), h_edge_offsets.data(), h_edge_offsets.size(), handle_.get_stream());

      // FIXME: Add support for edge_id and edge_type_id.
      result_ = new cugraph::c_api::cugraph_induced_subgraph_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_dst, graph_->vertex_type_),
        result_wgt ? new cugraph::c_api::cugraph_type_erased_device_array_t(*result_wgt,
                                                                            graph_->weight_type_)
                   : NULL,
        NULL,
        NULL,
        new cugraph::c_api::cugraph_type_erased_device_array_t(edge_offsets,
                                                               cugraph_data_type_id_t::SIZE_T)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_k_truss_subgraph(const cugraph_resource_handle_t* handle,
                                                         cugraph_graph_t* graph,
                                                         size_t k,
                                                         bool_t do_expensive_check,
                                                         cugraph_induced_subgraph_result_t** result,
                                                         cugraph_error_t** error)
{
  k_truss_functor functor(handle, graph, k, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
