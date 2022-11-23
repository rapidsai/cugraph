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
#include <c_api/core_result.hpp>
#include <c_api/graph.hpp>
#include <c_api/resource_handle.hpp>
#include <c_api/utils.hpp>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <optional>

namespace {

struct k_core_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  size_t k_;
  cugraph_k_core_degree_type_t degree_type_;
  cugraph::c_api::cugraph_core_result_t const* core_result_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_k_core_result_t* result_{};

  k_core_functor(cugraph_resource_handle_t const* handle,
                 cugraph_graph_t* graph,
                 size_t k,
                 cugraph_k_core_degree_type_t degree_type,
                 cugraph_core_result_t const* core_result,
                 bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      k_(k),
      degree_type_(degree_type),
      core_result_(reinterpret_cast<cugraph::c_api::cugraph_core_result_t const*>(core_result)),
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
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS)
          ;
      }
      // FIXME: Transpose_storage may have a bug, since if store_transposed is True it can reverse
      // the bool value of is_symmetric
      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_weights = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 weight_t>*>(graph_->edge_weights_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<edge_t> k_cores(graph_view.local_vertex_partition_range_size(),
                                          handle_.get_stream());

      rmm::device_uvector<edge_t> core_result_values(0, handle_.get_stream());

      if (core_result_ != nullptr) {
        rmm::device_uvector<vertex_t> core_result_vertices(core_result_->core_numbers_->size_,
                                                           handle_.get_stream());
        core_result_values.resize(core_result_->core_numbers_->size_, handle_.get_stream());

        raft::copy(core_result_vertices.data(),
                   core_result_->vertex_ids_->as_type<vertex_t>(),
                   core_result_->vertex_ids_->size_,
                   handle_.get_stream());

        raft::copy(core_result_values.data(),
                   core_result_->core_numbers_->as_type<edge_t>(),
                   core_result_->core_numbers_->size_,
                   handle_.get_stream());

        core_result_values = cugraph::detail::
          collect_local_vertex_values_from_ext_vertex_value_pairs<vertex_t, edge_t, multi_gpu>(
            handle_,
            std::move(core_result_vertices),
            std::move(core_result_values),
            *number_map,
            graph_view.local_vertex_partition_range_first(),
            graph_view.local_vertex_partition_range_last(),
            vertex_t{0},
            do_expensive_check_);
      }

      auto degree_type = reinterpret_cast<cugraph::k_core_degree_type_t>(degree_type);

      auto [result_src, result_dst, result_wgt] =
        cugraph::k_core<vertex_t, edge_t, weight_t, multi_gpu>(
          handle_,
          graph_view,
          (edge_weights != nullptr) ? std::make_optional(edge_weights->view()) : std::nullopt,
          k_,
          std::make_optional(degree_type),
          (core_result_ == nullptr) ? std::nullopt
                                    : std::make_optional<raft::device_span<edge_t const>>(
                                        core_result_values.data(), core_result_values.size()),
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

      result_ = new cugraph::c_api::cugraph_k_core_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_src, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(result_dst, graph_->vertex_type_),
        result_wgt ? new cugraph::c_api::cugraph_type_erased_device_array_t(*result_wgt,
                                                                            graph_->weight_type_)
                   : NULL};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_k_core(const cugraph_resource_handle_t* handle,
                                               cugraph_graph_t* graph,
                                               size_t k,
                                               cugraph_k_core_degree_type_t degree_type,
                                               const cugraph_core_result_t* core_result,
                                               bool_t do_expensive_check,
                                               cugraph_k_core_result_t** result,
                                               cugraph_error_t** error)
{
  k_core_functor functor(handle, graph, k, degree_type, core_result, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
