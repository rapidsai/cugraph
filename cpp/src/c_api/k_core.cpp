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
  cugraph::c_api::cugraph_core_result_t* result_{};

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
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, weight_t, false, multi_gpu>*>(
          graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<edge_t> k_cores(graph_view.local_vertex_partition_range_size(),
                                          handle_.get_stream());

      auto degree_type = reinterpret_cast<cugraph::k_core_degree_type_t>(degree_type);

      // FIXME:  calling `view()` on an array returns an object allocated on the host heap.
      //         This needs to be freed (memory leak)
      cugraph::k_core<vertex_t, edge_t, weight_t, multi_gpu>(
        handle_,
        graph_view,
        k_,
        std::make_optional(degree_type),
        (core_result_ == nullptr)
          ? std::nullopt
          : std::make_optional<raft::device_span<edge_t const>>(
              // reinterpret_cast<edge_t const*>(core_result_->core_numbers_->data_.data()),
              core_result_->core_numbers_->as_type<edge_t const>(),
              core_result_->core_numbers_->size_),
        do_expensive_check_);

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_core_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(k_cores, graph_->edge_type_)};
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
