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

#include "c_api/abstract_functor.hpp"
#include "c_api/centrality_result.hpp"
#include "c_api/graph.hpp"
#include "c_api/graph_functions.hpp"
#include "c_api/graph_helper.hpp"
#include "c_api/hierarchical_clustering_result.hpp"
#include "c_api/random.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/community_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <raft/core/handle.hpp>

#include <optional>

namespace cugraph {
namespace c_api {

struct cugraph_edge_ids_lookup_result_t {
  cugraph_type_erased_device_array_t* edge_ids_;
  cugraph_vertex_pairs_t* vertex_pairs_;
};

}  // namespace c_api
}  // namespace cugraph

namespace {
struct edge_ids_lookup_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{nullptr};
  cugraph::c_api::cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup_{};
  bool do_expensive_check_;
  cugraph::c_api::cugraph_edge_ids_lookup_result_t* result_{};

  edge_ids_lookup_functor(::cugraph_resource_handle_t const* handle,
                          ::cugraph_graph_t* graph,
                          ::cugraph_type_erased_device_array_view_t const* edge_ids_to_lookup,
                          bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      edge_ids_to_lookup_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          edge_ids_to_lookup)),
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
      // expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = CUGRAPH_SUCCESS;
        // error_code_ = cugraph::c_api::
        //   transpose_storage<vertex_t, edge_t, edge_t, store_transposed, multi_gpu>(
        //     handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto edge_ids = reinterpret_cast<
        cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>,
                                 edge_t>*>(graph_->edge_ids_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto [d_edge_ids, d_srcs, d_dsts] = cugraph::lookup_edge_ids(
        handle_,
        graph_view,
        (edge_ids != nullptr) ? std::make_optional(edge_ids->view()) : std::nullopt,
        raft::device_span<edge_t const>{edge_ids_to_lookup_->as_type<edge_t const>(),
                                        edge_ids_to_lookup_->size_});

      result_ = new cugraph::c_api::cugraph_edge_ids_lookup_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(d_edge_ids, graph_->edge_type_),
        new cugraph::c_api::cugraph_vertex_pairs_t{
          new cugraph::c_api::cugraph_type_erased_device_array_t(d_srcs, graph_->vertex_type_),
          new cugraph::c_api::cugraph_type_erased_device_array_t(d_dsts, graph_->vertex_type_)}};
    }
  }
};

}  // namespace

extern "C" cugraph_type_erased_device_array_view_t* cugraph_edge_ids_lookup_result_get_edge_ids(
  cugraph_edge_ids_lookup_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_ids_lookup_result_t const*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->edge_ids_->view());
}

extern "C" cugraph_vertex_pairs_t* cugraph_edge_ids_lookup_result_get_vertex_pairs(
  cugraph_edge_ids_lookup_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_ids_lookup_result_t*>(result);
  return reinterpret_cast<cugraph_vertex_pairs_t*>(internal_pointer->vertex_pairs_);
}

extern "C" void cugraph_edge_ids_lookup_result_free(cugraph_edge_ids_lookup_result_t* result)
{
  auto internal_pointer =
    reinterpret_cast<cugraph::c_api::cugraph_edge_ids_lookup_result_t*>(result);
  delete internal_pointer->edge_ids_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_lookup_src_dst_from_edge_id(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* edge_ids_to_lookup,
  bool_t do_expensive_check,
  cugraph_edge_ids_lookup_result_t** result,
  cugraph_error_t** error)
{
  edge_ids_lookup_functor functor(handle, graph, edge_ids_to_lookup, do_expensive_check);
  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
