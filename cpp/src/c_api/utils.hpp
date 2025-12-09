/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/generic_cascaded_dispatch.hpp"
#include "c_api/graph.hpp"

namespace cugraph {
namespace c_api {

template <typename functor_t, typename result_t>
cugraph_error_code_t run_algorithm(::cugraph_graph_t const* graph,
                                   functor_t& functor,
                                   result_t* result,
                                   ::cugraph_error_t** error)
{
  *result = result_t{};
  *error  = nullptr;

  try {
    auto p_graph = reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph);

    cugraph::c_api::vertex_dispatcher(p_graph->vertex_type_,
                                      p_graph->edge_type_,
                                      p_graph->weight_type_,
                                      p_graph->edge_type_id_type_,
                                      p_graph->edge_time_type_,
                                      p_graph->store_transposed_,
                                      p_graph->multi_gpu_,
                                      functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<::cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    if constexpr (std::is_same_v<result_t, decltype(functor.result_)>) {
      *result = functor.result_;
    } else {
      *result = reinterpret_cast<result_t>(functor.result_);
    }
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

}  // namespace c_api
}  // namespace cugraph
