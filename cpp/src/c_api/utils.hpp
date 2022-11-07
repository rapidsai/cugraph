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

#include <c_api/graph.hpp>
#include <cugraph/visitors/generic_cascaded_dispatch.hpp>

namespace cugraph {
namespace c_api {

template <typename functor_t, typename result_t>
cugraph_error_code_t run_algorithm(::cugraph_graph_t const* graph,
                                   functor_t& functor,
                                   result_t** result,
                                   ::cugraph_error_t** error)
{
  *result = nullptr;
  *error  = nullptr;

  try {
    auto p_graph = reinterpret_cast<cugraph::c_api::cugraph_graph_t const*>(graph);

    cugraph::dispatch::vertex_dispatcher(
      cugraph::c_api::dtypes_mapping[p_graph->vertex_type_],
      cugraph::c_api::dtypes_mapping[p_graph->edge_type_],
      cugraph::c_api::dtypes_mapping[p_graph->weight_type_],
      cugraph::c_api::dtypes_mapping[p_graph->edge_type_id_type_],
      p_graph->store_transposed_,
      p_graph->multi_gpu_,
      functor);

    if (functor.error_code_ != CUGRAPH_SUCCESS) {
      *error = reinterpret_cast<::cugraph_error_t*>(functor.error_.release());
      return functor.error_code_;
    }

    *result = reinterpret_cast<result_t*>(functor.result_);
  } catch (std::exception const& ex) {
    *error = reinterpret_cast<::cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ex.what()});
    return CUGRAPH_UNKNOWN_ERROR;
  }

  return CUGRAPH_SUCCESS;
}

}  // namespace c_api
}  // namespace cugraph
