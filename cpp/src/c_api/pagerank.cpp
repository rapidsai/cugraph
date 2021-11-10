/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace c_api {

struct cugraph_pagerank_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* pageranks_;
};

}  // namespace c_api

extern "C" cugraph_type_erased_device_array_t* cugraph_pagerank_result_get_vertices(
  cugraph_pagerank_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_pagerank_result_t*>(result);
  return internal_pointer->vertex_ids_;
}

extern "C" cugraph_type_erased_device_array_t* cugraph_pagerank_result_get_pageranks(
  cugraph_pagerank_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_pagerank_result_t*>(result);
  return internal_pointer->pageranks_;
}

extern "C" void cugraph_pagerank_result_free(cugraph_pagerank_result_t* result)
{
  auto internal_pointer = reinterpret_cast<c_api::cugraph_pagerank_result_t*>(result);
  delete internal_pointer->vertex_ids_;
  delete internal_pointer->pageranks_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t compute_pagerank(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool do_expensive_check,
  cugraph_pagerank_result_t** result,
  cugraph_error_t** error)
{
  //
  //  TODO:  (all algorithms will have this
  //          basic construct, only defining here)
  //    1) Adapt visitor implementation to handle cugraph_graph_t *
  //       instead of graph envelope
  //    2) Create erased pack (or whatever is required)
  //    3) Add calls here (as appropriate based on 1 and 2) to:
  //        a) if has_initial_guess, renumber the vertex_ids array
  //           and organize the pageranks accordingly
  //        b) cast graph as appropriate thing
  //        c) call visitor method for pagerank
  //        d) unrenumber result
  return CUGRAPH_NOT_IMPLEMENTED;
}

extern "C" cugraph_error_code_t cugraph_personalized_pagerank(
  const cugraph_resource_handle_t* handle,
  const cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
  const cugraph_type_erased_device_array_t* personalization_vertices,
  const cugraph_type_erased_device_array_t* personalization_values,
  double alpha,
  double epsilon,
  size_t max_iterations,
  bool has_initial_guess,
  bool do_expensive_check,
  cugraph_pagerank_result_t** result,
  cugraph_error_t** error)
{
  return CUGRAPH_NOT_IMPLEMENTED;
}
