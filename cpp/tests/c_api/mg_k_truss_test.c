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

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/community_algorithms.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>

#include <stdio.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int generic_k_truss_test(const cugraph_resource_handle_t* handle,
                         vertex_t* h_src,
                         vertex_t* h_dst,
                         weight_t* h_wgt,
                         size_t num_edges,
                         size_t num_results,
                         size_t k,
                         bool_t store_transposed,
                         vertex_t* h_result_src,
                         vertex_t* h_result_dst,
                         weight_t* h_result_wgt)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph = NULL;

  cugraph_induced_subgraph_result_t* result = NULL;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t size_t_tid = SIZE_T;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, TRUE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_k_truss_subgraph(handle, graph, k, FALSE, &result, &ret_error);
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_k_truss failed.");

  cugraph_type_erased_device_array_view_t* k_truss_src;
  cugraph_type_erased_device_array_view_t* k_truss_dst;
  cugraph_type_erased_device_array_view_t* k_truss_wgt;

  k_truss_src = cugraph_induced_subgraph_get_sources(result);
  k_truss_dst = cugraph_induced_subgraph_get_destinations(result);
  k_truss_wgt = cugraph_induced_subgraph_get_edge_weights(result);

  size_t k_truss_size = cugraph_type_erased_device_array_view_size(k_truss_src);

  vertex_t h_k_truss_src[k_truss_size];
  vertex_t h_k_truss_dst[k_truss_size];
  weight_t h_k_truss_wgt[k_truss_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_k_truss_src, k_truss_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_k_truss_dst, k_truss_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_k_truss_wgt, k_truss_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (size_t i = 0; (i < k_truss_size) && (test_ret_value == 0); ++i) {
    bool_t found = FALSE;
    for (size_t j = 0; (j < num_results) && !found; ++j) {
      if ((h_k_truss_src[i] == h_result_src[j]) && (h_k_truss_dst[i] == h_result_dst[j]) &&
          (h_k_truss_wgt[i] == h_result_wgt[j]))
        found = TRUE;
    }
    TEST_ASSERT(test_ret_value, found, "k_truss subgraph has an edge that doesn't match");
  }

  cugraph_induced_subgraph_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_k_truss_subgraph(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 14;
  size_t num_vertices = 7;
  size_t num_results  = 6;
  size_t k            = 3;

  vertex_t h_src[] = {0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 6};
  vertex_t h_dst[] = {1, 2, 5, 0, 2, 3, 4, 6, 0, 1, 1, 1, 0, 1};
  weight_t h_wgt[] = {
    1.2f, 1.3f, 1.6f, 1.2f, 2.3f, 2.4f, 2.5f, 2.7f, 1.3f, 2.3f, 2.4f, 2.5f, 1.6f, 2.7f};

  vertex_t h_result_src[] = {0, 2, 2, 1, 1, 0};
  vertex_t h_result_dst[] = {1, 1, 0, 0, 2, 2};
  weight_t h_result_wgt[] = {1.2f, 2.3f, 1.3f, 1.2f, 2.3f, 1.3f};

  return generic_k_truss_test(handle,
                              h_src,
                              h_dst,
                              h_wgt,
                              num_edges,
                              num_results,
                              k,
                              FALSE,
                              h_result_src,
                              h_result_dst,
                              h_result_wgt);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_k_truss_subgraph, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
