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

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_k_core_test(vertex_t* h_src,
                        vertex_t* h_dst,
                        weight_t* h_wgt,
                        vertex_t* h_result_src,
                        vertex_t* h_result_dst,
                        weight_t* h_result_wgt,
                        size_t num_vertices,
                        size_t num_edges,
                        size_t num_result_edges,
                        size_t k,
                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* resource_handle = NULL;
  cugraph_graph_t* graph                     = NULL;
  cugraph_core_result_t* core_result         = NULL;
  cugraph_k_core_result_t* k_core_result     = NULL;

  resource_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, resource_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(resource_handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               num_edges,
                               store_transposed,
                               FALSE,
                               TRUE,
                               &graph,
                               &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_core_number(
    resource_handle, graph, K_CORE_DEGREE_TYPE_IN, FALSE, &core_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_core_number failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_k_core(resource_handle,
                            graph,
                            k,
                            K_CORE_DEGREE_TYPE_IN,
                            core_result,
                            FALSE,
                            &k_core_result,
                            &ret_error);

  cugraph_type_erased_device_array_view_t* src_vertices;
  cugraph_type_erased_device_array_view_t* dst_vertices;
  cugraph_type_erased_device_array_view_t* weights;

  src_vertices = cugraph_k_core_result_get_src_vertices(k_core_result);
  dst_vertices = cugraph_k_core_result_get_dst_vertices(k_core_result);
  weights      = cugraph_k_core_result_get_weights(k_core_result);

  size_t number_of_result_edges = cugraph_type_erased_device_array_view_size(src_vertices);

  vertex_t h_src_vertices[number_of_result_edges];
  vertex_t h_dst_vertices[number_of_result_edges];
  weight_t h_weights[number_of_result_edges];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_src_vertices, src_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_dst_vertices, dst_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_weights, weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  TEST_ASSERT(test_ret_value,
              number_of_result_edges == num_result_edges,
              "Number of resulting edges does not match.");

  weight_t M[num_vertices][num_vertices];
  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = 0;

  for (int i = 0; i < num_result_edges; ++i)
    M[h_result_src[i]][h_result_dst[i]] = h_result_wgt[i];

  for (int i = 0; (i < number_of_result_edges) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_src_vertices[i]][h_dst_vertices[i]] == h_weights[i],
                "edge does not match");
  }

  cugraph_k_core_result_free(k_core_result);
  cugraph_core_result_free(core_result);
  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(resource_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_k_core()
{
  size_t num_edges        = 22;
  size_t num_vertices     = 7;
  size_t num_result_edges = 12;
  size_t k                = 3;

  vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
  vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
  weight_t h_wgt[]        = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_result_src[] = {1, 1, 3, 4, 3, 4, 3, 4, 5, 5, 1, 5};
  vertex_t h_result_dst[] = {3, 4, 5, 5, 1, 3, 4, 1, 3, 4, 5, 1};
  weight_t h_result_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  return generic_k_core_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result_src,
                             h_result_dst,
                             h_result_wgt,
                             num_vertices,
                             num_edges,
                             num_result_edges,
                             k,
                             FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_k_core);
  return result;
}
