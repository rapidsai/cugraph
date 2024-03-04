/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>

#include <stdio.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int generic_degrees_test(const cugraph_resource_handle_t* handle,
                         vertex_t* h_src,
                         vertex_t* h_dst,
                         weight_t* h_wgt,
                         size_t num_vertices,
                         size_t num_edges,
                         bool_t store_transposed,
                         bool_t is_symmetric,
                         edge_t* h_in_degrees,
                         edge_t* h_out_degrees)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                    = NULL;
  cugraph_degrees_result_t* result = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, is_symmetric, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_degrees(
    handle, graph, FALSE, &result, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_extract_degrees failed.");

  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* result_in_degrees;
  cugraph_type_erased_device_array_view_t* result_out_degrees;

  result_vertices    = cugraph_degrees_result_get_vertices(result);
  result_in_degrees  = cugraph_degrees_result_get_in_degrees(result);
  result_out_degrees = cugraph_degrees_result_get_out_degrees(result);

  size_t num_result_vertices = cugraph_type_erased_device_array_view_size(result_vertices);

  vertex_t h_result_vertices[num_result_vertices];
  edge_t   h_result_in_degrees[num_result_vertices];
  edge_t   h_result_out_degrees[num_result_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_in_degrees, result_in_degrees, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_out_degrees, result_out_degrees, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (size_t i = 0; (i < num_result_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value, h_result_in_degrees[i] == h_in_degrees[h_result_vertices[i]], "in degree did not match");
    TEST_ASSERT(test_ret_value, h_result_out_degrees[i] == h_out_degrees[h_result_vertices[i]], "out degree did not match");
  }

  cugraph_degrees_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_degrees(const cugraph_resource_handle_t* handle)
{
  size_t num_edges            = 8;
  size_t num_vertices         = 6;

  vertex_t h_src[]               = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]               = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]               = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_in_degrees[]        = {1, 2, 0, 2, 1, 2};
  vertex_t h_out_degrees[]       = {1, 2, 3, 1, 1, 0};

  // Pagerank wants store_transposed = TRUE
  return generic_degrees_test(handle,
                              h_src,
                              h_dst,
                              h_wgt,
                              num_vertices,
                              num_edges,
                              TRUE,
                              FALSE,
                              h_in_degrees,
                              h_out_degrees);
}

int test_degrees_symmetric(const cugraph_resource_handle_t* handle)
{
  size_t num_edges         = 16;
  size_t num_vertices      = 6;

  vertex_t h_src[]         = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]         = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]         = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f,
                              0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_in_degrees[]  = {2, 4, 3, 3, 2, 2};
  vertex_t h_out_degrees[] = {2, 4, 3, 3, 2, 2};

  // Pagerank wants store_transposed = TRUE
  return generic_degrees_test(handle,
                              h_src,
                              h_dst,
                              h_wgt,
                              num_vertices,
                              num_edges,
                              TRUE,
                              TRUE,
                              h_in_degrees,
                              h_out_degrees);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_degrees, handle);
  result |= RUN_MG_TEST(test_degrees_symmetric, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
