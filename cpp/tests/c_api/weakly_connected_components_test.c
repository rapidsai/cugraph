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
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_wcc_test(vertex_t* h_src,
                     vertex_t* h_dst,
                     weight_t* h_wgt,
                     vertex_t* h_result,
                     size_t num_vertices,
                     size_t num_edges,
                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_labeling_result_t* p_result = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, TRUE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_weakly_connected_components(p_handle, p_graph, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_weakly_connected_components failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* components;

  vertices   = cugraph_labeling_result_get_vertices(p_result);
  components = cugraph_labeling_result_get_labels(p_result);

  vertex_t h_vertices[num_vertices];
  vertex_t h_components[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_components, components, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  vertex_t component_check[num_vertices];
  for (vertex_t i = 0; i < num_vertices; ++i) {
    component_check[i] = num_vertices;
  }

  vertex_t num_errors = 0;
  for (vertex_t i = 0; i < num_vertices; ++i) {
    if (component_check[h_components[i]] == num_vertices) {
      component_check[h_components[i]] = h_result[h_vertices[i]];
    } else if (component_check[h_components[i]] != h_result[h_vertices[i]]) {
      ++num_errors;
    }
  }

  TEST_ASSERT(test_ret_value, num_errors == 0, "weakly connected components results don't match");

  cugraph_type_erased_device_array_view_free(components);
  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_labeling_result_free(p_result);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_weakly_connected_components()
{
  size_t num_edges    = 32;
  size_t num_vertices = 12;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                      1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                      0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
  weight_t h_wgt[]    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  // WCC wants store_transposed = FALSE
  return generic_wcc_test(h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE);
}

int test_weakly_connected_components_transpose()
{
  size_t num_edges    = 32;
  size_t num_vertices = 12;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                      1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                      0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
  weight_t h_wgt[]    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  // WCC wants store_transposed = FALSE
  return generic_wcc_test(h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, TRUE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_weakly_connected_components);
  result |= RUN_TEST(test_weakly_connected_components_transpose);
  return result;
}
