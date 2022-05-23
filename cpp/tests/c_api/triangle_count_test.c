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

int generic_triangle_count_test(vertex_t* h_src,
                                vertex_t* h_dst,
                                weight_t* h_wgt,
                                vertex_t* h_verts,
                                edge_t* h_result,
                                size_t num_vertices,
                                size_t num_edges,
                                size_t num_results,
                                bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle                   = NULL;
  cugraph_graph_t* p_graph                              = NULL;
  cugraph_triangle_count_result_t* p_result             = NULL;
  cugraph_type_erased_device_array_t* p_start           = NULL;
  cugraph_type_erased_device_array_view_t* p_start_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, TRUE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (h_verts != NULL) {
    ret_code =
      cugraph_type_erased_device_array_create(p_handle, num_results, INT32, &p_start, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_start create failed.");

    p_start_view = cugraph_type_erased_device_array_view(p_start);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      p_handle, p_start_view, (byte_t*)h_verts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");
  }

  ret_code = cugraph_triangle_count(p_handle, p_graph, p_start_view, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_triangle_count failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* vertices;
    cugraph_type_erased_device_array_view_t* counts;

    vertices = cugraph_triangle_count_result_get_vertices(p_result);
    counts   = cugraph_triangle_count_result_get_counts(p_result);

    TEST_ASSERT(test_ret_value,
                cugraph_type_erased_device_array_view_size(vertices) == num_results,
                "invalid number of results");

    vertex_t num_local_results = cugraph_type_erased_device_array_view_size(vertices);

    vertex_t h_vertices[num_local_results];
    edge_t h_counts[num_local_results];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_vertices, vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_counts, counts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    for (int i = 0; (i < num_local_results) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(
        test_ret_value, h_result[h_vertices[i]] == h_counts[i], "counts results don't match");
    }

    cugraph_triangle_count_result_free(p_result);
  }

  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_triangle_count()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_results  = 4;

  vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 4,
                        1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 5, 5,
                        0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f,
                        0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_verts[] = {0, 1, 2, 4};
  edge_t h_result[]  = {1, 2, 2, 0};

  // Triangle Count wants store_transposed = FALSE
  return generic_triangle_count_test(
    h_src, h_dst, h_wgt, h_verts, h_result, num_vertices, num_edges, num_results, FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_triangle_count);
  return result;
}
