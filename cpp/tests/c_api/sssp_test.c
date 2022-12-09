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

#include <float.h>
#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;

const float EPSILON = 0.001;

int generic_sssp_test(vertex_t* h_src,
                      vertex_t* h_dst,
                      float* h_wgt,
                      vertex_t source,
                      float const* expected_distances,
                      vertex_t const* expected_predecessors,
                      size_t num_vertices,
                      size_t num_edges,
                      float cutoff,
                      bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_paths_result_t* p_result    = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  ret_code = cugraph_sssp(p_handle, p_graph, source, cutoff, TRUE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_sssp failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  float h_distances[num_vertices];
  vertex_t h_predecessors[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_predecessors, predecessors, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(expected_distances[h_vertices[i]], h_distances[i], EPSILON),
                "sssp distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "sssp predecessors don't match");
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_type_erased_device_array_view_free(predecessors);
  cugraph_paths_result_free(p_result);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_sssp_test_double(vertex_t* h_src,
                      vertex_t* h_dst,
                      double* h_wgt,
                      vertex_t source,
                      double const* expected_distances,
                      vertex_t const* expected_predecessors,
                      size_t num_vertices,
                      size_t num_edges,
                      double cutoff,
                      bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_paths_result_t* p_result    = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph_double(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  ret_code = cugraph_sssp(p_handle, p_graph, source, cutoff, TRUE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_sssp failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  double h_distances[num_vertices];
  vertex_t h_predecessors[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_predecessors, predecessors, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqualDouble(expected_distances[h_vertices[i]], h_distances[i], EPSILON),
                "sssp distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "sssp predecessors don't match");
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_type_erased_device_array_view_free(predecessors);
  cugraph_paths_result_free(p_result);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_sssp()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  float wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  float expected_distances[]    = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  return generic_sssp_test(src,
                           dst,
                           wgt,
                           0,
                           expected_distances,
                           expected_predecessors,
                           num_vertices,
                           num_edges,
                           10,
                           FALSE);
}

int test_sssp_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  float wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  float expected_distances[]    = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_sssp to transpose the graph
  return generic_sssp_test(
    src, dst, wgt, 0, expected_distances, expected_predecessors, num_vertices, num_edges, 10, TRUE);
}

int test_sssp_with_transpose_double()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  double wgt[]                   = {0.1d, 2.1d, 1.1d, 5.1d, 3.1d, 4.1d, 7.2d, 3.2d};
  double expected_distances[]    = {0.0d, 0.1d, DBL_MAX, 2.2d, 1.2d, 4.4d};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_sssp to transpose the graph
  return generic_sssp_test_double(
    src, dst, wgt, 0, expected_distances, expected_predecessors, num_vertices, num_edges, 10, TRUE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_sssp);
  result |= RUN_TEST(test_sssp_with_transpose);
  result |= RUN_TEST(test_sssp_with_transpose_double);
  return result;
}
