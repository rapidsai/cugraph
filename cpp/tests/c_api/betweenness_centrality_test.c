/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <cugraph_c/random.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_betweenness_centrality_test(vertex_t* h_src,
                                        vertex_t* h_dst,
                                        weight_t* h_wgt,
                                        weight_t* h_result,
                                        size_t num_vertices,
                                        size_t num_edges,
                                        bool_t store_transposed,
                                        bool_t is_symmetric,
                                        bool_t normalized,
                                        size_t num_vertices_to_sample)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle                   = NULL;
  cugraph_graph_t* p_graph                            = NULL;
  cugraph_centrality_result_t* p_result               = NULL;
  cugraph_rng_state_t* rng_state                      = NULL;
  cugraph_type_erased_device_array_t* seeds           = NULL;
  cugraph_type_erased_device_array_view_t* seeds_view = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "failed to create rng_state.");

  ret_code = create_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, is_symmetric, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_select_random_vertices(
    handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "select random seeds failed.");

  seeds_view = cugraph_type_erased_device_array_view(seeds);

  ret_code = cugraph_betweenness_centrality(
    handle, p_graph, seeds_view, normalized, FALSE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_betweenness_centrality failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* centralities;

  vertices     = cugraph_centrality_result_get_vertices(p_result);
  centralities = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_centralities[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_centralities, centralities, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_centralities[i], 0.001),
                "centralities results don't match");
  }

  cugraph_centrality_result_free(p_result);

  cugraph_type_erased_device_array_view_free(seeds_view);
  cugraph_type_erased_device_array_free(seeds);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_betweenness_centrality()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0, 3.33333, 0.66667, 1.08333, 0.416667, 0.5};

  // Betweenness centrality wants store_transposed = FALSE
  return generic_betweenness_centrality_test(
                                             h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, TRUE, FALSE, 5);
}

int test_betweenness_centrality_normalized()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0, .333333, .066667, .108333, 0.0416667, 0.05};

  // Betweenness centrality wants store_transposed = FALSE
  return generic_betweenness_centrality_test(
                                             h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, TRUE, TRUE, 5);
}

int test_betweenness_centrality_full()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = { 0, 3.66667, 0.833333, 2.16667, 0.833333, 0.5 };

  return generic_betweenness_centrality_test(
                                             h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, TRUE, FALSE, 6);
}

int test_betweenness_centrality_full_directed()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[] = { 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f };
  weight_t h_result[] = {0, 4, 0, 2, 1, 0};

  return generic_betweenness_centrality_test(
                                             h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, FALSE, FALSE, 6);
}


/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_betweenness_centrality);
  result |= RUN_TEST(test_betweenness_centrality_normalized);
  result |= RUN_TEST(test_betweenness_centrality_full);
  result |= RUN_TEST(test_betweenness_centrality_full_directed);
  return result;
}
