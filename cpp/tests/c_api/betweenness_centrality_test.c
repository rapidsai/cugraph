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
                                        vertex_t* h_seeds,
                                        weight_t* h_result,
                                        size_t num_vertices,
                                        size_t num_edges,
                                        size_t num_seeds,
                                        bool_t store_transposed,
                                        bool_t is_symmetric,
                                        bool_t normalized,
                                        bool_t include_endpoints,
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

  ret_code = create_test_graph(handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               num_edges,
                               store_transposed,
                               FALSE,
                               is_symmetric,
                               &p_graph,
                               &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (h_seeds == NULL) {
    ret_code = cugraph_select_random_vertices(
      handle, p_graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "select random seeds failed.");

    seeds_view = cugraph_type_erased_device_array_view(seeds);
  } else {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_seeds, INT32, &seeds, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seeds create failed.");

    seeds_view = cugraph_type_erased_device_array_view(seeds);
    ret_code   = cugraph_type_erased_device_array_view_copy_from_host(
      handle, seeds_view, (byte_t*)h_seeds, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seeds copy_from_host failed.");
  }

  ret_code = cugraph_betweenness_centrality(
    handle, p_graph, seeds_view, normalized, include_endpoints, FALSE, &p_result, &ret_error);
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
                nearlyEqual(h_result[h_vertices[i]], h_centralities[i], 0.00001),
                "centralities results don't match");
  }

  cugraph_centrality_result_free(p_result);

  cugraph_type_erased_device_array_view_free(seeds_view);
  cugraph_type_erased_device_array_free(seeds);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_betweenness_centrality_full()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0, 3.66667, 0.833333, 2.16667, 0.833333, 0.5};

  return generic_betweenness_centrality_test(
    h_src, h_dst, h_wgt, NULL, h_result, num_vertices, num_edges, 0, FALSE, TRUE, FALSE, FALSE, 6);
}

int test_betweenness_centrality_full_directed()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0, 4, 0, 2, 1, 0};

  return generic_betweenness_centrality_test(
    h_src, h_dst, h_wgt, NULL, h_result, num_vertices, num_edges, 0, FALSE, FALSE, FALSE, FALSE, 6);
}

int test_betweenness_centrality_specific_normalized()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_seeds    = 2;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_seeds[]  = {0, 3};
  weight_t h_result[] = {0, 0.475, 0.2, 0.1, 0.05, 0.075};

  return generic_betweenness_centrality_test(h_src,
                                             h_dst,
                                             h_wgt,
                                             h_seeds,
                                             h_result,
                                             num_vertices,
                                             num_edges,
                                             num_seeds,
                                             FALSE,
                                             FALSE,
                                             TRUE,
                                             FALSE,
                                             num_seeds);
}

int test_betweenness_centrality_specific_unnormalized()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_seeds    = 2;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_seeds[]  = {0, 3};
  weight_t h_result[] = {0, 3.16667, 1.33333, 0.666667, 0.333333, 0.5};

  return generic_betweenness_centrality_test(h_src,
                                             h_dst,
                                             h_wgt,
                                             h_seeds,
                                             h_result,
                                             num_vertices,
                                             num_edges,
                                             num_seeds,
                                             FALSE,
                                             FALSE,
                                             FALSE,
                                             FALSE,
                                             num_seeds);
}

int test_betweenness_centrality_test_endpoints()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.166667, 0.3, 0.166667, 0.2, 0.166667, 0.166667};

  return generic_betweenness_centrality_test(
    h_src, h_dst, h_wgt, NULL, h_result, num_vertices, num_edges, 0, FALSE, FALSE, TRUE, TRUE, 6);
}

int test_betweenness_centrality_full_directed_normalized_karate()
{
  size_t num_edges    = 156;
  size_t num_vertices = 34;

  vertex_t h_src[] = {
    1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19, 21,
    30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30, 32, 33, 33, 33,
    32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 31, 29, 33, 33,
    31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,
    5,  5,  5,  6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
    23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};

  vertex_t h_dst[] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,
    1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,  8,  8,  9,  13,
    14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27,
    28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19,
    21, 31, 2,  3,  7,  13, 17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10,
    6,  10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
    32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};

  weight_t h_wgt[] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  weight_t h_result[] = {462.142914, 56.957146,  151.701584, 12.576191, 0.666667, 31.666668,
                         31.666668,  0.000000,   59.058739,  0.895238,  0.666667, 0.000000,
                         0.000000,   48.431747,  0.000000,   0.000000,  0.000000, 0.000000,
                         0.000000,   34.293652,  0.000000,   0.000000,  0.000000, 18.600000,
                         2.333333,   4.055556,   0.000000,   23.584126, 1.895238, 3.085714,
                         15.219049,  146.019043, 153.380981, 321.103180};

  return generic_betweenness_centrality_test(h_src,
                                             h_dst,
                                             h_wgt,
                                             NULL,
                                             h_result,
                                             num_vertices,
                                             num_edges,
                                             0,
                                             FALSE,
                                             FALSE,
                                             FALSE,
                                             FALSE,
                                             34);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_betweenness_centrality_full);
  result |= RUN_TEST(test_betweenness_centrality_full_directed);
  result |= RUN_TEST(test_betweenness_centrality_specific_normalized);
  result |= RUN_TEST(test_betweenness_centrality_specific_unnormalized);
  result |= RUN_TEST(test_betweenness_centrality_test_endpoints);
  result |= RUN_TEST(test_betweenness_centrality_full_directed_normalized_karate);
  return result;
}
