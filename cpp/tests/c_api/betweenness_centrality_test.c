/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
    if (isnan(h_result[h_vertices[i]])) {
      TEST_ASSERT(test_ret_value, isnan(h_centralities[i]), "expected NaN, got a non-NaN value");
    } else {
      if (!nearlyEqual(h_result[h_vertices[i]], h_centralities[i], 0.0001))
        printf("  expected: %g, got %g\n", h_result[h_vertices[i]], h_centralities[i]);

      TEST_ASSERT(test_ret_value,
                  nearlyEqual(h_result[h_vertices[i]], h_centralities[i], 0.0001),
                  "centralities results don't match");
    }
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
  weight_t h_result[] = {0, 0.395833, 0.166667, 0.166667, 0.0416667, 0.0625};

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
  weight_t h_result[] = {0, 7.91667, 3.33333, 3.33333, 0.833333, 1.25};

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

int test_issue_4941()
{
  size_t num_edges_asymmetric = 4;
  size_t num_edges_symmetric  = 8;
  size_t num_vertices         = 5;

  vertex_t h_src_asymmetric[] = {1, 2, 3, 4};
  vertex_t h_dst_asymmetric[] = {0, 0, 0, 0};
  vertex_t h_src_symmetric[]  = {1, 2, 3, 4, 0, 0, 0, 0};
  vertex_t h_dst_symmetric[]  = {0, 0, 0, 0, 1, 2, 3, 4};
  weight_t h_wgt[]            = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_seeds[]          = {1};

  struct variations {
    bool_t normalized;
    bool_t endpoints;
    bool_t is_directed;
    int k;
    weight_t results[5];
  };

  struct variations test_list[] = {
    {TRUE, TRUE, TRUE, 0, {1.0, 0.4, 0.4, 0.4, 0.4}},
    {TRUE, TRUE, TRUE, 1, {1.0, 1.0, 0.25, 0.25, 0.25}},
    {TRUE, TRUE, FALSE, 0, {1.0, 0.4, 0.4, 0.4, 0.4}},
    {TRUE, TRUE, FALSE, 1, {1.0, 1.0, 0.25, 0.25, 0.25}},
    {TRUE, FALSE, TRUE, 0, {1.0, 0.0, 0.0, 0.0, 0.0}},
    {TRUE, FALSE, TRUE, 1, {1.0, NAN, 0.0, 0.0, 0.0}},
    {TRUE, FALSE, FALSE, 0, {1.0, 0.0, 0.0, 0.0, 0.0}},
    {TRUE, FALSE, FALSE, 1, {1.0, NAN, 0.0, 0.0, 0.0}},
    {FALSE, TRUE, TRUE, 0, {20.0, 8.0, 8.0, 8.0, 8.0}},
    {FALSE, TRUE, TRUE, 1, {20.0, 20.0, 5.0, 5.0, 5.0}},
    {FALSE, TRUE, FALSE, 0, {10.0, 4.0, 4.0, 4.0, 4.0}},
    {FALSE, TRUE, FALSE, 1, {10.0, 10.0, 2.5, 2.5, 2.5}},
    {FALSE, FALSE, TRUE, 0, {12.0, 0.0, 0.0, 0.0, 0.0}},
    {FALSE, FALSE, TRUE, 1, {12, NAN, 0.0, 0.0, 0.0}},
    {FALSE, FALSE, FALSE, 0, {6.0, 0.0, 0.0, 0.0, 0.0}},
    {FALSE, FALSE, FALSE, 1, {6.0, NAN, 0.0, 0.0, 0.0}},
  };

  int test_result = 0;

  for (size_t i = 0; (test_result == 0) && (i < (sizeof(test_list) / sizeof(test_list[0]))); ++i) {
    test_result = generic_betweenness_centrality_test(h_src_symmetric,
                                                      h_dst_symmetric,
                                                      h_wgt,
                                                      (test_list[i].k == 0) ? NULL : h_seeds,
                                                      test_list[i].results,
                                                      num_vertices,
                                                      num_edges_symmetric,
                                                      test_list[i].k,
                                                      FALSE,
                                                      !test_list[i].is_directed,
                                                      test_list[i].normalized,
                                                      test_list[i].endpoints,
                                                      num_vertices);
    test_result = 0;
  }

  return test_result;
}

int test_issue_4941_with_endpoints()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {5, 0, 1, 2, 4, 0, 3, 3};
  vertex_t h_dst[]    = {0, 1, 2, 4, 3, 3, 5, 2};
  weight_t h_wgt[]    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_seeds[]  = {5};
  weight_t h_result[] = {1.0, .4, .4, .4, .2, 1.0};

  return generic_betweenness_centrality_test(h_src,
                                             h_dst,
                                             h_wgt,
                                             h_seeds,
                                             h_result,
                                             num_vertices,
                                             num_edges,
                                             1,
                                             FALSE,
                                             FALSE,
                                             TRUE,
                                             TRUE,
                                             0);
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
  result |= RUN_TEST(test_issue_4941);
  result |= RUN_TEST(test_issue_4941_with_endpoints);
  return result;
}
