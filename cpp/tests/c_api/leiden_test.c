/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */
#include "cugraph_c/types.h"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

/*
 * Runs cugraph_leiden on the (symmetric) graph and checks the result against the expected
 * clustering (h_result, compared up to a bijection of the cluster ids) and the expected
 * modularity, which is exact (the modularity of the returned clustering), so the tolerance is
 * tight. Leiden is deterministic, so a second run with a fresh random state of the same seed must
 * return the identical clustering.
 */
int generic_leiden_test(vertex_t* h_src,
                        vertex_t* h_dst,
                        weight_t* h_wgt,
                        vertex_t* h_result,
                        double expected_modularity,
                        size_t num_vertices,
                        size_t num_edges,
                        int32_t n_iterations,
                        double resolution,
                        double beta,
                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle                = NULL;
  cugraph_rng_state_t* p_rng_state                   = NULL;
  cugraph_graph_t* p_graph                           = NULL;
  cugraph_hierarchical_clustering_result_t* p_result = NULL;

  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  vertex_tid,
                                  edge_tid,
                                  h_src,
                                  h_dst,
                                  weight_tid,
                                  h_wgt,
                                  edge_type_tid,
                                  NULL,
                                  edge_id_tid,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  store_transposed,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  vertex_t h_first_clusters[num_vertices];

  for (int run = 0; (run < 2) && (test_ret_value == 0); ++run) {
    ret_code = cugraph_rng_state_create(p_handle, 0, &p_rng_state, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
    TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    ret_code = cugraph_leiden(
      p_handle, p_rng_state, p_graph, n_iterations, resolution, beta, FALSE, &p_result, &ret_error);

    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
    TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_leiden failed.");

    if (test_ret_value == 0) {
      cugraph_type_erased_device_array_view_t* vertices;
      cugraph_type_erased_device_array_view_t* clusters;

      vertices          = cugraph_hierarchical_clustering_result_get_vertices(p_result);
      clusters          = cugraph_hierarchical_clustering_result_get_clusters(p_result);
      double modularity = cugraph_hierarchical_clustering_result_get_modularity(p_result);

      vertex_t h_vertices[num_vertices];
      vertex_t h_clusters[num_vertices];

      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        p_handle, (byte_t*)h_vertices, vertices, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        p_handle, (byte_t*)h_clusters, clusters, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

      TEST_ASSERT(test_ret_value,
                  nearlyEqualDouble(modularity, expected_modularity, 1e-12),
                  "modularity doesn't match");

      // the clustering equals the expected one up to a bijection of the cluster ids
      vertex_t to_expected[num_vertices];
      vertex_t to_result[num_vertices];
      for (size_t i = 0; i < num_vertices; ++i) {
        to_expected[i] = -1;
        to_result[i]   = -1;
      }
      for (size_t i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
        vertex_t v = h_vertices[i];
        vertex_t c = h_clusters[i];
        TEST_ASSERT(
          test_ret_value, (c >= 0) && (c < (vertex_t)num_vertices), "cluster id out of range");
        if (test_ret_value != 0) break;
        if (to_expected[c] == -1) to_expected[c] = h_result[v];
        if (to_result[h_result[v]] == -1) to_result[h_result[v]] = c;
        TEST_ASSERT(test_ret_value,
                    (to_expected[c] == h_result[v]) && (to_result[h_result[v]] == c),
                    "cluster results don't match");
      }

      // deterministic: a fresh random state with the same seed gives the identical clustering
      for (size_t i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
        if (run == 0) {
          h_first_clusters[h_vertices[i]] = h_clusters[i];
        } else {
          TEST_ASSERT(test_ret_value,
                      h_first_clusters[h_vertices[i]] == h_clusters[i],
                      "same seed, different clustering");
        }
      }

      cugraph_hierarchical_clustering_result_free(p_result);
    }
    cugraph_rng_state_free(p_rng_state);
  }

  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

/*
 * Runs cugraph_leiden with an invalid parameter, which must fail with CUGRAPH_INVALID_INPUT.
 */
int generic_leiden_invalid_parameter_test(int32_t n_iterations, double resolution, double beta)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                = NULL;
  cugraph_rng_state_t* p_rng_state                   = NULL;
  cugraph_graph_t* p_graph                           = NULL;
  cugraph_hierarchical_clustering_result_t* p_result = NULL;

  vertex_t h_src[] = {0, 1, 1, 2};
  vertex_t h_dst[] = {1, 0, 2, 1};
  weight_t h_wgt[] = {1.0f, 1.0f, 1.0f, 1.0f};

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = cugraph_rng_state_create(p_handle, 0, &p_rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  INT32,
                                  INT32,
                                  h_src,
                                  h_dst,
                                  FLOAT32,
                                  h_wgt,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  4,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");

  ret_code = cugraph_leiden(
    p_handle, p_rng_state, p_graph, n_iterations, resolution, beta, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value,
              ret_code == CUGRAPH_INVALID_INPUT,
              "cugraph_leiden accepted an invalid parameter.");

  if (ret_code == CUGRAPH_SUCCESS) cugraph_hierarchical_clustering_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_rng_state_free(p_rng_state);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_leiden()
{
  size_t num_edges     = 16;
  size_t num_vertices  = 6;
  int32_t n_iterations = 2;
  double resolution    = 1.0;
  double beta          = 0.0;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_result[]        = {0, 0, 0, 1, 1, 1};
  double expected_modularity = 0.21596893567080255;

  // Leiden wants store_transposed = FALSE
  return generic_leiden_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result,
                             expected_modularity,
                             num_vertices,
                             num_edges,
                             n_iterations,
                             resolution,
                             beta,
                             FALSE);
}

int test_leiden_no_weights()
{
  size_t num_edges     = 16;
  size_t num_vertices  = 6;
  int32_t n_iterations = 2;
  double resolution    = 1.0;
  double beta          = 0.0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_result[]        = {0, 0, 0, 0, 1, 1};
  double expected_modularity = 0.125;

  // Leiden wants store_transposed = FALSE
  return generic_leiden_test(h_src,
                             h_dst,
                             NULL,
                             h_result,
                             expected_modularity,
                             num_vertices,
                             num_edges,
                             n_iterations,
                             resolution,
                             beta,
                             FALSE);
}

int test_leiden_until_stable()
{
  size_t num_edges     = 16;
  size_t num_vertices  = 6;
  int32_t n_iterations = -1;
  double resolution    = 1.0;
  double beta          = 0.0;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_result[]        = {0, 0, 0, 1, 1, 1};
  double expected_modularity = 0.21596893567080255;

  return generic_leiden_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result,
                             expected_modularity,
                             num_vertices,
                             num_edges,
                             n_iterations,
                             resolution,
                             beta,
                             FALSE);
}

int test_leiden_rejects_beta() { return generic_leiden_invalid_parameter_test(2, 1.0, 1.0); }

int test_leiden_rejects_n_iterations()
{
  return generic_leiden_invalid_parameter_test(0, 1.0, 0.0);
}

int test_leiden_rejects_resolution() { return generic_leiden_invalid_parameter_test(2, -1.0, 0.0); }

int test_leiden_rejects_nan_resolution()
{
  return generic_leiden_invalid_parameter_test(2, NAN, 0.0);
}

int test_leiden_rejects_negative_beta()
{
  return generic_leiden_invalid_parameter_test(2, 1.0, -1.0);
}

int test_leiden_rejects_negative_n_iterations()
{
  return generic_leiden_invalid_parameter_test(-2, 1.0, 0.0);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_leiden);
  result |= RUN_TEST(test_leiden_no_weights);
  result |= RUN_TEST(test_leiden_until_stable);
  result |= RUN_TEST(test_leiden_rejects_beta);
  result |= RUN_TEST(test_leiden_rejects_n_iterations);
  result |= RUN_TEST(test_leiden_rejects_resolution);
  result |= RUN_TEST(test_leiden_rejects_nan_resolution);
  result |= RUN_TEST(test_leiden_rejects_negative_beta);
  result |= RUN_TEST(test_leiden_rejects_negative_n_iterations);
  return result;
}
