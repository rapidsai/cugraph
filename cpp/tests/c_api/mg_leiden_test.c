/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

/*
 * Multi-GPU Leiden gathers the graph and runs the single-GPU algorithm with the seed of rank 0 on
 * every rank, so the clustering is that of the single-GPU test (leiden_test.c) on this graph, up
 * to a bijection of the cluster ids, and the modularity is exact.
 */
int generic_leiden_test(const cugraph_resource_handle_t* p_handle,
                        vertex_t* h_src,
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

  cugraph_graph_t* p_graph                           = NULL;
  cugraph_hierarchical_clustering_result_t* p_result = NULL;

  int rank = cugraph_resource_handle_get_rank(p_handle);
  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(p_handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = create_mg_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_leiden(
    p_handle, rng_state, p_graph, n_iterations, resolution, beta, FALSE, &p_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_leiden failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* vertices;
    cugraph_type_erased_device_array_view_t* clusters;

    vertices          = cugraph_hierarchical_clustering_result_get_vertices(p_result);
    clusters          = cugraph_hierarchical_clustering_result_get_clusters(p_result);
    double modularity = cugraph_hierarchical_clustering_result_get_modularity(p_result);

    vertex_t h_vertices[num_vertices];
    edge_t h_clusters[num_vertices];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_vertices, vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_clusters, clusters, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    size_t num_local_vertices = cugraph_type_erased_device_array_view_size(vertices);

    TEST_ASSERT(test_ret_value,
                nearlyEqualDouble(modularity, expected_modularity, 1e-12),
                "modularity doesn't match");

    // the local clustering equals the expected one up to a bijection of the cluster ids
    vertex_t to_expected[num_vertices];
    vertex_t to_result[num_vertices];
    for (size_t i = 0; i < num_vertices; ++i) {
      to_expected[i] = -1;
      to_result[i]   = -1;
    }
    for (size_t i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
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

    cugraph_hierarchical_clustering_result_free(p_result);
  }

  cugraph_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_leiden(const cugraph_resource_handle_t* handle)
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
  return generic_leiden_test(handle,
                             h_src,
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

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_leiden, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
