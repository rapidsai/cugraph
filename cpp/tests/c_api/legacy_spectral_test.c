/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

int generic_spectral_test(vertex_t* h_src,
                          vertex_t* h_dst,
                          weight_t* h_wgt,
                          vertex_t* h_result,
                          weight_t expected_modularity,
                          weight_t expected_edge_cut,
                          weight_t expected_ratio_cut,
                          size_t num_vertices,
                          size_t num_edges,
                          size_t num_clusters,
                          size_t num_eigenvectors,
                          double evs_tolerance,
                          int evs_max_iterations,
                          double k_means_tolerance,
                          int k_means_max_iterations,
                          bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle   = NULL;
  cugraph_graph_t* graph              = NULL;
  cugraph_clustering_result_t* result = NULL;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(handle, vertex_tid, edge_tid, h_src, h_dst, weight_tid, h_wgt, edge_type_tid, NULL, edge_id_tid, NULL, num_edges, store_transposed, FALSE, FALSE, FALSE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_spectral_modularity_maximization(handle, graph, num_clusters, num_eigenvectors,
                                             evs_tolerance, evs_max_iterations, k_means_tolerance,
                                             k_means_max_iterations, FALSE, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_spectral_modularity_maximization failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* vertices;
    cugraph_type_erased_device_array_view_t* clusters;
    double modularity;
    double edge_cut;
    double ratio_cut;

    vertices = cugraph_clustering_result_get_vertices(result);
    clusters = cugraph_clustering_result_get_clusters(result);

    ret_code = cugraph_analyze_clustering_modularity(handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    vertex_t h_vertices[num_vertices];
    edge_t h_clusters[num_vertices];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_vertices, vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_clusters, clusters, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(
        test_ret_value, h_result[h_vertices[i]] == h_clusters[i], "cluster results don't match");
    }

    TEST_ASSERT(test_ret_value,
                nearlyEqual(modularity, expected_modularity, 0.001),
                "modularity doesn't match");

    TEST_ASSERT(test_ret_value,
                nearlyEqual(edge_cut, expected_edge_cut, 0.001),
                "edge_cut doesn't match");

    TEST_ASSERT(test_ret_value,
                nearlyEqual(ratio_cut, expected_ratio_cut, 0.001),
                "ratio_cut doesn't match");

    cugraph_clustering_result_free(result);
  }

  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_balanced_cut_test(vertex_t* h_src,
                              vertex_t* h_dst,
                              weight_t* h_wgt,
                              vertex_t* h_result,
                              weight_t expected_modularity,
                              weight_t expected_edge_cut,
                              weight_t expected_ratio_cut,
                              size_t num_vertices,
                              size_t num_edges,
                              size_t num_clusters,
                              size_t num_eigenvectors,
                              double evs_tolerance,
                              int evs_max_iterations,
                              double k_means_tolerance,
                              int k_means_max_iterations,
                              bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  cugraph_resource_handle_t* handle   = NULL;
  cugraph_graph_t* graph              = NULL;
  cugraph_clustering_result_t* result = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(handle, vertex_tid, edge_tid, h_src, h_dst, weight_tid, h_wgt, edge_type_tid, NULL, edge_id_tid, NULL, num_edges, store_transposed, FALSE, FALSE, FALSE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_balanced_cut_clustering(handle, graph, num_clusters, num_eigenvectors,
                                    evs_tolerance, evs_max_iterations, k_means_tolerance,
                                    k_means_max_iterations, FALSE, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_spectral_modularity_maximization failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* vertices;
    cugraph_type_erased_device_array_view_t* clusters;
    double modularity;
    double edge_cut;
    double ratio_cut;

    vertices = cugraph_clustering_result_get_vertices(result);
    clusters = cugraph_clustering_result_get_clusters(result);

    ret_code = cugraph_analyze_clustering_modularity(handle, graph, num_clusters, vertices, clusters, &modularity, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    ret_code = cugraph_analyze_clustering_edge_cut(handle, graph, num_clusters, vertices, clusters, &edge_cut, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    ret_code = cugraph_analyze_clustering_ratio_cut(handle, graph, num_clusters, vertices, clusters, &ratio_cut, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    vertex_t h_vertices[num_vertices];
    edge_t h_clusters[num_vertices];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_vertices, vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_clusters, clusters, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(
        test_ret_value, h_result[h_vertices[i]] == h_clusters[i], "cluster results don't match");
    }

    TEST_ASSERT(test_ret_value,
                nearlyEqual(modularity, expected_modularity, 0.001),
                "modularity doesn't match");

    TEST_ASSERT(test_ret_value,
                nearlyEqual(edge_cut, expected_edge_cut, 0.001),
                "edge_cut doesn't match");

    TEST_ASSERT(test_ret_value,
                nearlyEqual(ratio_cut, expected_ratio_cut, 0.001),
                "ratio_cut doesn't match");

    cugraph_clustering_result_free(result);
  }

  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_spectral()
{
  size_t num_clusters        = 2;
  size_t num_eigenvectors    = 2;
  size_t num_edges           = 14;
  size_t num_vertices        = 6;
  double evs_tolerance       = 0.001;
  int evs_max_iterations     = 100;
  double k_means_tolerance   = 0.001;
  int k_means_max_iterations = 100;

  vertex_t h_src[] = { 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5 };
  vertex_t h_dst[] = { 1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4 };
  weight_t h_wgt[] = { 0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f };
  vertex_t h_result[]          = { 0, 0, 0, 1, 1, 1 };
  weight_t expected_modularity = 0.136578;
  weight_t expected_edge_cut = 0;
  weight_t expected_ratio_cut = 0;

  // spectral clustering wants store_transposed = FALSE
  return generic_spectral_test(h_src,
                               h_dst,
                               h_wgt,
                               h_result,
                               expected_modularity,
                               expected_edge_cut,
                               expected_ratio_cut,
                               num_vertices,
                               num_edges,
                               num_clusters,
                               num_eigenvectors,
                               evs_tolerance,
                               evs_max_iterations,
                               k_means_tolerance,
                               k_means_max_iterations,
                               FALSE);
}

int test_balanced_cut_unequal_weight()
{
  size_t num_clusters        = 2;
  size_t num_eigenvectors    = 2;
  size_t num_edges           = 14;
  size_t num_vertices        = 6;
  double evs_tolerance       = 0.001;
  int evs_max_iterations     = 100;
  double k_means_tolerance   = 0.001;
  int k_means_max_iterations = 100;

  vertex_t h_src[] = { 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5 };
  vertex_t h_dst[] = { 1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4 };
  weight_t h_wgt[] = { 0.1f, 0.2f, 0.1f, 1.2f, 0.2f, 1.2f, 2.3f, 2.3f, 3.4f, 3.5f, 3.4f, 4.5f, 3.5f, 4.5f };
  vertex_t h_result[]          = { 0, 0, 1, 0, 0, 0 };
  weight_t expected_modularity = -0.02963;
  weight_t expected_edge_cut = 3.7;
  weight_t expected_ratio_cut = 4.44;

  // balanced cut clustering wants store_transposed = FALSE
  return generic_balanced_cut_test(h_src,
                                   h_dst,
                                   h_wgt,
                                   h_result,
                                   expected_modularity,
                                   expected_edge_cut,
                                   expected_ratio_cut,
                                   num_vertices,
                                   num_edges,
                                   num_clusters,
                                   num_eigenvectors,
                                   evs_tolerance,
                                   evs_max_iterations,
                                   k_means_tolerance,
                                   k_means_max_iterations,
                                   FALSE);
}

int test_balanced_cut_equal_weight()
{
  size_t num_clusters        = 2;
  size_t num_eigenvectors    = 2;
  size_t num_edges           = 14;
  size_t num_vertices        = 6;
  double evs_tolerance       = 0.001;
  int evs_max_iterations     = 100;
  double k_means_tolerance   = 0.001;
  int k_means_max_iterations = 100;

  vertex_t h_src[] = { 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5 };
  vertex_t h_dst[] = { 1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4 };
  weight_t h_wgt[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  vertex_t h_result[]          = { 1, 1, 1, 0, 0, 0 };
  weight_t expected_modularity = 0.357143;
  weight_t expected_edge_cut = 1;
  weight_t expected_ratio_cut = 0.666667;

  // balanced cut clustering wants store_transposed = FALSE
  return generic_balanced_cut_test(h_src,
                                   h_dst,
                                   h_wgt,
                                   h_result,
                                   expected_modularity,
                                   expected_edge_cut,
                                   expected_ratio_cut,
                                   num_vertices,
                                   num_edges,
                                   num_clusters,
                                   num_eigenvectors,
                                   evs_tolerance,
                                   evs_max_iterations,
                                   k_means_tolerance,
                                   k_means_max_iterations,
                                   FALSE);
}

int test_balanced_cut_no_weight()
{
  size_t num_clusters        = 2;
  size_t num_eigenvectors    = 2;
  size_t num_edges           = 14;
  size_t num_vertices        = 6;
  double evs_tolerance       = 0.001;
  int evs_max_iterations     = 100;
  double k_means_tolerance   = 0.001;
  int k_means_max_iterations = 100;

  vertex_t h_src[] = { 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5 };
  vertex_t h_dst[] = { 1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4 };
  vertex_t h_result[]          = { 1, 1, 1, 0, 0, 0 };
  weight_t expected_modularity = 0.357143;
  weight_t expected_edge_cut = 1;
  weight_t expected_ratio_cut = 0.666667;

  // balanced cut clustering wants store_transposed = FALSE
  return generic_balanced_cut_test(h_src,
                                   h_dst,
                                   NULL,
                                   h_result,
                                   expected_modularity,
                                   expected_edge_cut,
                                   expected_ratio_cut,
                                   num_vertices,
                                   num_edges,
                                   num_clusters,
                                   num_eigenvectors,
                                   evs_tolerance,
                                   evs_max_iterations,
                                   k_means_tolerance,
                                   k_means_max_iterations,
                                   FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_spectral);
  result |= RUN_TEST(test_balanced_cut_equal_weight);
  result |= RUN_TEST(test_balanced_cut_unequal_weight);
  result |= RUN_TEST(test_balanced_cut_no_weight);
  return result;
}
