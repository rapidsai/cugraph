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

int generic_edge_betweenness_centrality_test(vertex_t* h_src,
                                             vertex_t* h_dst,
                                             weight_t* h_wgt,
                                             vertex_t* h_seeds,
                                             weight_t* h_result,
                                             size_t num_vertices,
                                             size_t num_edges,
                                             size_t num_seeds,
                                             bool_t store_transposed,
                                             size_t num_vertices_to_sample)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle                   = NULL;
  cugraph_graph_t* graph                              = NULL;
  cugraph_edge_centrality_result_t* result            = NULL;
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
                               FALSE,
                               &graph,
                               &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (h_seeds == NULL) {
    ret_code = cugraph_select_random_vertices(
      handle, graph, rng_state, num_vertices_to_sample, &seeds, &ret_error);
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

  ret_code = cugraph_edge_betweenness_centrality(
    handle, graph, seeds_view, FALSE, FALSE, &result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_edge_betweenness_centrality failed.");

  cugraph_type_erased_device_array_view_t* srcs;
  cugraph_type_erased_device_array_view_t* dsts;
  cugraph_type_erased_device_array_view_t* centralities;

  srcs         = cugraph_edge_centrality_result_get_src_vertices(result);
  dsts         = cugraph_edge_centrality_result_get_dst_vertices(result);
  centralities = cugraph_edge_centrality_result_get_values(result);

  size_t num_local_edges = cugraph_type_erased_device_array_view_size(srcs);

  vertex_t h_cugraph_src[num_local_edges];
  vertex_t h_cugraph_dst[num_local_edges];
  weight_t h_centralities[num_local_edges];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_cugraph_src, srcs , &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_cugraph_dst, dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_centralities, centralities, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M[i][j]         = 0.0;
    }

  for (int i = 0; i < num_edges; ++i) {
    M[h_src[i]][h_dst[i]] = h_result[i];
  }

  for (int i = 0; (i < num_local_edges) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(M[h_cugraph_src[i]][h_cugraph_dst[i]], h_centralities[i], 0.001),
                "betweenness centrality results don't match");
  }

  cugraph_edge_centrality_result_free(result);
  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_edge_betweenness_centrality()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = { 0, 2, 3, 1.83333, 2, 2, 3, 2, 3.16667, 2.83333, 4.33333, 0, 2, 2.83333, 3.66667, 2.33333 };

  double epsilon        = 1e-6;
  size_t max_iterations = 200;

  // Eigenvector centrality wants store_transposed = TRUE
  return generic_edge_betweenness_centrality_test(
    h_src, h_dst, h_wgt, NULL, h_result, num_vertices, num_edges, 0, TRUE, 5);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_edge_betweenness_centrality);
  return result;
}
