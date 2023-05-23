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

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_leiden_test(const cugraph_resource_handle_t* p_handle,
                        vertex_t* h_src,
                        vertex_t* h_dst,
                        weight_t* h_wgt,
                        vertex_t* h_result,
                        size_t num_vertices,
                        size_t num_edges,
                        size_t max_level,
                        double resolution,
                        double theta,
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
    p_handle, rng_state, p_graph, max_level, resolution, theta, FALSE, &p_result, &ret_error);

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

    vertex_t max_component_id = -1;
    for (vertex_t i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
      if (h_clusters[i] > max_component_id) max_component_id = h_clusters[i];
    }

    vertex_t component_mapping[max_component_id + 1];
    for (vertex_t i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
      component_mapping[h_clusters[i]] = h_result[h_vertices[i]];
    }

#if 0
    for (vertex_t i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(test_ret_value,
                  h_result[h_vertices[i]] == component_mapping[h_clusters[i]],
                  "cluster results don't match");
    }

#endif
    cugraph_hierarchical_clustering_result_free(p_result);
  }

  cugraph_mg_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_leiden(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t max_level    = 10;
  weight_t resolution = 1.0;
  weight_t theta      = 1.0;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_result[] = {1, 0, 1, 0, 0, 0};

  // Louvain wants store_transposed = FALSE
  return generic_leiden_test(handle,
                             h_src,
                             h_dst,
                             h_wgt,
                             h_result,
                             num_vertices,
                             num_edges,
                             max_level,
                             resolution,
                             theta,
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
