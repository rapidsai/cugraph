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
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_two_hop_nbr_test(vertex_t* h_src,
                             vertex_t* h_dst,
                             weight_t* h_wgt,
                             vertex_t* h_sources,
                             vertex_t* h_result_v1,
                             vertex_t* h_result_v2,
                             size_t num_vertices,
                             size_t num_edges,
                             size_t num_sources,
                             size_t num_result_pairs,
                             bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* resource_handle                   = NULL;
  cugraph_graph_t* graph                                       = NULL;
  cugraph_type_erased_device_array_t* start_vertices           = NULL;
  cugraph_type_erased_device_array_view_t* start_vertices_view = NULL;
  cugraph_vertex_pairs_t* result                               = NULL;

  resource_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, resource_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(resource_handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               num_edges,
                               store_transposed,
                               FALSE,
                               TRUE,
                               &graph,
                               &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (num_sources > 0) {
    ret_code = cugraph_type_erased_device_array_create(
      resource_handle, num_sources, INT32, &start_vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

    start_vertices_view = cugraph_type_erased_device_array_view(start_vertices);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      resource_handle, start_vertices_view, (byte_t*)h_sources, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");
  }

  ret_code = cugraph_two_hop_neighbors(
    resource_handle, graph, start_vertices_view, FALSE, &result, &ret_error);

  cugraph_type_erased_device_array_view_t* v1;
  cugraph_type_erased_device_array_view_t* v2;

  v1 = cugraph_vertex_pairs_get_first(result);
  v2 = cugraph_vertex_pairs_get_second(result);

  size_t number_of_pairs = cugraph_type_erased_device_array_view_size(v1);

  vertex_t h_v1[number_of_pairs];
  vertex_t h_v2[number_of_pairs];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_v1, v1, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_v2, v2, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  bool_t M[num_vertices][num_vertices];
  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = FALSE;

  for (int i = 0; i < num_result_pairs; ++i)
    M[h_result_v1[i]][h_result_v2[i]] = TRUE;

  TEST_ASSERT(test_ret_value, number_of_pairs == num_result_pairs, "results are different sizes");

  for (int i = 0; (i < number_of_pairs) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value, M[h_v1[i]][h_v2[i]], "result not found");
  }

  cugraph_vertex_pairs_free(result);
  cugraph_type_erased_device_array_view_free(start_vertices_view);
  cugraph_type_erased_device_array_free(start_vertices);
  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(resource_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_two_hop_nbr_all()
{
  size_t num_edges        = 22;
  size_t num_vertices     = 7;
  size_t num_sources      = 0;
  size_t num_result_pairs = 43;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
  weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3,
                            3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6};
  vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2,
                            3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 1, 3, 4, 6};

  return generic_two_hop_nbr_test(h_src,
                                  h_dst,
                                  h_wgt,
                                  NULL,
                                  h_result_v1,
                                  h_result_v2,
                                  num_vertices,
                                  num_edges,
                                  num_sources,
                                  num_result_pairs,
                                  FALSE);
}

int test_two_hop_nbr_one()
{
  size_t num_edges        = 22;
  size_t num_vertices     = 7;
  size_t num_sources      = 1;
  size_t num_result_pairs = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
  weight_t h_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  vertex_t h_sources[] = {0};

  vertex_t h_result_v1[] = {0, 0, 0, 0, 0, 0};
  vertex_t h_result_v2[] = {0, 1, 2, 3, 4, 5};

  return generic_two_hop_nbr_test(h_src,
                                  h_dst,
                                  h_wgt,
                                  h_sources,
                                  h_result_v1,
                                  h_result_v2,
                                  num_vertices,
                                  num_edges,
                                  num_sources,
                                  num_result_pairs,
                                  FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_two_hop_nbr_all);
  result |= RUN_TEST(test_two_hop_nbr_one);
  return result;
}
