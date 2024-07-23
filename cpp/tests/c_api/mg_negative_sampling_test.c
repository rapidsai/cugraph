/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdbool.h>
#include <unistd.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

data_type_id_t vertex_tid    = INT32;
data_type_id_t edge_tid      = INT32;
data_type_id_t weight_tid    = FLOAT32;
data_type_id_t edge_id_tid   = INT32;
data_type_id_t edge_type_tid = INT32;

int generic_negative_sampling_test(const cugraph_resource_handle_t* handle,
                                   vertex_t* h_src,
                                   vertex_t* h_dst,
                                   size_t num_vertices,
                                   size_t num_edges,
                                   size_t num_samples,
                                   vertex_t* h_vertices,
                                   weight_t* h_src_bias,
                                   weight_t* h_dst_bias,
                                   size_t num_biases,
                                   bool_t remove_duplicates,
                                   bool_t remove_false_negatives,
                                   bool_t exact_number_of_samples)
{
  // Create graph
  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;
  cugraph_graph_t* graph        = NULL;
  cugraph_coo_t* result         = NULL;

  ret_code = create_mg_test_graph_new(handle,
                                      vertex_tid,
                                      edge_tid,
                                      h_src,
                                      h_dst,
                                      weight_tid,
                                      NULL,
                                      edge_type_tid,
                                      NULL,
                                      edge_id_tid,
                                      NULL,
                                      num_edges,
                                      FALSE,
                                      TRUE,
                                      FALSE,
                                      FALSE,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_vertices           = NULL;
  cugraph_type_erased_device_array_view_t* d_vertices_view = NULL;
  cugraph_type_erased_device_array_t* d_src_bias           = NULL;
  cugraph_type_erased_device_array_view_t* d_src_bias_view = NULL;
  cugraph_type_erased_device_array_t* d_dst_bias           = NULL;
  cugraph_type_erased_device_array_view_t* d_dst_bias_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  if (num_biases > 0) {
    if (rank == 0) {
      ret_code = cugraph_type_erased_device_array_create(
        handle, num_biases, vertex_tid, &d_vertices, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_vertices create failed.");

      d_vertices_view = cugraph_type_erased_device_array_view(d_vertices);

      ret_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, d_vertices_view, (byte_t*)h_vertices, &ret_error);

      ret_code = cugraph_type_erased_device_array_create(
        handle, num_biases, weight_tid, &d_src_bias, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_src_bias create failed.");

      d_src_bias_view = cugraph_type_erased_device_array_view(d_src_bias);

      ret_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, d_src_bias_view, (byte_t*)h_src_bias, &ret_error);

      ret_code = cugraph_type_erased_device_array_create(
        handle, num_biases, weight_tid, &d_dst_bias, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_dst_bias create failed.");

      d_dst_bias_view = cugraph_type_erased_device_array_view(d_dst_bias);

      ret_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, d_dst_bias_view, (byte_t*)h_dst_bias, &ret_error);
    } else {
      d_vertices_view = cugraph_type_erased_device_array_view_create(NULL, 0, vertex_tid);
      d_src_bias_view = cugraph_type_erased_device_array_view_create(NULL, 0, weight_tid);
      d_dst_bias_view = cugraph_type_erased_device_array_view_create(NULL, 0, weight_tid);
    }
  }

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  ret_code = cugraph_negative_sampling(handle,
                                       rng_state,
                                       graph,
                                       num_samples,
                                       d_vertices_view,
                                       d_src_bias_view,
                                       d_dst_bias_view,
                                       remove_duplicates,
                                       remove_false_negatives,
                                       exact_number_of_samples,
                                       FALSE,
                                       &result,
                                       &ret_error);

  cugraph_type_erased_device_array_view_t* result_srcs = NULL;
  cugraph_type_erased_device_array_view_t* result_dsts = NULL;

  result_srcs = cugraph_coo_get_sources(result);
  result_dsts = cugraph_coo_get_destinations(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);

  vertex_t h_result_srcs[result_size];
  vertex_t h_result_dsts[result_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_srcs, result_srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_dsts, result_dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  First, check that all edges are actually part of the graph
  int32_t M_exists[num_vertices][num_vertices];
  int32_t M_duplicates[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M_exists[i][j]     = 0;
      M_duplicates[i][j] = 0;
    }

  for (int i = 0; i < num_edges; ++i) {
    M_exists[h_src[i]][h_dst[i]] = 1;
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                (h_result_srcs[i] >= 0) && (h_result_srcs[i] < num_vertices),
                "negative_sampling generated an edge that with an invalid vertex");
    TEST_ASSERT(test_ret_value,
                (h_result_dsts[i] >= 0) && (h_result_dsts[i] < num_vertices),
                "negative_sampling generated an edge that with an invalid vertex");
    if (remove_false_negatives == TRUE) {
      TEST_ASSERT(test_ret_value,
                  M_exists[h_result_srcs[i]][h_result_dsts[i]] == 0,
                  "negative_sampling generated a false negative edge that should be suppressed");
    }

    if (remove_duplicates == TRUE) {
      TEST_ASSERT(test_ret_value,
                  M_duplicates[h_result_srcs[i]][h_result_dsts[i]] == 0,
                  "negative_sampling generated a duplicate edge that should be suppressed");
      M_duplicates[h_result_srcs[i]][h_result_dsts[i]] = 1;
    }
  }

  if (exact_number_of_samples == TRUE)
    TEST_ASSERT(test_ret_value,
                result_size == num_samples,
                "negative_sampling generated a result with an incorrect number of samples");

  cugraph_type_erased_device_array_view_free(d_vertices_view);
  cugraph_type_erased_device_array_view_free(d_src_bias_view);
  cugraph_type_erased_device_array_view_free(d_dst_bias_view);
  cugraph_type_erased_device_array_free(d_vertices);
  cugraph_type_erased_device_array_free(d_src_bias);
  cugraph_type_erased_device_array_free(d_dst_bias);
  cugraph_coo_free(result);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_negative_sampling_uniform(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t num_biases   = 0;
  size_t num_samples  = 10;

  vertex_t src[] = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[] = {1, 2, 3, 4, 0, 1, 3, 5, 5};

  bool_t remove_duplicates       = FALSE;
  bool_t remove_false_negatives  = TRUE;
  bool_t exact_number_of_samples = FALSE;

  return generic_negative_sampling_test(handle,
                                        src,
                                        dst,
                                        num_vertices,
                                        num_edges,
                                        num_samples,
                                        NULL,
                                        NULL,
                                        NULL,
                                        num_biases,
                                        remove_duplicates,
                                        remove_false_negatives,
                                        exact_number_of_samples);
}

int test_negative_sampling_biased(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t num_biases   = 6;
  size_t num_samples  = 10;

  vertex_t src[]      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]      = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  weight_t src_bias[] = {1, 1, 2, 2, 1, 1};
  weight_t dst_bias[] = {2, 2, 1, 1, 1, 1};
  vertex_t vertices[] = {0, 1, 2, 3, 4, 5};

  bool_t remove_duplicates       = FALSE;
  bool_t remove_false_negatives  = TRUE;
  bool_t exact_number_of_samples = FALSE;

  return generic_negative_sampling_test(handle,
                                        src,
                                        dst,
                                        num_vertices,
                                        num_edges,
                                        num_samples,
                                        vertices,
                                        src_bias,
                                        dst_bias,
                                        num_biases,
                                        remove_duplicates,
                                        remove_false_negatives,
                                        exact_number_of_samples);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_negative_sampling_uniform, handle);
  result |= RUN_MG_TEST(test_negative_sampling_biased, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
