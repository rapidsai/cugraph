/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

cugraph_data_type_id_t vertex_tid = INT32;
cugraph_data_type_id_t edge_tid   = INT32;
cugraph_data_type_id_t weight_tid = FLOAT32;

int generic_k_core_test(const cugraph_resource_handle_t* resource_handle,
                        vertex_t* h_src,
                        vertex_t* h_dst,
                        weight_t* h_wgt,
                        vertex_t* h_result_src,
                        vertex_t* h_result_dst,
                        weight_t* h_result_wgt,
                        size_t num_vertices,
                        size_t num_edges,
                        size_t num_result_edges,
                        size_t k,
                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                 = NULL;
  cugraph_core_result_t* core_result     = NULL;
  cugraph_k_core_result_t* k_core_result = NULL;

  ret_code = create_mg_test_graph_new(resource_handle,
                                      vertex_tid,
                                      edge_tid,
                                      h_src,
                                      h_dst,
                                      weight_tid,
                                      h_wgt,
                                      INT32,
                                      NULL,
                                      edge_tid,
                                      NULL,
                                      INT32,
                                      NULL,
                                      NULL,
                                      num_edges,
                                      store_transposed,
                                      TRUE,
                                      TRUE,
                                      FALSE,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_core_number(
    resource_handle, graph, K_CORE_DEGREE_TYPE_IN, FALSE, &core_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_core_number failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_k_core(resource_handle,
                            graph,
                            k,
                            K_CORE_DEGREE_TYPE_IN,
                            core_result,
                            FALSE,
                            &k_core_result,
                            &ret_error);

  cugraph_type_erased_device_array_view_t* src_vertices;
  cugraph_type_erased_device_array_view_t* dst_vertices;
  cugraph_type_erased_device_array_view_t* weights;

  src_vertices = cugraph_k_core_result_get_src_vertices(k_core_result);
  dst_vertices = cugraph_k_core_result_get_dst_vertices(k_core_result);
  weights      = cugraph_k_core_result_get_weights(k_core_result);

  size_t number_of_result_edges = cugraph_type_erased_device_array_view_size(src_vertices);

  vertex_t h_src_vertices[number_of_result_edges];
  vertex_t h_dst_vertices[number_of_result_edges];
  weight_t h_weights[number_of_result_edges];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_src_vertices, src_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_dst_vertices, dst_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    resource_handle, (byte_t*)h_weights, weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  weight_t M[num_vertices][num_vertices];
  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = 0;

  for (int i = 0; i < num_result_edges; ++i)
    M[h_result_src[i]][h_result_dst[i]] = h_result_wgt[i];

  for (int i = 0; (i < number_of_result_edges) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_src_vertices[i]][h_dst_vertices[i]] == h_weights[i],
                "edge does not match");
  }

  cugraph_k_core_result_free(k_core_result);
  cugraph_core_result_free(core_result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_k_core(const cugraph_resource_handle_t* resource_handle)
{
  size_t num_edges        = 22;
  size_t num_vertices     = 7;
  size_t num_result_edges = 12;
  size_t k                = 3;

  vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5, 3, 1, 4, 5, 5, 6};
  vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 3, 1, 6, 5};
  weight_t h_wgt[]        = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_result_src[] = {1, 1, 3, 4, 3, 4, 3, 4, 5, 5, 1, 5};
  vertex_t h_result_dst[] = {3, 4, 5, 5, 1, 3, 4, 1, 3, 4, 5, 1};
  weight_t h_result_wgt[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  return generic_k_core_test(resource_handle,
                             h_src,
                             h_dst,
                             h_wgt,
                             h_result_src,
                             h_result_dst,
                             h_result_wgt,
                             num_vertices,
                             num_edges,
                             num_result_edges,
                             k,
                             FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_k_core, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
