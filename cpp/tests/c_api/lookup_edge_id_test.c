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

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;

int generic_leiden_test(vertex_t* h_src,
                        vertex_t* h_dst,
                        edge_t* h_ids,
                        size_t num_vertices,
                        size_t num_edges,
                        edge_t* edge_ids_to_lookup,
                        size_t num_edge_ids_to_lookup,
                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;

  cugraph_graph_t* p_graph         = NULL;
  cugraph_vertex_pairs_t* p_result = NULL;

  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = create_sg_test_graph(p_handle,
                                  vertex_tid,
                                  edge_tid,
                                  h_src,
                                  h_dst,
                                  weight_tid,
                                  NULL,  // h_ids,
                                  edge_type_tid,
                                  NULL,
                                  edge_id_tid,
                                  h_ids,  // NULL,
                                  num_edges,
                                  store_transposed,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_t* d_edge_ids_to_lookup = NULL;

  cugraph_type_erased_device_array_view_t* d_edge_ids_to_lookup_view = NULL;

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, num_edge_ids_to_lookup, edge_id_tid, &d_edge_ids_to_lookup, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_edge_ids_to_lookup create failed.");

  d_edge_ids_to_lookup_view = cugraph_type_erased_device_array_view(d_edge_ids_to_lookup);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, d_edge_ids_to_lookup_view, (byte_t*)edge_ids_to_lookup, &ret_error);

  ret_code = cugraph_lookup_src_dst_from_edge_id(
    p_handle, p_graph, d_edge_ids_to_lookup_view, FALSE, &p_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  //

  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_lookup_src_dst_from_edge_id failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* srcs;
    cugraph_type_erased_device_array_view_t* dsts;

    srcs = cugraph_vertex_pairs_get_first(p_result);
    dsts = cugraph_vertex_pairs_get_second(p_result);

    vertex_t h_srcs[num_edge_ids_to_lookup];
    vertex_t h_dsts[num_edge_ids_to_lookup];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_srcs, srcs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_dsts, dsts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    for (int i = 0; i < num_edge_ids_to_lookup; i++) {
      printf("=> %d:  %d   %d\n", edge_ids_to_lookup[i], h_srcs[i], h_dsts[i]);
    }

    cugraph_vertex_pairs_free(p_result);
  }

  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_leiden()
{
  size_t num_edges    = 8;
  size_t num_vertices = 5;

  vertex_t h_src[]              = {0, 1, 1, 2, 1, 3, 4, 0};
  vertex_t h_dst[]              = {1, 3, 4, 0, 0, 1, 1, 2};
  edge_t h_ids[]                = {10, 13, 14, 20, 10, 13, 14, 20};
  edge_t edge_ids_to_lookup[]   = {10, 12, 14};
  size_t num_edge_ids_to_lookup = 3;

  // Louvain wants store_transposed = FALSE
  return generic_leiden_test(h_src,
                             h_dst,
                             h_ids,
                             num_vertices,
                             num_edges,
                             edge_ids_to_lookup,
                             num_edge_ids_to_lookup,
                             FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_leiden);
  return result;
}
