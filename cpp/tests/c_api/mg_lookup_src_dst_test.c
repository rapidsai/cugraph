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

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef int32_t edge_type_t;

typedef float weight_t;

data_type_id_t vertex_tid    = INT32;
data_type_id_t edge_tid      = INT32;
data_type_id_t weight_tid    = FLOAT32;
data_type_id_t edge_id_tid   = INT32;
data_type_id_t edge_type_tid = INT32;

int generic_lookup_src_dst_test(const cugraph_resource_handle_t* handle,
                                vertex_t* h_srcs,
                                vertex_t* h_dsts,
                                edge_t* h_edge_ids,
                                edge_type_t* h_edge_types,
                                size_t num_vertices,
                                size_t num_edges,
                                bool_t store_transposed,
                                edge_t* edge_ids_to_lookup,
                                edge_type_t* edge_types_to_lookup,
                                size_t num_edge_ids_to_lookup,
                                vertex_t* h_expected_srcs,
                                vertex_t* h_expected_dsts)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                       = NULL;
  cugraph_lookup_container_t* lookup_container = NULL;
  cugraph_lookup_result_t* result              = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = create_mg_test_graph_new(handle,
                                      vertex_tid,
                                      edge_tid,
                                      h_srcs,
                                      h_dsts,
                                      weight_tid,
                                      NULL,
                                      edge_type_tid,
                                      h_edge_types,
                                      edge_id_tid,
                                      h_edge_ids,
                                      num_edges,
                                      FALSE, /*store_transposed*/
                                      TRUE,  /*renumber*/
                                      TRUE,  /*is_symmetric*/
                                      FALSE, /*is_multigraph*/
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_build_edge_id_and_type_to_src_dst_lookup_map(
    handle, graph, &lookup_container, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_ecg failed.");

  cugraph_type_erased_device_array_t* d_edge_ids_to_lookup           = NULL;
  cugraph_type_erased_device_array_view_t* d_edge_ids_to_lookup_view = NULL;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_edge_ids_to_lookup, edge_id_tid, &d_edge_ids_to_lookup, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_edge_ids_to_lookup create failed.");

  d_edge_ids_to_lookup_view = cugraph_type_erased_device_array_view(d_edge_ids_to_lookup);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_edge_ids_to_lookup_view, (byte_t*)edge_ids_to_lookup, &ret_error);

  cugraph_type_erased_device_array_t* d_edge_types_to_lookup           = NULL;
  cugraph_type_erased_device_array_view_t* d_edge_types_to_lookup_view = NULL;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_edge_ids_to_lookup, edge_type_tid, &d_edge_types_to_lookup, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_edge_types_to_lookup create failed.");

  d_edge_types_to_lookup_view = cugraph_type_erased_device_array_view(d_edge_types_to_lookup);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_edge_types_to_lookup_view, (byte_t*)edge_types_to_lookup, &ret_error);

  ret_code = cugraph_lookup_endpoints_from_edge_ids_and_types(handle,
                                                              graph,
                                                              lookup_container,
                                                              d_edge_ids_to_lookup_view,
                                                              d_edge_types_to_lookup_view,
                                                              &result,
                                                              &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_ecg failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* d_srcs;
    cugraph_type_erased_device_array_view_t* d_dsts;

    d_srcs = cugraph_lookup_result_get_srcs(result);
    d_dsts = cugraph_lookup_result_get_dsts(result);

    vertex_t h_result_srcs[num_edge_ids_to_lookup];
    edge_t h_result_dsts[num_edge_ids_to_lookup];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_srcs, d_srcs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_dsts, d_dsts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    size_t result_num_edges = cugraph_type_erased_device_array_view_size(d_srcs);

    TEST_ALWAYS_ASSERT(result_num_edges == num_edge_ids_to_lookup,
                       "number of edges in returned result")

    for (int i = 0; i < num_edge_ids_to_lookup; i++) {
      vertex_t src = (h_result_srcs[i] < h_result_dsts[i]) ? h_result_srcs[i] : h_result_dsts[i];
      vertex_t dst = (h_result_srcs[i] >= h_result_dsts[i]) ? h_result_srcs[i] : h_result_dsts[i];
      TEST_ASSERT(test_ret_value,
                  src == h_expected_srcs[i],
                  "expected sources don't match with returned ones");
      TEST_ASSERT(test_ret_value,
                  dst == h_expected_dsts[i],
                  "expected destinations don't match with returned ones");
    }
  }

  cugraph_lookup_result_free(result);

  ret_code = cugraph_lookup_endpoints_from_edge_ids_and_single_type(handle,
                                                                    graph,
                                                                    lookup_container,
                                                                    d_edge_ids_to_lookup_view,
                                                                    edge_types_to_lookup[0],
                                                                    &result,
                                                                    &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_ecg failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* d_srcs;
    cugraph_type_erased_device_array_view_t* d_dsts;

    d_srcs = cugraph_lookup_result_get_srcs(result);
    d_dsts = cugraph_lookup_result_get_dsts(result);

    vertex_t h_result_srcs[num_edge_ids_to_lookup];
    edge_t h_result_dsts[num_edge_ids_to_lookup];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_srcs, d_srcs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_result_dsts, d_dsts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    size_t result_num_edges = cugraph_type_erased_device_array_view_size(d_srcs);

    TEST_ALWAYS_ASSERT(result_num_edges == num_edge_ids_to_lookup,
                       "number of edges in returned result")

    for (int i = 0; i < num_edge_ids_to_lookup; i++) {
      vertex_t src = (h_result_srcs[i] < h_result_dsts[i]) ? h_result_srcs[i] : h_result_dsts[i];
      vertex_t dst = (h_result_srcs[i] >= h_result_dsts[i]) ? h_result_srcs[i] : h_result_dsts[i];
      TEST_ASSERT(test_ret_value,
                  src == h_expected_srcs[i],
                  "expected sources don't match with returned ones");
      TEST_ASSERT(test_ret_value,
                  dst == h_expected_dsts[i],
                  "expected destinations don't match with returned ones");
    }
  }

  cugraph_lookup_result_free(result);

  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_lookup_src_dst_test(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 10;
  size_t num_vertices = 5;

  vertex_t h_srcs[] = {7, 1, 1, 2, 1, 8, 3, 4, 0, 0};
  vertex_t h_dsts[] = {8, 3, 4, 0, 0, 7, 1, 1, 2, 1};

  edge_t h_edge_ids[] = {78, 13, 14, 20, 10, 78, 13, 14, 20, 10};

  edge_type_t h_edge_types[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2};

  edge_t edge_ids_to_lookup[]   = {10, 12, 78, 20};
  edge_t edge_types_to_lookup[] = {2, 0, 3, 2};

  // expected results
  vertex_t h_expected_srcs[]    = {0, -1, -1, 0};
  vertex_t h_expected_dsts[]    = {1, -1, -1, 2};
  size_t num_edge_ids_to_lookup = 4;

  return generic_lookup_src_dst_test(handle,
                                     h_srcs,
                                     h_dsts,
                                     h_edge_ids,
                                     h_edge_types,
                                     num_vertices,
                                     num_edges,
                                     FALSE,
                                     edge_ids_to_lookup,
                                     edge_types_to_lookup,
                                     num_edge_ids_to_lookup,
                                     h_expected_srcs,
                                     h_expected_dsts);

  return 0;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_lookup_src_dst_test, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
