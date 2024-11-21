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

#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>

#include <stdio.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

cugraph_data_type_id_t vertex_tid    = INT32;
cugraph_data_type_id_t edge_tid      = INT32;
cugraph_data_type_id_t weight_tid    = FLOAT32;
cugraph_data_type_id_t edge_id_tid   = INT32;
cugraph_data_type_id_t edge_type_tid = INT32;

/*
 * Create graph and count multi-edges
 */
int generic_count_multi_edges_test(const cugraph_resource_handle_t* handle,
                                   vertex_t* h_src,
                                   vertex_t* h_dst,
                                   weight_t* h_wgt,
                                   size_t num_vertices,
                                   size_t num_edges,
                                   bool_t store_transposed,
                                   bool_t is_symmetric,
                                   bool_t is_multigraph,
                                   size_t multi_edges_count)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph = NULL;
  size_t result          = 0;

  ret_code = create_mg_test_graph_new(handle,
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
                                      num_edges,
                                      store_transposed,
                                      FALSE,
                                      is_symmetric,
                                      is_multigraph,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_count_multi_edges(handle, graph, FALSE, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_count_multi_edges failed.");

  TEST_ASSERT(test_ret_value, result == multi_edges_count, "multi-edge count did not match");

  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_multi_edges_count(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 14;
  size_t num_vertices = 6;

  vertex_t h_src[]        = {0, 1, 1, 2, 2, 2, 3, 4, 0, 1, 1, 3, 0, 1};
  vertex_t h_dst[]        = {1, 3, 4, 0, 1, 3, 5, 5, 1, 3, 0, 1, 1, 0};
  weight_t h_wgt[]        = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  size_t multi_edge_count = 4;

  return generic_count_multi_edges_test(
    handle, h_src, h_dst, h_wgt, num_vertices, num_edges, TRUE, TRUE, TRUE, multi_edge_count);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_multi_edges_count, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
