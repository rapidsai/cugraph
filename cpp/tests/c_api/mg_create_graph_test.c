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

#include "c_test_utils.h"  /* RUN_TEST */
#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int test_create_mg_graph_simple(const cugraph_resource_handle_t* handle)
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  cugraph_graph_t* graph = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  int my_rank = cugraph_resource_handle_get_rank(handle);

  for (int i = 0; i < num_edges; ++i) {
    h_src[i] += 10 * my_rank;
    h_dst[i] += 10 * my_rank;
  }

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_graph_create_mg(handle,
                                     &properties,
                                     NULL,
                                     (cugraph_type_erased_device_array_view_t const* const*) &src_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) &dst_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) &wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     1,
                                     FALSE,
                                     FALSE,
                                     TRUE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_graph_free(graph);

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_mg_graph_multiple_edge_lists(const cugraph_resource_handle_t* handle)
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 8;
  size_t num_vertices = 7;

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  vertex_t h_vertices[] = { 0, 1, 2, 3, 4, 5, 6 };
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = { 0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337 };

  cugraph_graph_t* graph = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  const size_t num_local_arrays = 2;

  cugraph_type_erased_device_array_t* vertices[num_local_arrays];
  cugraph_type_erased_device_array_t* src[num_local_arrays];
  cugraph_type_erased_device_array_t* dst[num_local_arrays];
  cugraph_type_erased_device_array_t* wgt[num_local_arrays];
  cugraph_type_erased_device_array_view_t* vertices_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* src_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* dst_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* wgt_view[num_local_arrays];

  int my_rank = cugraph_resource_handle_get_rank(handle);
  int comm_size = cugraph_resource_handle_get_comm_size(handle);

  size_t local_num_vertices = num_vertices / comm_size;
  size_t local_start_vertex = my_rank * local_num_vertices;
  size_t local_num_edges = num_edges / comm_size;
  size_t local_start_edge = my_rank * local_num_edges;

  local_num_edges = (my_rank != (comm_size - 1)) ? local_num_edges : (num_edges - local_start_edge);
  local_num_vertices = (my_rank != (comm_size - 1)) ? local_num_vertices : (num_vertices - local_start_vertex);

  for (size_t i = 0 ; i < num_local_arrays ; ++i) {
    size_t vertex_count = local_num_vertices / num_local_arrays;
    size_t vertex_start = i * vertex_count;
    vertex_count = (i != (num_local_arrays - 1)) ? vertex_count : (local_num_vertices - vertex_start);

    ret_code =
      cugraph_type_erased_device_array_create(handle, vertex_count, vertex_tid, vertices + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    size_t edge_count = (local_num_edges + num_local_arrays - 1) / num_local_arrays;
    size_t edge_start = i * edge_count;
    edge_count = (edge_count < (local_num_edges - edge_start)) ? edge_count : (local_num_edges - edge_start);

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, vertex_tid, src + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, vertex_tid, dst + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, weight_tid, wgt + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

    vertices_view[i] = cugraph_type_erased_device_array_view(vertices[i]);
    src_view[i] = cugraph_type_erased_device_array_view(src[i]);
    dst_view[i] = cugraph_type_erased_device_array_view(dst[i]);
    wgt_view[i] = cugraph_type_erased_device_array_view(wgt[i]);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, vertices_view[i], (byte_t*)(h_vertices + local_start_vertex + vertex_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, src_view[i], (byte_t*)(h_src + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, dst_view[i], (byte_t*)(h_dst + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, wgt_view[i], (byte_t*)(h_wgt + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");
  }

  ret_code = cugraph_graph_create_mg(handle,
                                     &properties,
                                     (cugraph_type_erased_device_array_view_t const* const*) vertices_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) src_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) dst_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     num_local_arrays,
                                     FALSE,
                                     FALSE,
                                     TRUE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  //
  //  Now call pagerank and check results...
  //
  cugraph_centrality_result_t* result = NULL;

  ret_code = cugraph_pagerank(handle,
                              graph,
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              alpha,
                              epsilon,
                              max_iterations,
                              FALSE,
                              &result,
                              &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_pagerank failed.");

  // NOTE: Because we get back vertex ids and pageranks, we can simply compare
  //       the returned values with the expected results for the entire
  //       graph.  Each GPU will have a subset of the total vertices, so
  //       they will do a subset of the comparisons.
  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  result_vertices  = cugraph_centrality_result_get_vertices(result);
  pageranks = cugraph_centrality_result_get_values(result);

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(result_vertices);

  vertex_t h_result_vertices[num_local_vertices];
  weight_t h_pageranks[num_local_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_result_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(result);
  cugraph_graph_free(graph);

  for (size_t i = 0 ; i < num_local_arrays ; ++i) {
    cugraph_type_erased_device_array_view_free(wgt_view[i]);
    cugraph_type_erased_device_array_view_free(dst_view[i]);
    cugraph_type_erased_device_array_view_free(src_view[i]);
    cugraph_type_erased_device_array_view_free(vertices_view[i]);
    cugraph_type_erased_device_array_free(wgt[i]);
    cugraph_type_erased_device_array_free(dst[i]);
    cugraph_type_erased_device_array_free(src[i]);
    cugraph_type_erased_device_array_free(vertices[i]);
  }

  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_mg_graph_multiple_edge_lists_multi_edge(const cugraph_resource_handle_t* handle)
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 11;
  size_t num_vertices = 7;

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  vertex_t h_vertices[] = { 0, 1, 2, 3, 4, 5, 6 };
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.1f};
  weight_t h_result[] = { 0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337 };

  cugraph_graph_t* graph = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  const size_t num_local_arrays = 2;

  cugraph_type_erased_device_array_t* vertices[num_local_arrays];
  cugraph_type_erased_device_array_t* src[num_local_arrays];
  cugraph_type_erased_device_array_t* dst[num_local_arrays];
  cugraph_type_erased_device_array_t* wgt[num_local_arrays];
  cugraph_type_erased_device_array_view_t* vertices_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* src_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* dst_view[num_local_arrays];
  cugraph_type_erased_device_array_view_t* wgt_view[num_local_arrays];

  int my_rank = cugraph_resource_handle_get_rank(handle);
  int comm_size = cugraph_resource_handle_get_comm_size(handle);

  size_t local_num_vertices = num_vertices / comm_size;
  size_t local_start_vertex = my_rank * local_num_vertices;
  size_t local_num_edges = num_edges / comm_size;
  size_t local_start_edge = my_rank * local_num_edges;

  local_num_edges = (my_rank != (comm_size - 1)) ? local_num_edges : (num_edges - local_start_edge);
  local_num_vertices = (my_rank != (comm_size - 1)) ? local_num_vertices : (num_vertices - local_start_vertex);

  for (size_t i = 0 ; i < num_local_arrays ; ++i) {
    size_t vertex_count = (local_num_vertices + num_local_arrays - 1) / num_local_arrays;
    size_t vertex_start = i * vertex_count;
    vertex_count = (i != (num_local_arrays - 1)) ? vertex_count : (local_num_vertices - vertex_start);

    ret_code =
      cugraph_type_erased_device_array_create(handle, vertex_count, vertex_tid, vertices + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices create failed.");
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

    size_t edge_count = (local_num_edges + num_local_arrays - 1) / num_local_arrays;
    size_t edge_start = i * edge_count;
    edge_count = (edge_count < (local_num_edges - edge_start)) ? edge_count : (local_num_edges - edge_start);

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, vertex_tid, src + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, vertex_tid, dst + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, edge_count, weight_tid, wgt + i, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

    vertices_view[i] = cugraph_type_erased_device_array_view(vertices[i]);
    src_view[i] = cugraph_type_erased_device_array_view(src[i]);
    dst_view[i] = cugraph_type_erased_device_array_view(dst[i]);
    wgt_view[i] = cugraph_type_erased_device_array_view(wgt[i]);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, vertices_view[i], (byte_t*)(h_vertices + local_start_vertex + vertex_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, src_view[i], (byte_t*)(h_src + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, dst_view[i], (byte_t*)(h_dst + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, wgt_view[i], (byte_t*)(h_wgt + local_start_edge + edge_start), &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");
  }

  ret_code = cugraph_graph_create_mg(handle,
                                     &properties,
                                     (cugraph_type_erased_device_array_view_t const* const*) vertices_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) src_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) dst_view,
                                     (cugraph_type_erased_device_array_view_t const* const*) wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     num_local_arrays,
                                     TRUE,
                                     TRUE,
                                     TRUE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  //
  //  Now call pagerank and check results...
  //
  cugraph_centrality_result_t* result = NULL;

  ret_code = cugraph_pagerank(handle,
                              graph,
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              alpha,
                              epsilon,
                              max_iterations,
                              FALSE,
                              &result,
                              &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_pagerank failed.");

  // NOTE: Because we get back vertex ids and pageranks, we can simply compare
  //       the returned values with the expected results for the entire
  //       graph.  Each GPU will have a subset of the total vertices, so
  //       they will do a subset of the comparisons.
  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  result_vertices  = cugraph_centrality_result_get_vertices(result);
  pageranks = cugraph_centrality_result_get_values(result);

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(result_vertices);

  vertex_t h_result_vertices[num_local_vertices];
  weight_t h_pageranks[num_local_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_result_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(result);
  cugraph_graph_free(graph);

  for (size_t i = 0 ; i < num_local_arrays ; ++i) {
    cugraph_type_erased_device_array_view_free(wgt_view[i]);
    cugraph_type_erased_device_array_view_free(dst_view[i]);
    cugraph_type_erased_device_array_view_free(src_view[i]);
    cugraph_type_erased_device_array_view_free(vertices_view[i]);
    cugraph_type_erased_device_array_free(wgt[i]);
    cugraph_type_erased_device_array_free(dst[i]);
    cugraph_type_erased_device_array_free(src[i]);
    cugraph_type_erased_device_array_free(vertices[i]);
  }

  cugraph_error_free(ret_error);

  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_create_mg_graph_simple, handle);
  result |= RUN_MG_TEST(test_create_mg_graph_multiple_edge_lists, handle);
  result |= RUN_MG_TEST(test_create_mg_graph_multiple_edge_lists_multi_edge, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
