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
typedef float weight_t;

int create_test_graph_with_edge_ids(const cugraph_resource_handle_t* p_handle,
                                    vertex_t* h_src,
                                    vertex_t* h_dst,
                                    edge_t* h_ids,
                                    size_t num_edges,
                                    bool_t store_transposed,
                                    bool_t renumber,
                                    bool_t is_symmetric,
                                    cugraph_graph_t** p_graph,
                                    cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* ids;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* ids_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, edge_tid, &ids, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "ids create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  ids_view = cugraph_type_erased_device_array_view(ids);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, ids_view, (byte_t*)h_ids, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_as_type(ids, weight_tid, &wgt_view, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt cast from ids failed.");

  ret_code = cugraph_sg_graph_create(p_handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     store_transposed,
                                     renumber,
                                     FALSE,
                                     p_graph,
                                     ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(ids_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(ids);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

int generic_uniform_neighbor_sample_test(const cugraph_resource_handle_t* handle,
                                         vertex_t* h_src,
                                         vertex_t* h_dst,
                                         edge_t* h_ids,
                                         size_t num_vertices,
                                         size_t num_edges,
                                         vertex_t* h_start,
                                         size_t num_starts,
                                         int* fan_out,
                                         size_t max_depth,
                                         bool_t with_replacement,
                                         bool_t renumber,
                                         bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  ret_code = create_test_graph_with_edge_ids(
    handle, h_src, h_dst, h_ids, num_edges, store_transposed, renumber, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, max_depth, INT32);

  ret_code = cugraph_uniform_neighbor_sample(
    handle, graph, d_start_view, h_fan_out_view, with_replacement, FALSE, &result, &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed")
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* srcs;
  cugraph_type_erased_device_array_view_t* dsts;
  cugraph_type_erased_device_array_view_t* index;

  srcs  = cugraph_sample_result_get_sources(result);
  dsts  = cugraph_sample_result_get_destinations(result);
  index = cugraph_sample_result_get_index(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(srcs);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_index[result_size];

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_srcs, srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_dsts, dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_index, index, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  edge_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_ids[i];

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_srcs[i]][h_dsts[i]] == h_index[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_neighbor_sample(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[] = {1, 2, 3, 4, 5, 6, 7, 8};
  vertex_t start[]  = {2, 2};
  int fan_out[]     = {1, 2};

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              edge_ids,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              num_starts,
                                              fan_out,
                                              fan_out_size,
                                              TRUE,
                                              FALSE,
                                              FALSE);
}

int test_uniform_neighbor_sample_all_neighbors(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 1;
  size_t num_starts   = 2;

  vertex_t src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[] = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[]  = {2};
  int fan_out[]     = {-1};

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              edge_ids,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              num_starts,
                                              fan_out,
                                              fan_out_size,
                                              TRUE,
                                              FALSE,
                                              FALSE);
}

int test_uniform_neighbor_sample_with_properties(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 1;
  size_t num_starts   = 2;

  vertex_t src[]       = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  int32_t edge_types[] = {7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2};
  int fan_out[]        = {-1};

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  ret_code = create_sg_test_graph(handle,
                                  vertex_tid,
                                  edge_tid,
                                  src,
                                  dst,
                                  weight_tid,
                                  weight,
                                  edge_type_tid,
                                  edge_types,
                                  edge_id_tid,
                                  edge_ids,
                                  num_edges,
                                  FALSE,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  &graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, 1, INT32);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  NULL,
                                                                  h_fan_out_view,
                                                                  rng_state,
                                                                  FALSE,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed")
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_srcs;
  cugraph_type_erased_device_array_view_t* result_dsts;
  cugraph_type_erased_device_array_view_t* result_edge_id;
  cugraph_type_erased_device_array_view_t* result_weights;
  cugraph_type_erased_device_array_view_t* result_edge_types;
  cugraph_type_erased_device_array_view_t* result_hops;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_edge_types = cugraph_sample_result_get_edge_type(result);
  result_hops       = cugraph_sample_result_get_hop(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_edge_id[result_size];
  weight_t h_weight[result_size];
  int32_t h_edge_types[result_size];
  int32_t h_hops[result_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_srcs, result_srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_dsts, result_dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_edge_id, result_edge_id, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_weight, result_weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_edge_types, result_edge_types, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_hops, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M_w[num_vertices][num_vertices];
  edge_t M_edge_id[num_vertices][num_vertices];
  int32_t M_edge_type[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M_w[i][j]         = 0.0;
      M_edge_id[i][j]   = -1;
      M_edge_type[i][j] = -1;
    }

  for (int i = 0; i < num_edges; ++i) {
    M_w[src[i]][dst[i]]         = weight[i];
    M_edge_id[src[i]][dst[i]]   = edge_ids[i];
    M_edge_type[src[i]][dst[i]] = edge_types[i];
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M_w[h_srcs[i]][h_dsts[i]] == h_weight[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_type[h_srcs[i]][h_dsts[i]] == h_edge_types[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
}

int main(int argc, char** argv)
{
  cugraph_resource_handle_t* handle = NULL;

  handle = cugraph_create_resource_handle(NULL);

  int result = 0;
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_all_neighbors, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_with_properties, handle);

  cugraph_free_resource_handle(handle);

  return result;
}
