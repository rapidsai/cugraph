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

int generic_experimental_uniform_neighbor_sample_test(vertex_t* h_src,
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

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_sample_result_t* result   = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

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

  ret_code = cugraph_experimental_uniform_neighbor_sample(
    handle, graph, d_start_view, h_fan_out_view, with_replacement, FALSE, &result, &ret_error);

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
                M[h_srcs[i]][h_dsts[i]] > 0,
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_type_erased_host_array_view_free(h_fan_out_view);

  return test_ret_value;
}

int generic_uniform_neighbor_sample_test(vertex_t* h_src,
                                         vertex_t* h_dst,
                                         weight_t* h_wgt,
                                         size_t num_vertices,
                                         size_t num_edges,
                                         vertex_t* h_start,
                                         int* h_start_label,
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

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_sample_result_t* result   = NULL;

  cugraph_type_erased_device_array_t* d_start                 = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view       = NULL;
  cugraph_type_erased_device_array_t* d_start_label           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_label_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view       = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, renumber, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start_label, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_label create failed.");

  d_start_label_view = cugraph_type_erased_device_array_view(d_start_label);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_label_view, (byte_t*)h_start_label, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_label copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, max_depth, INT32);

  ret_code = cugraph_uniform_neighbor_sample(handle,
                                             graph,
                                             d_start_view,
                                             d_start_label_view,
                                             h_fan_out_view,
                                             with_replacement,
                                             FALSE,
                                             &result,
                                             &ret_error);

  TEST_ASSERT(test_ret_value,
              ret_code != CUGRAPH_SUCCESS,
              "cugraph_uniform_neighbor_sample expected to fail in SG test");

#if 0
  // FIXME:  cugraph_uniform_neighbor_sample does not support SG
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* srcs;
  cugraph_type_erased_device_array_view_t* dsts;
  cugraph_type_erased_device_array_view_t* labels;
  cugraph_type_erased_device_array_view_t* index;
  cugraph_type_erased_host_array_view_t* counts;

  srcs   = cugraph_sample_result_get_sources(result);
  dsts   = cugraph_sample_result_get_destinations(result);
  labels = cugraph_sample_result_get_start_labels(result);
  index  = cugraph_sample_result_get_index(result);
  counts = cugraph_sample_result_get_counts(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(srcs);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  int h_labels[result_size];
  edge_t h_index[result_size];
  size_t* h_counts;

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_srcs, srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_dsts, dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_labels, labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code =
    cugraph_type_erased_device_array_view_copy_to_host(handle, (byte_t*)h_index, index, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  h_counts = (size_t*)cugraph_type_erased_host_array_pointer(counts);

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = 0.0;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_wgt[i];

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_srcs[i]][h_dsts[i]] > 0.0,
                "uniform_neighbor_sample got edge that doesn't exist");

    bool_t found = FALSE;
    for (int j = 0; j < num_starts; ++j)
      found = found || (h_labels[i] == h_start_label[j]);

    TEST_ASSERT(test_ret_value, found, "invalid label");
  }

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
#endif

  return test_ret_value;
}

int test_uniform_neighbor_sample()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]          = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]          = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]          = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t start[]        = {2, 2};
  vertex_t start_labels[] = {0, 1};
  int fan_out[]           = {1, 2};

  return generic_uniform_neighbor_sample_test(src,
                                              dst,
                                              wgt,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_labels,
                                              num_starts,
                                              fan_out,
                                              fan_out_size,
                                              TRUE,
                                              FALSE,
                                              FALSE);
}

int test_experimental_uniform_neighbor_sample()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[] = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[]  = {2, 2};
  int fan_out[]     = {1, 2};

  return generic_experimental_uniform_neighbor_sample_test(src,
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

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_uniform_neighbor_sample);
  result |= RUN_TEST(test_experimental_uniform_neighbor_sample);
  return result;
}
