/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <stdio.h>

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int test_create_sg_graph_simple()
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

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

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

  ret_code = cugraph_graph_create_sg(handle,
                                     &properties,
                                     NULL,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_graph_free(graph);

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_sg_graph_csr()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  /*
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  */
  edge_t h_offsets[]   = {0, 1, 3, 6, 7, 8, 8};
  vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
  weight_t h_wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  bool_t with_replacement = FALSE;
  bool_t return_hops = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources = FALSE;
  bool_t renumber_results = FALSE;
  cugraph_compression_type_t compression = COO;
  bool_t compress_per_hop = FALSE;

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* offsets;
  cugraph_type_erased_device_array_t* indices;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* offsets_view;
  cugraph_type_erased_device_array_view_t* indices_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_vertices + 1, vertex_tid, &offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "offsets create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &indices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "indices create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  offsets_view = cugraph_type_erased_device_array_view(offsets);
  indices_view = cugraph_type_erased_device_array_view(indices);
  wgt_view     = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, offsets_view, (byte_t*)h_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "offsets copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, indices_view, (byte_t*)h_indices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "indices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_sg_graph_create_from_csr(handle,
                                              &properties,
                                              offsets_view,
                                              indices_view,
                                              wgt_view,
                                              NULL,
                                              NULL,
                                              FALSE,
                                              FALSE,
                                              FALSE,
                                              &graph,
                                              &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_vertices; ++i)
    for (size_t j = h_offsets[i]; j < h_offsets[i + 1]; ++j) {
      M[i][h_indices[j]] = h_wgt[j];
    }

  int fan_out[] = {-1};

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;
  cugraph_sample_result_t* result                       = NULL;

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, 1, INT32);

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);
  ret_code     = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t *sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
  cugraph_sampling_set_renumber_results(sampling_options, renumber_results);
  cugraph_sampling_set_compression_type(sampling_options, compression);
  cugraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);

  ret_code = cugraph_uniform_neighbor_sample(
                                              handle, graph, d_start_view, NULL, NULL, NULL, h_fan_out_view, rng_state, sampling_options, FALSE, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* srcs;
  cugraph_type_erased_device_array_view_t* dsts;
  cugraph_type_erased_device_array_view_t* wgts;

  srcs = cugraph_sample_result_get_sources(result);
  dsts = cugraph_sample_result_get_destinations(result);
  wgts = cugraph_sample_result_get_edge_weight(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(srcs);

  vertex_t h_result_srcs[result_size];
  vertex_t h_result_dsts[result_size];
  weight_t h_result_wgts[result_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_srcs, srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_dsts, dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_wgts, wgts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  TEST_ASSERT(test_ret_value, result_size == num_edges, "number of edges does not match");

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_result_srcs[i]][h_result_dsts[i]] == h_result_wgts[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
  cugraph_graph_free(graph);
  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(indices_view);
  cugraph_type_erased_device_array_view_free(offsets_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(indices);
  cugraph_type_erased_device_array_free(offsets);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);
  cugraph_sampling_options_free(sampling_options);

  return test_ret_value;
}

int test_create_sg_graph_symmetric_error()
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

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = TRUE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

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

  ret_code = cugraph_graph_create_sg(handle,
                                     &properties,
                                     NULL,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     TRUE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code != CUGRAPH_SUCCESS, "graph creation succeeded but should have failed.");

  if (ret_code == CUGRAPH_SUCCESS) cugraph_graph_free(graph);

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_sg_graph_with_isolated_vertices()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 8;
  size_t num_vertices = 7;
  double alpha = 0.95;
  double epsilon = 0.0001;
  size_t max_iterations = 20;

  vertex_t h_vertices[] = { 0, 1, 2, 3, 4, 5, 6 };
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = { 0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337 };

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* vertices;
  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* vertices_view;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_vertices, vertex_tid, &vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  vertices_view = cugraph_type_erased_device_array_view(vertices);
  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, vertices_view, (byte_t*)h_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_graph_create_sg(handle,
                                     &properties,
                                     vertices_view,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     FALSE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_centrality_result_t* result = NULL;

  // To verify we will call pagerank
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
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  result_vertices  = cugraph_centrality_result_get_vertices(result);
  pageranks = cugraph_centrality_result_get_values(result);

  vertex_t h_result_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_result_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(result);
  cugraph_graph_free(graph);

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_view_free(vertices_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);
  cugraph_type_erased_device_array_free(vertices);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_sg_graph_csr_with_isolated()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 8;
  size_t num_vertices = 7;
  double alpha = 0.95;
  double epsilon = 0.0001;
  size_t max_iterations = 20;

  /*
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  */
  edge_t h_offsets[]   = {0, 1, 3, 6, 7, 8, 8, 8};
  vertex_t h_indices[] = {1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_start[]   = {0, 1, 2, 3, 4, 5};
  weight_t h_wgt[]     = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = { 0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337 };

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* offsets;
  cugraph_type_erased_device_array_t* indices;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* offsets_view;
  cugraph_type_erased_device_array_view_t* indices_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_vertices + 1, vertex_tid, &offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "offsets create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &indices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "indices create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  offsets_view = cugraph_type_erased_device_array_view(offsets);
  indices_view = cugraph_type_erased_device_array_view(indices);
  wgt_view     = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, offsets_view, (byte_t*)h_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "offsets copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, indices_view, (byte_t*)h_indices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "indices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_sg_graph_create_from_csr(handle,
                                              &properties,
                                              offsets_view,
                                              indices_view,
                                              wgt_view,
                                              NULL,
                                              NULL,
                                              FALSE,
                                              FALSE,
                                              FALSE,
                                              &graph,
                                              &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_centrality_result_t* result = NULL;

  // To verify we will call pagerank
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
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  result_vertices  = cugraph_centrality_result_get_vertices(result);
  pageranks = cugraph_centrality_result_get_values(result);

  vertex_t h_result_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_result_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(result);
  cugraph_graph_free(graph);
  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(indices_view);
  cugraph_type_erased_device_array_view_free(offsets_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(indices);
  cugraph_type_erased_device_array_free(offsets);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_create_sg_graph_with_isolated_vertices_multi_input()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;
  size_t num_edges    = 11;
  size_t num_vertices = 7;
  double alpha = 0.95;
  double epsilon = 0.0001;
  size_t max_iterations = 20;

  vertex_t h_vertices[] = { 0, 1, 2, 3, 4, 5, 6 };
  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 5, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 3.2f, 3.2f, 1.7f};
  weight_t h_result[] = { 0.0859168, 0.158029, 0.0616337, 0.179675, 0.113239, 0.339873, 0.0616337 };

  cugraph_resource_handle_t* handle = NULL;
  cugraph_graph_t* graph            = NULL;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = FALSE;
  properties.is_multigraph = FALSE;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* vertices;
  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* vertices_view;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_vertices, vertex_tid, &vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, vertex_tid, &dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_edges, weight_tid, &wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  vertices_view = cugraph_type_erased_device_array_view(vertices);
  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, vertices_view, (byte_t*)h_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, src_view, (byte_t*)h_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, dst_view, (byte_t*)h_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, wgt_view, (byte_t*)h_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_graph_create_sg(handle,
                                     &properties,
                                     vertices_view,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     FALSE,
                                     FALSE,
                                     TRUE,
                                     TRUE,
                                     FALSE,
                                     &graph,
                                     &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_centrality_result_t* result = NULL;

  // To verify we will call pagerank
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
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_view_t* result_vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  result_vertices  = cugraph_centrality_result_get_vertices(result);
  pageranks = cugraph_centrality_result_get_values(result);

  vertex_t h_result_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_vertices, result_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_result_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(result);
  cugraph_graph_free(graph);

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_view_free(vertices_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);
  cugraph_type_erased_device_array_free(vertices);

  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_create_sg_graph_simple);
  result |= RUN_TEST(test_create_sg_graph_csr);
  result |= RUN_TEST(test_create_sg_graph_symmetric_error);
  result |= RUN_TEST(test_create_sg_graph_with_isolated_vertices);
  result |= RUN_TEST(test_create_sg_graph_csr_with_isolated);
  result |= RUN_TEST(test_create_sg_graph_with_isolated_vertices_multi_input);
  return result;
}
