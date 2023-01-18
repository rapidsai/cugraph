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

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <unistd.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_uniform_neighbor_sample_test(const cugraph_resource_handle_t* handle,
                                         vertex_t* h_src,
                                         vertex_t* h_dst,
                                         edge_t* h_idx,
                                         size_t num_vertices,
                                         size_t num_edges,
                                         vertex_t* h_start,
                                         size_t num_starts,
                                         int* fan_out,
                                         size_t max_depth,
                                         bool_t with_replacement,
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

  ret_code = create_mg_test_graph_with_edge_ids(
    handle, h_src, h_dst, h_idx, num_edges, store_transposed, FALSE, &graph, &ret_error);
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
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed");
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
  int h_labels[result_size];
  edge_t h_index[result_size];
  size_t* h_counts;

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
    M[h_src[i]][h_dst[i]] = h_idx[i];

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_srcs[i]][h_dsts[i]] >= 0,
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_neighbor_sample(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t idx[]     = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[] = {2, 2};
  int fan_out[]    = {1, 2};

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              idx,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              num_starts,
                                              fan_out,
                                              fan_out_size,
                                              TRUE,
                                              FALSE);
}

int test_uniform_neighbor_from_alex(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 12;
  size_t num_vertices = 5;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]   = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
  vertex_t dst[]   = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
  edge_t idx[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t typ[]    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
  weight_t wgt[]   = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
  vertex_t start[] = {0, 4};
  int32_t batch[]  = {0, 1};
  int fan_out[]    = {2, 2};

  bool_t with_replacement = TRUE;
  bool_t store_transposed = FALSE;

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_t* d_label           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_device_array_view_t* d_label_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = create_mg_test_graph_with_properties(
    handle, src, dst, idx, typ, wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_label, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label create failed.");

  d_label_view = cugraph_type_erased_device_array_view(d_label);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_view, (byte_t*)batch, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  d_label_view,
                                                                  h_fan_out_view,
                                                                  rng_state,
                                                                  with_replacement,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed");
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_src;
  cugraph_type_erased_device_array_view_t* result_dst;
  cugraph_type_erased_device_array_view_t* result_index;
  cugraph_type_erased_device_array_view_t* result_type;
  cugraph_type_erased_device_array_view_t* result_weight;
  cugraph_type_erased_device_array_view_t* result_labels;
  cugraph_type_erased_device_array_view_t* result_hops;

  result_src    = cugraph_sample_result_get_sources(result);
  result_dst    = cugraph_sample_result_get_destinations(result);
  result_index  = cugraph_sample_result_get_edge_id(result);
  result_type   = cugraph_sample_result_get_edge_type(result);
  result_weight = cugraph_sample_result_get_edge_weight(result);
  result_labels = cugraph_sample_result_get_start_labels(result);
  result_hops   = cugraph_sample_result_get_hop(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_src);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_index[result_size];
  int h_type[result_size];
  weight_t h_wgt[result_size];
  int h_labels[result_size];
  int h_hop[result_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_srcs, result_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_dsts, result_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_index, result_index, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_type, result_type, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_wgt, result_weight, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_labels, result_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_hop, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  edge_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[src[i]][dst[i]] = idx[i];

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_srcs[i]][h_dsts[i]] >= 0,
                "uniform_neighbor_sample got edge that doesn't exist");
  }
#endif

  cugraph_sample_result_free(result);

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  // Set up MPI:
  int comm_rank;
  int comm_size;
  int num_gpus_per_node;
  cudaError_t status;
  int mpi_status;
  int result                        = 0;
  cugraph_resource_handle_t* handle = NULL;
  cugraph_error_t* ret_error;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  int prows                     = 1;

  C_MPI_TRY(MPI_Init(&argc, &argv));
  C_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));
  C_MPI_TRY(MPI_Comm_size(MPI_COMM_WORLD, &comm_size));
  C_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  C_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));

  void* raft_handle = create_raft_handle(prows);
  handle            = cugraph_create_resource_handle(raft_handle);

  if (result == 0) {
    result |= RUN_MG_TEST(test_uniform_neighbor_sample, handle);
    result |= RUN_MG_TEST(test_uniform_neighbor_from_alex, handle);

    cugraph_free_resource_handle(handle);
  }

  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return result;
}
