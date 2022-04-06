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

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_pagerank_test(const cugraph_resource_handle_t* handle,
                          vertex_t* h_src,
                          vertex_t* h_dst,
                          weight_t* h_wgt,
                          weight_t* h_result,
                          size_t num_vertices,
                          size_t num_edges,
                          bool_t store_transposed,
                          double alpha,
                          double epsilon,
                          size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* p_graph            = NULL;
  cugraph_centrality_result_t* p_result = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  ret_code = cugraph_pagerank(
    handle, p_graph, NULL, alpha, epsilon, max_iterations, FALSE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_pagerank failed.");

  // NOTE: Because we get back vertex ids and pageranks, we can simply compare
  //       the returned values with the expected results for the entire
  //       graph.  Each GPU will have a subset of the total vertices, so
  //       they will do a subset of the comparisons.
  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  vertices  = cugraph_centrality_result_get_vertices(p_result);
  pageranks = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(vertices);

  for (int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(p_result);
  cugraph_mg_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_pagerank(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // Pagerank wants store_transposed = TRUE
  return generic_pagerank_test(handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               h_result,
                               num_vertices,
                               num_edges,
                               TRUE,
                               alpha,
                               epsilon,
                               max_iterations);
}

int test_pagerank_with_transpose(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // Pagerank wants store_transposed = TRUE
  //    This call will force cugraph_pagerank to transpose the graph
  //    But we're passing src/dst backwards so the results will be the same
  return generic_pagerank_test(handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               h_result,
                               num_vertices,
                               num_edges,
                               FALSE,
                               alpha,
                               epsilon,
                               max_iterations);
}

int test_pagerank_4(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {
    0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 500;

  return generic_pagerank_test(handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               h_result,
                               num_vertices,
                               num_edges,
                               FALSE,
                               alpha,
                               epsilon,
                               max_iterations);
}

int test_pagerank_4_with_transpose(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {
    0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 500;

  return generic_pagerank_test(handle,
                               h_src,
                               h_dst,
                               h_wgt,
                               h_result,
                               num_vertices,
                               num_edges,
                               TRUE,
                               alpha,
                               epsilon,
                               max_iterations);
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

#if 0
  // TODO:  Need something a bit more sophisticated for bigger systems
  prows = (int)sqrt((double)comm_size);
  while (comm_size % prows != 0) {
    --prows;
  }

  ret_code = cugraph_resource_handle_init_comms(handle, prows, &ret_error);
  TEST_ASSERT(result, ret_code == CUGRAPH_SUCCESS, "handle create failed.");
  TEST_ASSERT(result, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
#endif

  void* raft_handle = create_raft_handle(prows);
  handle            = cugraph_create_resource_handle(raft_handle);

  if (result == 0) {
    result |= RUN_MG_TEST(test_pagerank, handle);
    result |= RUN_MG_TEST(test_pagerank_with_transpose, handle);
    result |= RUN_MG_TEST(test_pagerank_4, handle);
    result |= RUN_MG_TEST(test_pagerank_4_with_transpose, handle);

    cugraph_free_resource_handle(handle);
  }

  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return result;
}
