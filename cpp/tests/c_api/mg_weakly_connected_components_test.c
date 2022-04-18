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

int generic_wcc_test(const cugraph_resource_handle_t* handle,
                     vertex_t* h_src,
                     vertex_t* h_dst,
                     weight_t* h_wgt,
                     vertex_t* h_result,
                     size_t num_vertices,
                     size_t num_edges,
                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* p_graph            = NULL;
  cugraph_labeling_result_t* p_result = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, TRUE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  ret_code = cugraph_weakly_connected_components(handle, p_graph, FALSE, &p_result, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_weakly_connected_components failed.");

  // NOTE: Because we get back vertex ids and components, we can simply compare
  //       the returned values with the expected results for the entire
  //       graph.  Each GPU will have a subset of the total vertices, so
  //       they will do a subset of the comparisons.
  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* components;

  vertices   = cugraph_labeling_result_get_vertices(p_result);
  components = cugraph_labeling_result_get_labels(p_result);

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(vertices);

  vertex_t h_vertices[num_local_vertices];
  vertex_t h_components[num_local_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_components, components, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  vertex_t component_check[num_vertices];
  for (vertex_t i = 0; i < num_vertices; ++i) {
    component_check[i] = num_vertices;
  }

  vertex_t num_errors = 0;
  for (vertex_t i = 0; i < num_local_vertices; ++i) {
    if (component_check[h_components[i]] == num_vertices) {
      component_check[h_components[i]] = h_result[h_vertices[i]];
    } else if (component_check[h_components[i]] != h_result[h_vertices[i]]) {
      ++num_errors;
    }
  }

  TEST_ASSERT(test_ret_value, num_errors == 0, "weakly connected components results don't match");

  cugraph_type_erased_device_array_view_free(components);
  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_labeling_result_free(p_result);
  cugraph_mg_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_weakly_connected_components(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 32;
  size_t num_vertices = 12;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10,
                      1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 7, 9, 10, 6, 7, 9, 11, 11,
                      0, 1, 1, 2, 2, 2, 3, 4, 6, 7, 7,  8, 8, 8, 9,  10};
  weight_t h_wgt[]    = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  vertex_t h_result[] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};

  // WCC wants store_transposed = FALSE
  return generic_wcc_test(handle, h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE);
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
    result |= RUN_MG_TEST(test_weakly_connected_components, handle);

    cugraph_free_resource_handle(handle);
  }

  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return result;
}
