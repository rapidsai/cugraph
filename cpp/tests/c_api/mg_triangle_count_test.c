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

int generic_triangle_count_test(const cugraph_resource_handle_t* handle,
                                vertex_t* h_src,
                                vertex_t* h_dst,
                                weight_t* h_wgt,
                                vertex_t* h_verts,
                                edge_t* h_result,
                                size_t num_vertices,
                                size_t num_edges,
                                size_t num_results,
                                bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* p_graph                              = NULL;
  cugraph_triangle_count_result_t* p_result             = NULL;
  cugraph_type_erased_device_array_t* p_start           = NULL;
  cugraph_type_erased_device_array_view_t* p_start_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, TRUE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  if (h_verts != NULL) {
    if (rank == 0) {
      ret_code =
        cugraph_type_erased_device_array_create(handle, num_results, INT32, &p_start, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_start create failed.");

      p_start_view = cugraph_type_erased_device_array_view(p_start);

      ret_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, p_start_view, (byte_t*)h_verts, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");
    } else {
      p_start_view = cugraph_type_erased_device_array_view_create(NULL, 0, INT32);
    }
  }

  ret_code = cugraph_triangle_count(handle, p_graph, p_start_view, FALSE, &p_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_triangle_count failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* vertices;
    cugraph_type_erased_device_array_view_t* counts;

    vertices = cugraph_triangle_count_result_get_vertices(p_result);
    counts   = cugraph_triangle_count_result_get_counts(p_result);

    vertex_t num_local_results = cugraph_type_erased_device_array_view_size(vertices);

    vertex_t h_vertices[num_local_results];
    edge_t h_counts[num_local_results];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_vertices, vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_counts, counts, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    for (int i = 0; (i < num_local_results) && (test_ret_value == 0); ++i) {
      for (int j = 0 ; j < num_results ; ++j) {
        if (h_vertices[i] == h_verts[j]) {
          TEST_ASSERT(test_ret_value, h_counts[i] == h_result[j], "counts results don't match");
        }
      }
    }

    cugraph_triangle_count_result_free(p_result);
  }

  cugraph_mg_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_triangle_count(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_results  = 4;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_verts[] = {0, 1, 2, 4};
  edge_t h_result[]  = {1, 2, 2, 0};

  // Triangle Count wants store_transposed = FALSE
  return generic_triangle_count_test(
    handle, h_src, h_dst, h_wgt, h_verts, h_result, num_vertices, num_edges, num_results, FALSE);
}

int test_triangle_count_dolphins(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 318;
  size_t num_vertices = 62;

  vertex_t h_src[] = {
    10, 14, 15, 40, 42, 47, 17, 19, 26, 27, 28, 36, 41, 54, 10, 42, 44, 61, 8,  14, 59, 51, 9,
    13, 56, 57, 9,  13, 17, 54, 56, 57, 19, 27, 30, 40, 54, 20, 28, 37, 45, 59, 13, 17, 32, 41,
    57, 29, 42, 47, 51, 33, 17, 32, 41, 54, 57, 16, 24, 33, 34, 37, 38, 40, 43, 50, 52, 18, 24,
    40, 45, 55, 59, 20, 33, 37, 38, 50, 22, 25, 27, 31, 57, 20, 21, 24, 29, 45, 51, 30, 54, 28,
    36, 38, 44, 47, 50, 29, 33, 37, 45, 51, 36, 45, 51, 29, 45, 51, 26, 27, 27, 30, 47, 35, 43,
    45, 51, 52, 42, 47, 60, 34, 37, 38, 40, 43, 50, 37, 44, 49, 37, 39, 40, 59, 40, 43, 45, 61,
    43, 44, 52, 58, 57, 52, 54, 57, 47, 50, 46, 53, 50, 51, 59, 49, 57, 51, 55, 61, 57, 0,  0,
    0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  4,  5,  5,  5,
    5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  10,
    10, 10, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
    15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 20, 20, 20,
    20, 20, 20, 21, 21, 21, 21, 21, 23, 23, 23, 24, 24, 24, 25, 25, 26, 28, 28, 29, 29, 29, 29,
    29, 30, 30, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 36, 36, 36, 36, 37, 37, 37, 37, 38, 38,
    38, 38, 39, 40, 41, 41, 42, 42, 43, 43, 45, 45, 45, 46, 48, 50, 51, 53, 54};

  vertex_t h_dst[] = {
    0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  4,  5,
    5,  5,  5,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  9,  9,  9,  9,
    9,  10, 10, 10, 11, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15,
    15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 19, 20,
    20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 23, 23, 23, 24, 24, 24, 25, 25, 26, 28, 28, 29, 29,
    29, 29, 29, 30, 30, 32, 33, 33, 33, 33, 33, 33, 34, 34, 34, 36, 36, 36, 36, 37, 37, 37, 37,
    38, 38, 38, 38, 39, 40, 41, 41, 42, 42, 43, 43, 45, 45, 45, 46, 48, 50, 51, 53, 54, 10, 14,
    15, 40, 42, 47, 17, 19, 26, 27, 28, 36, 41, 54, 10, 42, 44, 61, 8,  14, 59, 51, 9,  13, 56,
    57, 9,  13, 17, 54, 56, 57, 19, 27, 30, 40, 54, 20, 28, 37, 45, 59, 13, 17, 32, 41, 57, 29,
    42, 47, 51, 33, 17, 32, 41, 54, 57, 16, 24, 33, 34, 37, 38, 40, 43, 50, 52, 18, 24, 40, 45,
    55, 59, 20, 33, 37, 38, 50, 22, 25, 27, 31, 57, 20, 21, 24, 29, 45, 51, 30, 54, 28, 36, 38,
    44, 47, 50, 29, 33, 37, 45, 51, 36, 45, 51, 29, 45, 51, 26, 27, 27, 30, 47, 35, 43, 45, 51,
    52, 42, 47, 60, 34, 37, 38, 40, 43, 50, 37, 44, 49, 37, 39, 40, 59, 40, 43, 45, 61, 43, 44,
    52, 58, 57, 52, 54, 57, 47, 50, 46, 53, 50, 51, 59, 49, 57, 51, 55, 61, 57};

  weight_t h_wgt[] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  vertex_t h_verts[] = {11, 48, 0};
  edge_t h_result[]  = {0, 0, 5};
  size_t num_results = 3;

  // Triangle Count wants store_transposed = FALSE
  return generic_triangle_count_test(
    handle, h_src, h_dst, h_wgt, h_verts, h_result, num_vertices, num_edges, num_results, FALSE);
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
    result |= RUN_MG_TEST(test_triangle_count, handle);
    result |= RUN_MG_TEST(test_triangle_count_dolphins, handle);

    cugraph_free_resource_handle(handle);
  }

  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return result;
}
