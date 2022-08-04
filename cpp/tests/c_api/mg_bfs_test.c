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

int generic_bfs_test(const cugraph_resource_handle_t* p_handle,
                     vertex_t* h_src,
                     vertex_t* h_dst,
                     weight_t* h_wgt,
                     vertex_t* h_seeds,
                     vertex_t const* expected_distances,
                     vertex_t const* expected_predecessors,
                     size_t num_vertices,
                     size_t num_edges,
                     size_t num_seeds,
                     size_t depth_limit,
                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* paths_result                   = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_seeds, INT32, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  p_source_view = cugraph_type_erased_device_array_view(p_sources);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, p_source_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = create_mg_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  ret_code = cugraph_bfs(
    p_handle, p_graph, p_source_view, FALSE, 10000000, TRUE, TRUE, &paths_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(paths_result);
  predecessors = cugraph_paths_result_get_predecessors(paths_result);
  distances    = cugraph_paths_result_get_distances(paths_result);

  vertex_t h_vertices[num_vertices];
  vertex_t h_predecessors[num_vertices];
  vertex_t h_distances[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_predecessors, predecessors, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  size_t num_local_vertices = cugraph_type_erased_device_array_view_size(vertices);

  for (int i = 0; (i < num_local_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                expected_distances[h_vertices[i]] == h_distances[i],
                "bfs distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "bfs predecessors don't match");
  }

  cugraph_paths_result_free(paths_result);
  cugraph_mg_graph_free(p_graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs(const cugraph_resource_handle_t* p_handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                 = {0};
  vertex_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

  // Bfs wants store_transposed = FALSE
  return generic_bfs_test(p_handle,
                          src,
                          dst,
                          wgt,
                          seeds,
                          expected_distances,
                          expected_predecessors,
                          num_vertices,
                          num_edges,
                          1,
                          10,
                          FALSE);
}

int main(int argc, char** argv)
{
  // Set up MPI:
  int comm_rank;
  int comm_size;
  int num_gpus_per_node;
  cudaError_t status;
  int mpi_status;
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
  int result        = 0;
  result |= RUN_MG_TEST(test_bfs, handle);

  cugraph_free_resource_handle(handle);
  free_raft_handle(raft_handle);

  C_MPI_TRY(MPI_Finalize());

  return result;
}
