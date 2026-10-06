/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mg_test_utils.h"
#include "simple_cycles_capi_test_utils.h"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <mpi.h>
#include <stdlib.h>

typedef int32_t vertex_t;
typedef float weight_t;

static const vertex_t k_graph_src[] = {0, 1, 1, 2, 2, 3};
static const vertex_t k_graph_dst[] = {1, 2, 3, 0, 1, 2};
static const weight_t k_graph_wgt[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
static const size_t k_num_edges     = 6;

static const vertex_t k_all_cycles_vertices[] = {0, 1, 2, 0, 1, 3, 2, 1, 2, 1, 3, 2};
static const size_t k_all_cycles_offsets[]    = {0, 3, 7, 9, 12};
static const size_t k_num_all_cycles          = 4;

static void merge_gathered_cycles(const vertex_t* gathered_vertices,
                                  const size_t* gathered_offsets,
                                  const size_t* num_cycles_per_rank,
                                  int comm_size,
                                  vertex_t* merged_vertices,
                                  size_t* merged_offsets,
                                  size_t* merged_num_cycles)
{
  size_t merged_vertex_count = 0;
  size_t merged_cycle_count  = 0;
  size_t offset_base         = 0;
  size_t vertex_base         = 0;

  merged_offsets[0] = 0;

  for (int rank = 0; rank < comm_size; ++rank) {
    size_t rank_num_cycles = num_cycles_per_rank[rank];
    for (size_t cycle = 0; cycle < rank_num_cycles; ++cycle) {
      size_t begin = gathered_offsets[offset_base + cycle] + vertex_base;
      size_t end   = gathered_offsets[offset_base + cycle + 1] + vertex_base;
      for (size_t i = begin; i < end; ++i) {
        merged_vertices[merged_vertex_count++] = gathered_vertices[i];
      }
      merged_offsets[++merged_cycle_count] = merged_vertex_count;
    }
    if (rank_num_cycles > 0) { vertex_base += gathered_offsets[offset_base + rank_num_cycles]; }
    offset_base += rank_num_cycles + 1;
  }

  *merged_num_cycles = merged_cycle_count;
}

static int generic_mg_simple_cycles_test(const cugraph_resource_handle_t* handle,
                                         vertex_t* h_src,
                                         vertex_t* h_dst,
                                         weight_t* h_wgt,
                                         size_t num_edges,
                                         size_t length_bound,
                                         const vertex_t* expected_vertices,
                                         const size_t* expected_offsets,
                                         size_t expected_num_cycles)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph                 = NULL;
  cugraph_simple_cycles_result_t* result = NULL;

  ret_code =
    create_mg_test_graph(handle, h_src, h_dst, h_wgt, num_edges, FALSE, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_mg_test_graph failed.");

  ret_code = cugraph_simple_cycles(handle, graph, NULL, length_bound, FALSE, &result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_simple_cycles failed.");

  size_t local_num_cycles  = cugraph_simple_cycles_result_get_num_cycles(result);
  size_t global_num_cycles = cugraph_size_t_allreduce(handle, local_num_cycles);
  TEST_ASSERT(
    test_ret_value, global_num_cycles == expected_num_cycles, "unexpected global number of cycles");

  int comm_rank = 0;
  int comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  uint64_t local_num_cycles_u64 = (uint64_t)local_num_cycles;
  uint64_t* all_num_cycles_u64  = (uint64_t*)malloc((size_t)comm_size * sizeof(uint64_t));
  TEST_ASSERT(test_ret_value, all_num_cycles_u64 != NULL, "malloc failed for all_num_cycles");
  MPI_Allgather(
    &local_num_cycles_u64, 1, MPI_UINT64_T, all_num_cycles_u64, 1, MPI_UINT64_T, MPI_COMM_WORLD);

  cugraph_type_erased_device_array_view_t* cycle_vertices =
    cugraph_simple_cycles_result_get_cycle_vertices(result);
  cugraph_type_erased_device_array_view_t* cycle_offsets =
    cugraph_simple_cycles_result_get_cycle_offsets(result);

  size_t local_offsets_size = local_num_cycles + 1;
  size_t h_local_offsets[local_offsets_size];
  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_local_offsets, cycle_offsets, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "local cycle_offsets copy_to_host failed.");

  size_t local_vertex_count     = cugraph_type_erased_device_array_view_size(cycle_vertices);
  size_t gathered_vertices_size = cugraph_size_t_allreduce(handle, local_vertex_count);

  size_t gathered_offsets_size = 0;
  for (int rank = 0; rank < comm_size; ++rank) {
    gathered_offsets_size += (size_t)all_num_cycles_u64[rank] + 1;
  }

  vertex_t* gathered_vertices = NULL;
  size_t* gathered_offsets    = NULL;
  if (gathered_vertices_size > 0) {
    gathered_vertices = (vertex_t*)malloc(gathered_vertices_size * sizeof(vertex_t));
    TEST_ASSERT(test_ret_value, gathered_vertices != NULL, "malloc failed for gathered_vertices");
  }
  if (gathered_offsets_size > 0) {
    gathered_offsets = (size_t*)malloc(gathered_offsets_size * sizeof(size_t));
    TEST_ASSERT(test_ret_value, gathered_offsets != NULL, "malloc failed for gathered_offsets");
  }

  ret_code = cugraph_test_device_gatherv_fill(handle, cycle_vertices, gathered_vertices);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cycle_vertices gatherv failed.");

  ret_code = cugraph_test_host_gatherv_fill(
    handle, h_local_offsets, local_offsets_size, SIZE_T, gathered_offsets);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cycle_offsets gatherv failed.");

  if (comm_rank == 0) {
    size_t num_cycles_per_rank[comm_size];
    for (int rank = 0; rank < comm_size; ++rank) {
      num_cycles_per_rank[rank] = (size_t)all_num_cycles_u64[rank];
    }

    vertex_t* merged_vertices = (vertex_t*)malloc(gathered_vertices_size * sizeof(vertex_t));
    size_t merged_offsets[expected_num_cycles + 1];
    size_t merged_num_cycles = 0;

    TEST_ASSERT(test_ret_value, merged_vertices != NULL, "malloc failed for merged_vertices");

    merge_gathered_cycles(gathered_vertices,
                          gathered_offsets,
                          num_cycles_per_rank,
                          comm_size,
                          merged_vertices,
                          merged_offsets,
                          &merged_num_cycles);

    TEST_ASSERT(test_ret_value,
                simple_cycles_capi_validate_flat_cycles(merged_vertices,
                                                        merged_offsets,
                                                        merged_num_cycles,
                                                        expected_vertices,
                                                        expected_offsets,
                                                        expected_num_cycles),
                "simple cycles do not match expected values");

    free(merged_vertices);
  }

  free(gathered_vertices);
  free(gathered_offsets);

  free(all_num_cycles_u64);

  cugraph_type_erased_device_array_view_free(cycle_offsets);
  cugraph_type_erased_device_array_view_free(cycle_vertices);
  cugraph_simple_cycles_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

static int test_simple_cycles(const cugraph_resource_handle_t* handle)
{
  return generic_mg_simple_cycles_test(handle,
                                       (vertex_t*)k_graph_src,
                                       (vertex_t*)k_graph_dst,
                                       (weight_t*)k_graph_wgt,
                                       k_num_edges,
                                       4,
                                       k_all_cycles_vertices,
                                       k_all_cycles_offsets,
                                       k_num_all_cycles);
}

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_simple_cycles, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
