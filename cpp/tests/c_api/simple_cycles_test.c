/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */
#include "simple_cycles_capi_test_utils.h"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

typedef int32_t vertex_t;
typedef float weight_t;

static const vertex_t k_graph_src[] = {0, 1, 1, 2, 2, 3};
static const vertex_t k_graph_dst[] = {1, 2, 3, 0, 1, 2};
static const weight_t k_graph_wgt[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
static const size_t k_num_edges     = 6;

static const vertex_t k_all_cycles_vertices[] = {0, 1, 2, 0, 1, 3, 2, 1, 2, 1, 3, 2};
static const size_t k_all_cycles_offsets[]    = {0, 3, 7, 9, 12};
static const size_t k_num_all_cycles          = 4;

static const vertex_t k_seed3_cycles_vertices[] = {0, 1, 3, 2, 1, 3, 2};
static const size_t k_seed3_cycles_offsets[]    = {0, 4, 7};
static const size_t k_num_seed3_cycles          = 2;

static int generic_simple_cycles_test(vertex_t* h_src,
                                      vertex_t* h_dst,
                                      weight_t* h_wgt,
                                      size_t num_edges,
                                      const vertex_t* h_seed_vertices,
                                      size_t num_seed_vertices,
                                      size_t length_bound,
                                      const vertex_t* expected_vertices,
                                      const size_t* expected_offsets,
                                      size_t expected_num_cycles)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* handle                   = NULL;
  cugraph_graph_t* graph                              = NULL;
  cugraph_simple_cycles_result_t* result              = NULL;
  cugraph_type_erased_device_array_t* seeds           = NULL;
  cugraph_type_erased_device_array_view_t* seeds_view = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, FALSE, FALSE, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (num_seed_vertices > 0) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_seed_vertices, INT32, &seeds, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seed array create failed.");
    seeds_view = cugraph_type_erased_device_array_view(seeds);
    ret_code   = cugraph_type_erased_device_array_view_copy_from_host(
      handle, seeds_view, (byte_t*)h_seed_vertices, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seed copy_from_host failed.");
  }

  ret_code =
    cugraph_simple_cycles(handle, graph, seeds_view, length_bound, FALSE, &result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_simple_cycles failed.");

  cugraph_type_erased_device_array_view_t* cycle_vertices =
    cugraph_simple_cycles_result_get_cycle_vertices(result);
  cugraph_type_erased_device_array_view_t* cycle_offsets =
    cugraph_simple_cycles_result_get_cycle_offsets(result);

  size_t num_cycles = cugraph_simple_cycles_result_get_num_cycles(result);
  TEST_ASSERT(test_ret_value, num_cycles == expected_num_cycles, "unexpected number of cycles");

  size_t num_offsets        = num_cycles + 1;
  size_t num_cycle_vertices = cugraph_type_erased_device_array_view_size(cycle_vertices);

  vertex_t h_cycle_vertices[num_cycle_vertices];
  size_t h_cycle_offsets[num_offsets];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_cycle_vertices, cycle_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cycle_vertices copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_cycle_offsets, cycle_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cycle_offsets copy_to_host failed.");

  TEST_ASSERT(test_ret_value,
              simple_cycles_capi_validate_flat_cycles(h_cycle_vertices,
                                                      h_cycle_offsets,
                                                      num_cycles,
                                                      expected_vertices,
                                                      expected_offsets,
                                                      expected_num_cycles),
              "simple cycles do not match expected values");

  cugraph_type_erased_device_array_view_free(cycle_offsets);
  cugraph_type_erased_device_array_view_free(cycle_vertices);
  cugraph_simple_cycles_result_free(result);
  if (seeds_view != NULL) { cugraph_type_erased_device_array_view_free(seeds_view); }
  if (seeds != NULL) { cugraph_type_erased_device_array_free(seeds); }
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

static int test_simple_cycles_all()
{
  return generic_simple_cycles_test((vertex_t*)k_graph_src,
                                    (vertex_t*)k_graph_dst,
                                    (weight_t*)k_graph_wgt,
                                    k_num_edges,
                                    NULL,
                                    0,
                                    4,
                                    k_all_cycles_vertices,
                                    k_all_cycles_offsets,
                                    k_num_all_cycles);
}

static int test_simple_cycles_with_seeds()
{
  vertex_t seeds[] = {3};
  return generic_simple_cycles_test((vertex_t*)k_graph_src,
                                    (vertex_t*)k_graph_dst,
                                    (weight_t*)k_graph_wgt,
                                    k_num_edges,
                                    seeds,
                                    1,
                                    4,
                                    k_seed3_cycles_vertices,
                                    k_seed3_cycles_offsets,
                                    k_num_seed3_cycles);
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_simple_cycles_all);
  result |= RUN_TEST(test_simple_cycles_with_seeds);
  return result;
}
