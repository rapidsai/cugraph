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

int generic_k_truss_test(vertex_t* h_src,
                         vertex_t* h_dst,
                         weight_t* h_wgt,
                         vertex_t* h_expected_src,
                         vertex_t* h_expected_dst,
                         size_t* h_expected_offsets,
                         size_t num_vertices,
                         size_t num_edges,
                         size_t k,
                         bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  cugraph_resource_handle_t* resource_handle          = NULL;
  cugraph_graph_t* graph                              = NULL;
  cugraph_type_erased_device_array_t* seeds           = NULL;
  cugraph_type_erased_device_array_view_t* seeds_view = NULL;
  cugraph_induced_subgraph_result_t* result           = NULL;

  resource_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, resource_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(resource_handle, vertex_tid, edge_tid, h_src, h_dst, weight_tid, h_wgt, edge_type_tid, NULL, edge_id_tid, NULL, num_edges, store_transposed, FALSE, FALSE, FALSE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  /*
  ret_code =
    cugraph_type_erased_device_array_create(resource_handle, num_seeds, INT32, &seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seeds create failed.");

  seeds_view = cugraph_type_erased_device_array_view(seeds);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    resource_handle, seeds_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");
  */

  printf(" Running k_truss subgraph");
  ret_code =
    cugraph_k_truss_subgraph(resource_handle, graph, k, FALSE, &result, &ret_error);
  printf(" Done running k_truss subgraph");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_k_truss_subgraph failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* src;
    cugraph_type_erased_device_array_view_t* dst;
    cugraph_type_erased_device_array_view_t* wgt;
    cugraph_type_erased_device_array_view_t* offsets;

    src     = cugraph_induced_subgraph_get_sources(result);
    dst     = cugraph_induced_subgraph_get_destinations(result);
    wgt     = cugraph_induced_subgraph_get_edge_weights(result);
    offsets = cugraph_induced_subgraph_get_subgraph_offsets(result);

    size_t num_result_edges   = cugraph_type_erased_device_array_view_size(src);
    size_t num_result_offsets = cugraph_type_erased_device_array_view_size(offsets);

    vertex_t h_result_src[num_result_edges];
    vertex_t h_result_dst[num_result_edges];
    weight_t h_result_wgt[num_result_edges];
    size_t h_result_offsets[num_result_offsets];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_src, src, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_dst, dst, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

#if 0
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_wgt, wgt, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
#endif

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_offsets, offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    cugraph_type_erased_device_array_view_free(src);
    cugraph_type_erased_device_array_view_free(dst);
    cugraph_type_erased_device_array_view_free(wgt);
    cugraph_type_erased_device_array_view_free(offsets);
    cugraph_induced_subgraph_result_free(result);
  }

  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(resource_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_k_truss()
{
  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t k = 2;

  vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
  vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
  weight_t h_wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 6.1f};

  vertex_t h_result_src[]   = {0, 1, 1, 3, 1, 1, 3, 3, 4};
  vertex_t h_result_dst[]   = {1, 3, 4, 4, 3, 4, 4, 5, 5};
  size_t h_result_offsets[] = {0, 4, 9};

  return generic_k_truss_test(h_src,
                             h_dst,
                             h_wgt,
                             h_result_src,
                             h_result_dst,
                             h_result_offsets,
                             num_vertices,
                             num_edges,
                             k,
                             FALSE);
}


/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_k_truss);
  return result;
}
