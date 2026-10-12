/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_minimum_spanning_tree_test(vertex_t* h_src,
                                       vertex_t* h_dst,
                                       weight_t* h_wgt,
                                       vertex_t* h_expected_src,
                                       vertex_t* h_expected_dst,
                                       weight_t* h_expected_wgt,
                                       size_t* h_expected_offsets,
                                       size_t num_vertices,
                                       size_t num_edges,
                                       size_t num_expected_offsets,
                                       size_t num_expected_edges,
                                       bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  cugraph_resource_handle_t* resource_handle          = NULL;
  cugraph_graph_t* graph                              = NULL;
  cugraph_type_erased_device_array_t* seeds           = NULL;
  cugraph_type_erased_device_array_view_t* seeds_view = NULL;
  cugraph_induced_subgraph_result_t* result           = NULL;

  resource_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, resource_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(resource_handle,
                                  vertex_tid,
                                  edge_tid,
                                  h_src,
                                  h_dst,
                                  weight_tid,
                                  h_wgt,
                                  edge_type_tid,
                                  NULL,
                                  edge_id_tid,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  store_transposed,
                                  FALSE,
                                  TRUE,
                                  FALSE,
                                  &graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_minimum_spanning_tree(resource_handle, graph, FALSE, &result, &ret_error);
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

    if (wgt != NULL) {
      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        resource_handle, (byte_t*)h_result_wgt, wgt, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
    }

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_offsets, offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    TEST_ASSERT(
      test_ret_value, num_result_edges == num_expected_edges, "results not the same size");

    for (size_t i = 0; (i < num_expected_offsets) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(
        test_ret_value, h_expected_offsets[i] == h_result_offsets[i], "graph offsets should match");
    }

    // With unit weights, any spanning tree is a minimum spanning tree.
    if (h_wgt == NULL) {
      vertex_t component[num_vertices];
      for (size_t i = 0; i < num_vertices; ++i)
        component[i] = i;

      for (size_t i = 0; (i < num_result_edges) && (test_ret_value == 0); ++i) {
        vertex_t src = h_result_src[i];
        vertex_t dst = h_result_dst[i];
        bool_t found = FALSE;
        for (size_t j = 0; (j < num_edges) && !found; ++j)
          found = (h_src[j] == src) && (h_dst[j] == dst);
        TEST_ASSERT(test_ret_value, found, "MST edge is not in the input graph");
        TEST_ASSERT(test_ret_value,
                    (wgt != NULL) && nearlyEqual(h_result_wgt[i], 1.0f, 0.001f),
                    "unweighted MST edge should have weight 1");

        size_t reverse_count = 0;
        for (size_t j = 0; j < num_result_edges; ++j)
          reverse_count += (h_result_src[j] == dst) && (h_result_dst[j] == src);
        TEST_ASSERT(test_ret_value, reverse_count == 1, "MST edges should be symmetric");

        if ((test_ret_value == 0) && (src < dst)) {
          vertex_t src_component = component[src];
          vertex_t dst_component = component[dst];
          TEST_ASSERT(test_ret_value, src_component != dst_component, "MST contains a cycle");
          for (size_t j = 0; j < num_vertices; ++j)
            if (component[j] == dst_component) component[j] = src_component;
        }
      }
      for (size_t i = 0; (i < num_vertices) && (test_ret_value == 0); ++i)
        TEST_ASSERT(test_ret_value, component[i] == component[0], "MST should span all vertices");
    }

    for (size_t i = 0; (h_wgt != NULL) && (i < num_expected_edges) && (test_ret_value == 0); ++i) {
      bool_t found = FALSE;
      for (size_t j = 0; (j < num_expected_edges) && !found; ++j) {
        if ((h_expected_src[i] == h_result_src[j]) && (h_expected_dst[i] == h_result_dst[j]))
          if (wgt != NULL) {
            found = (nearlyEqual(h_expected_wgt[i], h_result_wgt[j], 0.001));
          } else {
            found = TRUE;
          }
      }
      TEST_ASSERT(test_ret_value, found, "extracted an edge that doesn't match");
    }

    cugraph_type_erased_device_array_view_free(src);
    cugraph_type_erased_device_array_view_free(dst);
    cugraph_type_erased_device_array_view_free(wgt);
    cugraph_type_erased_device_array_view_free(offsets);
    cugraph_induced_subgraph_result_free(result);
  }

  cugraph_graph_free(graph);
  cugraph_free_resource_handle(resource_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_minimum_spanning_tree()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[] = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  vertex_t h_result_src[]     = {0, 1, 2, 3, 4, 5, 1, 1, 1, 4};
  vertex_t h_result_dst[]     = {1, 0, 1, 1, 1, 4, 2, 3, 4, 5};
  weight_t h_result_wgt[]     = {0.1, 0.1, 3.1, 2.1, 1.1, 3.2, 3.1, 2.1, 1.1, 3.2};
  size_t h_result_offsets[]   = {0, 10};
  size_t num_expected_edges   = 10;
  size_t num_expected_offsets = 2;

  return generic_minimum_spanning_tree_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            h_result_src,
                                            h_result_dst,
                                            h_result_wgt,
                                            h_result_offsets,
                                            num_vertices,
                                            num_edges,
                                            num_expected_offsets,
                                            num_expected_edges,
                                            FALSE);
}

int test_minimum_spanning_tree_no_weights()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};

  size_t h_result_offsets[]   = {0, 10};
  size_t num_expected_edges   = 10;
  size_t num_expected_offsets = 2;

  return generic_minimum_spanning_tree_test(h_src,
                                            h_dst,
                                            NULL,
                                            NULL,
                                            NULL,
                                            NULL,
                                            h_result_offsets,
                                            num_vertices,
                                            num_edges,
                                            num_expected_offsets,
                                            num_expected_edges,
                                            FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_minimum_spanning_tree);
  result |= RUN_TEST(test_minimum_spanning_tree_no_weights);
  return result;
}
