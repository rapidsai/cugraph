/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */
#include "cugraph_c/sampling_algorithms.h"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;
typedef int32_t time_stamp_t;

cugraph_data_type_id_t vertex_tid    = INT32;
cugraph_data_type_id_t edge_tid      = INT32;
cugraph_data_type_id_t weight_tid    = FLOAT32;
cugraph_data_type_id_t edge_id_tid   = INT32;
cugraph_data_type_id_t edge_type_tid = INT32;
cugraph_data_type_id_t edge_time_tid = INT32;

const time_stamp_t MAX_EDGE_TIME = INT32_MAX;
const time_stamp_t MIN_EDGE_TIME = -1;

int vertex_id_compare_function(const void* a, const void* b)
{
  if (*((vertex_t*)a) < *((vertex_t*)b))
    return -1;
  else if (*((vertex_t*)a) > *((vertex_t*)b))
    return 1;
  else
    return 0;
}

int generic_uniform_temporal_neighbor_sample_test(
  const cugraph_resource_handle_t* handle,
  vertex_t* h_src,
  vertex_t* h_dst,
  weight_t* h_wgt,
  edge_t* h_edge_ids,
  int32_t* h_edge_types,
  time_stamp_t* h_edge_start_times,
  time_stamp_t* h_edge_end_times,
  size_t num_vertices,
  size_t num_edges,
  vertex_t* h_start,
  time_stamp_t* h_start_vertex_start_times,
  size_t* h_start_vertex_label_offsets,
  size_t num_start_vertices,
  size_t num_start_labels,
  int* fan_out,
  size_t fan_out_size,
  bool_t with_replacement,
  bool_t return_hops,
  cugraph_prior_sources_behavior_t prior_sources_behavior,
  bool_t dedupe_sources,
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison,
  bool_t renumber_results)
{
  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  ret_code = create_sg_test_graph(handle,
                                  vertex_tid,
                                  edge_tid,
                                  h_src,
                                  h_dst,
                                  weight_tid,
                                  h_wgt,
                                  edge_type_tid,
                                  h_edge_types,
                                  edge_id_tid,
                                  h_edge_ids,
                                  edge_time_tid,
                                  h_edge_start_times,
                                  h_edge_end_times,
                                  num_edges,
                                  FALSE,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  &graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start                              = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view                    = NULL;
  cugraph_type_erased_device_array_t* d_start_vertex_start_times           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_vertex_start_times_view = NULL;
  cugraph_type_erased_device_array_t* d_start_label_offsets                = NULL;
  cugraph_type_erased_device_array_view_t* d_start_label_offsets_view      = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view                    = NULL;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_start_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  if (h_start_vertex_start_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices, INT32, &d_start_vertex_start_times, &ret_error);
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_vertex_start_times create failed.");

    d_start_vertex_start_times_view =
      cugraph_type_erased_device_array_view(d_start_vertex_start_times);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_vertex_start_times_view, (byte_t*)h_start_vertex_start_times, &ret_error);
  } else {
    d_start_vertex_start_times_view = NULL;
  }

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_start_labels, SIZE_T, &d_start_label_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

  d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_label_offsets_view, (byte_t*)h_start_vertex_label_offsets, &ret_error);

  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels_offsets copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t* sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
  cugraph_sampling_set_renumber_results(sampling_options, renumber_results);
  cugraph_sampling_set_temporal_sampling_comparison(sampling_options, temporal_sampling_comparison);

  ret_code = cugraph_homogeneous_uniform_temporal_neighbor_sample(handle,
                                                                  rng_state,
                                                                  graph,
                                                                  "edge_start_time",
                                                                  d_start_view,
                                                                  d_start_vertex_start_times_view,
                                                                  d_start_label_offsets_view,
                                                                  h_fan_out_view,
                                                                  sampling_options,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_temporal_neighbor_sample failed.");

  test_ret_value = validate_sample_result(handle,
                                          result,
                                          h_src,
                                          h_dst,
                                          h_wgt,
                                          h_edge_ids,
                                          h_edge_types,
                                          h_edge_start_times,
                                          h_edge_end_times,
                                          num_vertices,
                                          num_edges,
                                          h_start,
                                          num_start_vertices,
                                          h_start_vertex_label_offsets,
                                          num_start_labels,
                                          fan_out,
                                          fan_out_size,
                                          sampling_options,
                                          true);
  TEST_ASSERT(test_ret_value, test_ret_value == 0, "validate_sample_result failed.");

  cugraph_sampling_options_free(sampling_options);
  cugraph_sample_result_free(result);
  cugraph_type_erased_device_array_view_free(d_start_view);
  cugraph_type_erased_device_array_view_free(d_start_vertex_start_times_view);
  cugraph_type_erased_device_array_view_free(d_start_label_offsets_view);
  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_type_erased_device_array_free(d_start);
  cugraph_type_erased_device_array_free(d_start_vertex_start_times);
  cugraph_type_erased_device_array_free(d_start_label_offsets);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_temporal_neighbor_sample_with_labels(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 1;
  size_t num_starts   = 2;

  vertex_t src[]                      = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  int32_t edge_types[]                = {7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1};

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  bool_t with_replacement                                 = TRUE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;
  cugraph_compression_type_t compression                  = COO;
  bool_t compress_per_hop                                 = FALSE;

  ret_code = create_sg_test_graph(handle,
                                  vertex_tid,
                                  edge_tid,
                                  src,
                                  dst,
                                  weight_tid,
                                  weight,
                                  edge_type_tid,
                                  edge_types,
                                  edge_id_tid,
                                  edge_ids,
                                  edge_time_tid,
                                  edge_start_times,
                                  edge_end_times,
                                  num_edges,
                                  FALSE,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  &graph,
                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start                         = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view               = NULL;
  cugraph_type_erased_device_array_t* d_start_label_offsets           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_label_offsets_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view               = NULL;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_starts + 1, SIZE_T, &d_start_label_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

  d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_label_offsets_view, (byte_t*)start_vertex_label_offsets, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, 1, INT32);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t* sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
  cugraph_sampling_set_renumber_results(sampling_options, renumber_results);
  cugraph_sampling_set_compression_type(sampling_options, compression);
  cugraph_sampling_set_compress_per_hop(sampling_options, compress_per_hop);
  cugraph_sampling_set_temporal_sampling_comparison(sampling_options, MONOTONICALLY_INCREASING);

  ret_code = cugraph_homogeneous_uniform_temporal_neighbor_sample(handle,
                                                                  rng_state,
                                                                  graph,
                                                                  "edge_start_time",
                                                                  d_start_view,
                                                                  NULL,
                                                                  d_start_label_offsets_view,
                                                                  h_fan_out_view,
                                                                  sampling_options,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_temporal_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_srcs;
  cugraph_type_erased_device_array_view_t* result_dsts;
  cugraph_type_erased_device_array_view_t* result_edge_id;
  cugraph_type_erased_device_array_view_t* result_weights;
  cugraph_type_erased_device_array_view_t* result_edge_types;
  cugraph_type_erased_device_array_view_t* result_hops;
  cugraph_type_erased_device_array_view_t* result_offsets;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_edge_types = cugraph_sample_result_get_edge_type(result);
  result_hops       = cugraph_sample_result_get_hop(result);
  result_offsets    = cugraph_sample_result_get_offsets(result);

  size_t result_size         = cugraph_type_erased_device_array_view_size(result_srcs);
  size_t result_offsets_size = cugraph_type_erased_device_array_view_size(result_offsets);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_edge_id[result_size];
  weight_t h_weight[result_size];
  int32_t h_edge_types[result_size];
  int32_t h_hops[result_size];
  size_t h_result_offsets[result_offsets_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_srcs, result_srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_dsts, result_dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_edge_id, result_edge_id, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_weight, result_weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_edge_types, result_edge_types, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  TEST_ASSERT(test_ret_value, result_hops == NULL, "hops was not empty");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_offsets, result_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M_w[num_vertices][num_vertices];
  edge_t M_edge_id[num_vertices][num_vertices];
  int32_t M_edge_type[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M_w[i][j]         = 0.0;
      M_edge_id[i][j]   = -1;
      M_edge_type[i][j] = -1;
    }

  for (int i = 0; i < num_edges; ++i) {
    M_w[src[i]][dst[i]]         = weight[i];
    M_edge_id[src[i]][dst[i]]   = edge_ids[i];
    M_edge_type[src[i]][dst[i]] = edge_types[i];
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M_w[h_srcs[i]][h_dsts[i]] == h_weight[i],
                "uniform_temporal_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_temporal_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_type[h_srcs[i]][h_dsts[i]] == h_edge_types[i],
                "uniform_temporal_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
  cugraph_sampling_options_free(sampling_options);

  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_temporal_neighbor_sample_clean(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_dedupe_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = TRUE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_unique_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = EXCLUDE;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_carry_over_sources(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_renumber_results(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = TRUE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_strictly_increasing(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = STRICTLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 6, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_monotonically_decreasing(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_DECREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 6, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_strictly_decreasing(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = STRICTLY_DECREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                      = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                      = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                   = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                   = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]     = {0, 1, 2, 3, 4, 5, 6, 6, 8};
  time_stamp_t edge_end_times[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int test_uniform_temporal_neighbor_sample_with_vertex_start_times(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid                                   = INT32;
  cugraph_data_type_id_t edge_tid                                     = INT32;
  cugraph_data_type_id_t weight_tid                                   = FLOAT32;
  cugraph_data_type_id_t edge_id_tid                                  = INT32;
  cugraph_data_type_id_t edge_type_tid                                = INT32;
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison = MONOTONICALLY_INCREASING;

  size_t num_edges        = 9;
  size_t num_vertices     = 6;
  size_t fan_out_size     = 3;
  size_t num_starts       = 2;
  size_t num_start_labels = 3;

  vertex_t src[]                          = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                          = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]                       = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]                       = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]                    = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  time_stamp_t edge_start_times[]         = {0, 1, 2, 3, 4, 5, 6, 6, 8};
  time_stamp_t edge_end_times[]           = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  vertex_t start[]                        = {2, 3};
  time_stamp_t start_vertex_start_times[] = {0, 8};

  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       start_vertex_start_times,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_start_labels,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       temporal_sampling_comparison,
                                                       renumber_results);
}

int main(int argc, char** argv)
{
  cugraph_resource_handle_t* handle = NULL;

  handle = cugraph_create_resource_handle(NULL);

  int result = 0;
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_with_labels, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_clean, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_dedupe_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_unique_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_carry_over_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_renumber_results, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_strictly_increasing, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_monotonically_decreasing, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_strictly_decreasing, handle);
  result |= RUN_TEST_NEW(test_uniform_temporal_neighbor_sample_with_vertex_start_times, handle);

  cugraph_free_resource_handle(handle);

  return result;
}
