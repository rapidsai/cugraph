/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
  time_stamp_t* h_start_times,
  time_stamp_t* h_start_end_times,
  size_t* h_start_vertex_label_offsets,
  size_t num_start_vertices,
  size_t num_start_labels,
  int* fan_out,
  size_t fan_out_size,
  bool_t with_replacement,
  bool_t return_hops,
  cugraph_prior_sources_behavior_t prior_sources_behavior,
  bool_t dedupe_sources,
  bool_t renumber_results,
  bool_t is_multigraph,
  vertex_t* expected_dsts,
  size_t num_expected_edges)
{
  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = create_mg_test_graph_new(handle,
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
                                      is_multigraph,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start                         = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view               = NULL;
  cugraph_type_erased_device_array_t* d_start_times                   = NULL;
  cugraph_type_erased_device_array_view_t* d_start_times_view         = NULL;
  cugraph_type_erased_device_array_t* d_start_end_times               = NULL;
  cugraph_type_erased_device_array_view_t* d_start_end_times_view     = NULL;
  cugraph_type_erased_device_array_t* d_start_label_offsets           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_label_offsets_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view               = NULL;

  if (rank > 0) num_start_vertices = 0;

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_start_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  if (h_start_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices, INT32, &d_start_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_times create failed.");
    d_start_times_view = cugraph_type_erased_device_array_view(d_start_times);
    ret_code           = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_times_view, (byte_t*)h_start_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start times copy failed.");
  }

  if (h_start_end_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices, INT32, &d_start_end_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_end_times create failed.");
    d_start_end_times_view = cugraph_type_erased_device_array_view(d_start_end_times);
    ret_code               = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_end_times_view, (byte_t*)h_start_end_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start end times copy failed.");
  }

  if (h_start_vertex_label_offsets != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices + 1, SIZE_T, &d_start_label_offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

    d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_label_offsets_view, (byte_t*)h_start_vertex_label_offsets, &ret_error);

    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels_offsets copy_from_host failed.");
  }

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t* sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
  cugraph_sampling_set_renumber_results(sampling_options, renumber_results);
  cugraph_sampling_set_temporal_sampling_comparison(sampling_options, STRICTLY_INCREASING);
  // Temporal neighbor sampling requires disjoint sampling.
  cugraph_sampling_set_disjoint_sampling(sampling_options, TRUE);

  ret_code = cugraph_homogeneous_uniform_temporal_neighbor_sample(handle,
                                                                  rng_state,
                                                                  graph,
                                                                  "edge_start_time",
                                                                  d_start_view,
                                                                  d_start_times_view,
                                                                  d_start_end_times_view,
                                                                  d_start_label_offsets_view,
                                                                  h_fan_out_view,
                                                                  sampling_options,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  if (test_ret_value == 0) {
    test_ret_value = mg_validate_sample_result(handle,
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
  }

  if ((test_ret_value == 0) && (expected_dsts != NULL)) {
    cugraph_type_erased_device_array_view_t* result_dsts =
      cugraph_sample_result_get_destinations(result);
    cugraph_type_erased_device_array_view_t* result_times =
      cugraph_sample_result_get_edge_start_time(result);
    size_t result_size           = cugraph_test_device_gatherv_size(handle, result_dsts);
    vertex_t* gathered_dsts      = (vertex_t*)malloc(result_size * sizeof(vertex_t));
    time_stamp_t* gathered_times = (time_stamp_t*)malloc(result_size * sizeof(time_stamp_t));

    ret_code = cugraph_test_device_gatherv_fill(handle, result_dsts, gathered_dsts);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather destinations failed.");
    ret_code = cugraph_test_device_gatherv_fill(handle, result_times, gathered_times);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather edge times failed.");

    if (rank == 0) {
      TEST_ASSERT(test_ret_value, result_size == num_expected_edges, "unexpected result size.");
      for (size_t i = 0; i < result_size; ++i) {
        bool found = false;
        for (size_t j = 0; j < num_expected_edges; ++j) {
          if (gathered_dsts[i] == expected_dsts[j]) {
            found = true;
            break;
          }
        }
        TEST_ASSERT(test_ret_value, found, "unexpected destination for seed time window.");
        TEST_ASSERT(
          test_ret_value,
          (gathered_times[i] >= h_start_times[0]) && (gathered_times[i] <= h_start_end_times[0]),
          "sampled edge time falls outside seed time window.");
      }
    }
    free(gathered_dsts);
    free(gathered_times);
  }

  cugraph_sampling_options_free(sampling_options);
  if (result != NULL) { cugraph_sample_result_free(result); }
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_uniform_temporal_neighbor_sample(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]             = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]             = {1, 3, 4, 0, 1, 3, 5, 5};
  time_stamp_t start_times[] = {0, 1, 2, 3, 4, 5, 6, 7};
  time_stamp_t end_times[]   = {1, 2, 3, 4, 5, 6, 7, 8};
  edge_t idx[]               = {0, 1, 2, 3, 4, 5, 6, 7};
  // Disjoint sampling forbids duplicate starting vertices within a label.
  vertex_t start[]                    = {2, 3};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {1, 2};

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       NULL,
                                                       idx,
                                                       NULL,
                                                       start_times,
                                                       end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_starts + 1,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       FALSE,
                                                       FALSE,
                                                       NULL,
                                                       0);
}

int test_uniform_temporal_neighbor_sample_time_window(const cugraph_resource_handle_t* handle)
{
  size_t num_edges        = 4;
  size_t num_vertices     = 5;
  size_t fan_out_size     = 1;
  size_t num_starts       = 1;
  size_t num_start_labels = 2;

  vertex_t src[]                        = {0, 0, 0, 0};
  vertex_t dst[]                        = {1, 2, 3, 4};
  edge_t edge_ids[]                     = {0, 1, 2, 3};
  time_stamp_t edge_start_times[]       = {1, 2, 3, 5};
  time_stamp_t edge_end_times[]         = {2, 3, 4, 6};
  vertex_t start[]                      = {0};
  time_stamp_t start_vertex_times[]     = {0};
  time_stamp_t start_vertex_end_times[] = {2};
  size_t start_vertex_label_offsets[]   = {0, 1};
  int fan_out[]                         = {-1};
  vertex_t expected_dsts[]              = {1, 2};

  // The seed window [0, 2] admits times 1 and 2 and excludes edges at times 3 and 5.
  return generic_uniform_temporal_neighbor_sample_test(
    handle,
    src,
    dst,
    NULL,
    edge_ids,
    NULL,
    edge_start_times,
    edge_end_times,
    num_vertices,
    num_edges,
    start,
    start_vertex_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    num_start_labels,
    fan_out,
    fan_out_size,
    FALSE,
    TRUE,
    DEFAULT,
    FALSE,
    FALSE,
    FALSE,
    expected_dsts,
    sizeof(expected_dsts) / sizeof(expected_dsts[0]));
}

int test_uniform_temporal_neighbor_from_alex(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 12;
  size_t num_vertices = 5;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]                  = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
  vertex_t dst[]                  = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
  edge_t edge_ids[]               = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t edge_types[]            = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
  weight_t weights[]              = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
  time_stamp_t edge_start_times[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  time_stamp_t edge_end_times[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  vertex_t start[]                = {0, 4};
  size_t start_vertex_label_offsets[] = {0, 1, 2};
  int fan_out[]                       = {2, 2};

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weights,
                                                       edge_ids,
                                                       edge_types,
                                                       edge_start_times,
                                                       edge_end_times,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       NULL,
                                                       start_vertex_label_offsets,
                                                       num_starts,
                                                       num_starts + 1,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       renumber_results,
                                                       TRUE,
                                                       NULL,
                                                       0);
}

int test_uniform_temporal_neighbor_sample_dedupe_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]               = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]               = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]            = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]            = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]         = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]             = {2, 3};
  size_t start_label_offsets[] = {0, 1, 2};
  int fan_out[]                = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = TRUE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       NULL,
                                                       NULL,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       NULL,
                                                       start_label_offsets,
                                                       num_starts,
                                                       num_starts + 1,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       FALSE,
                                                       FALSE,
                                                       NULL,
                                                       0);
}

int test_uniform_temporal_neighbor_sample_unique_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]               = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]               = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]            = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]            = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]         = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]             = {2, 3};
  size_t start_label_offsets[] = {0, 1, 2};
  int fan_out[]                = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = EXCLUDE;
  bool_t dedupe_sources                                   = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       NULL,
                                                       NULL,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       NULL,
                                                       start_label_offsets,
                                                       num_starts,
                                                       num_starts + 1,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       FALSE,
                                                       FALSE,
                                                       NULL,
                                                       0);
}

int test_uniform_temporal_neighbor_sample_carry_over_sources(
  const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]               = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]               = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]            = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]            = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[]         = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]             = {2, 3};
  size_t start_label_offsets[] = {0, 1, 2};
  int fan_out[]                = {-1, -1, -1};

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
  bool_t dedupe_sources                                   = FALSE;

  return generic_uniform_temporal_neighbor_sample_test(handle,
                                                       src,
                                                       dst,
                                                       weight,
                                                       edge_ids,
                                                       edge_types,
                                                       NULL,
                                                       NULL,
                                                       num_vertices,
                                                       num_edges,
                                                       start,
                                                       NULL,
                                                       NULL,
                                                       start_label_offsets,
                                                       num_starts,
                                                       num_starts + 1,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       FALSE,
                                                       FALSE,
                                                       NULL,
                                                       0);
}

/******************************************************************************/

typedef struct {
  vertex_t src;
  vertex_t dst;
  time_stamp_t edge_start_time;
  int32_t hop;
} expected_temporal_sample_edge_t;

int expected_temporal_sample_edge_compare(const void* a, const void* b)
{
  expected_temporal_sample_edge_t const* lhs = (expected_temporal_sample_edge_t const*)a;
  expected_temporal_sample_edge_t const* rhs = (expected_temporal_sample_edge_t const*)b;

  if (lhs->src != rhs->src) return (lhs->src < rhs->src) ? -1 : 1;
  if (lhs->dst != rhs->dst) return (lhs->dst < rhs->dst) ? -1 : 1;
  if (lhs->edge_start_time != rhs->edge_start_time) {
    return (lhs->edge_start_time < rhs->edge_start_time) ? -1 : 1;
  }
  if (lhs->hop != rhs->hop) return (lhs->hop < rhs->hop) ? -1 : 1;
  return 0;
}

int compare_mg_temporal_neighbor_sample_to_expected(
  const cugraph_resource_handle_t* handle,
  const cugraph_sample_result_t* result,
  expected_temporal_sample_edge_t const* expected_edges,
  size_t num_expected_edges,
  size_t fan_out_size)
{
  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;
  int rank                      = cugraph_resource_handle_get_rank(handle);

  cugraph_type_erased_device_array_view_t* result_srcs = cugraph_sample_result_get_majors(result);
  cugraph_type_erased_device_array_view_t* result_dsts =
    cugraph_sample_result_get_destinations(result);
  cugraph_type_erased_device_array_view_t* result_edge_start_times =
    cugraph_sample_result_get_edge_start_time(result);
  cugraph_type_erased_device_array_view_t* result_hops = cugraph_sample_result_get_hop(result);
  cugraph_type_erased_device_array_view_t* result_label_hop_offsets =
    cugraph_sample_result_get_label_hop_offsets(result);
  cugraph_type_erased_device_array_view_t* result_label_type_hop_offsets =
    cugraph_sample_result_get_label_type_hop_offsets(result);

  size_t local_size  = cugraph_type_erased_device_array_view_size(result_srcs);
  size_t result_size = cugraph_test_device_gatherv_size(handle, result_srcs);

  vertex_t* h_srcs                 = (vertex_t*)malloc(local_size * sizeof(vertex_t));
  vertex_t* h_dsts                 = (vertex_t*)malloc(local_size * sizeof(vertex_t));
  time_stamp_t* h_edge_start_times = (time_stamp_t*)malloc(local_size * sizeof(time_stamp_t));
  int32_t* h_hops                  = (int32_t*)malloc(local_size * sizeof(int32_t));

  vertex_t* gathered_srcs                          = NULL;
  vertex_t* gathered_dsts                          = NULL;
  time_stamp_t* gathered_edge_start_times          = NULL;
  int32_t* gathered_hops                           = NULL;
  expected_temporal_sample_edge_t* actual_edges    = NULL;
  expected_temporal_sample_edge_t* sorted_expected = NULL;

  if (!h_srcs || !h_dsts || !h_edge_start_times || !h_hops) {
    test_ret_value = 1;
    goto cleanup;
  }

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_srcs, result_srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_dsts, result_dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_edge_start_times, result_edge_start_times, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  if (result_hops != NULL) {
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_hops, result_hops, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
  } else if ((result_label_type_hop_offsets != NULL) || (result_label_hop_offsets != NULL)) {
    cugraph_type_erased_device_array_view_t* offsets_view = (result_label_type_hop_offsets != NULL)
                                                              ? result_label_type_hop_offsets
                                                              : result_label_hop_offsets;
    size_t offsets_size = cugraph_type_erased_device_array_view_size(offsets_view);
    size_t* h_offsets   = (size_t*)malloc(offsets_size * sizeof(size_t));
    if (!h_offsets) {
      test_ret_value = 1;
      goto cleanup;
    }
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_offsets, offsets_view, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    int32_t hop = 0;
    for (size_t i = 0; i < offsets_size - 1; ++i) {
      for (size_t j = h_offsets[i]; j < h_offsets[i + 1]; ++j) {
        h_hops[j] = hop;
      }
      hop = (hop + 1) % (int32_t)fan_out_size;
    }
    free(h_offsets);
  } else {
    for (size_t i = 0; i < local_size; ++i) {
      h_hops[i] = 0;
    }
  }

  if (rank == 0) {
    gathered_srcs             = (vertex_t*)malloc(result_size * sizeof(vertex_t));
    gathered_dsts             = (vertex_t*)malloc(result_size * sizeof(vertex_t));
    gathered_edge_start_times = (time_stamp_t*)malloc(result_size * sizeof(time_stamp_t));
    gathered_hops             = (int32_t*)malloc(result_size * sizeof(int32_t));
    actual_edges              = (expected_temporal_sample_edge_t*)malloc(
      result_size * sizeof(expected_temporal_sample_edge_t));
    sorted_expected = (expected_temporal_sample_edge_t*)malloc(
      num_expected_edges * sizeof(expected_temporal_sample_edge_t));
    if (!gathered_srcs || !gathered_dsts || !gathered_edge_start_times || !gathered_hops ||
        !actual_edges || !sorted_expected) {
      test_ret_value = 1;
      goto cleanup;
    }
  }

  ret_code = cugraph_test_host_gatherv_fill(handle, h_srcs, local_size, INT32, gathered_srcs);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather sources failed.");
  ret_code = cugraph_test_host_gatherv_fill(handle, h_dsts, local_size, INT32, gathered_dsts);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather destinations failed.");
  ret_code = cugraph_test_host_gatherv_fill(
    handle, h_edge_start_times, local_size, INT32, gathered_edge_start_times);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather edge times failed.");
  ret_code = cugraph_test_host_gatherv_fill(handle, h_hops, local_size, INT32, gathered_hops);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gather hops failed.");

  if (rank == 0) {
    TEST_ASSERT(
      test_ret_value, result_size == num_expected_edges, "unexpected number of sampled edges");

    for (size_t i = 0; i < result_size; ++i) {
      actual_edges[i].src             = gathered_srcs[i];
      actual_edges[i].dst             = gathered_dsts[i];
      actual_edges[i].edge_start_time = gathered_edge_start_times[i];
      actual_edges[i].hop             = gathered_hops[i];
    }

    memcpy(sorted_expected,
           expected_edges,
           num_expected_edges * sizeof(expected_temporal_sample_edge_t));
    qsort(sorted_expected,
          num_expected_edges,
          sizeof(expected_temporal_sample_edge_t),
          expected_temporal_sample_edge_compare);
    qsort(actual_edges,
          result_size,
          sizeof(expected_temporal_sample_edge_t),
          expected_temporal_sample_edge_compare);

    for (size_t i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(test_ret_value,
                  actual_edges[i].src == sorted_expected[i].src,
                  "sampled edge source does not match expected");
      TEST_ASSERT(test_ret_value,
                  actual_edges[i].dst == sorted_expected[i].dst,
                  "sampled edge destination does not match expected");
      TEST_ASSERT(test_ret_value,
                  actual_edges[i].edge_start_time == sorted_expected[i].edge_start_time,
                  "sampled edge start time does not match expected");
      TEST_ASSERT(test_ret_value,
                  actual_edges[i].hop == sorted_expected[i].hop,
                  "sampled edge hop does not match expected");
    }
  }

cleanup:
  free(h_srcs);
  free(h_dsts);
  free(h_edge_start_times);
  free(h_hops);
  free(gathered_srcs);
  free(gathered_dsts);
  free(gathered_edge_start_times);
  free(gathered_hops);
  free(actual_edges);
  free(sorted_expected);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_neighbor_sample_expected_edges_test(
  const cugraph_resource_handle_t* handle,
  vertex_t* h_src,
  vertex_t* h_dst,
  weight_t* h_wgt,
  edge_t* h_edge_ids,
  int32_t* h_edge_types,
  time_stamp_t* h_edge_start_times,
  time_stamp_t* h_edge_end_times,
  size_t num_edges,
  vertex_t* h_start,
  time_stamp_t* h_start_times,
  time_stamp_t* h_start_end_times,
  size_t* h_start_vertex_label_offsets,
  size_t num_start_vertices,
  int* fan_out,
  size_t fan_out_size,
  cugraph_temporal_sampling_comparison_t temporal_sampling_comparison,
  cugraph_neighbor_selection_t neighbor_selection,
  expected_temporal_sample_edge_t const* expected_edges,
  size_t num_expected_edges)
{
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = create_mg_test_graph_new(handle,
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

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start                         = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view               = NULL;
  cugraph_type_erased_device_array_t* d_start_times                   = NULL;
  cugraph_type_erased_device_array_view_t* d_start_times_view         = NULL;
  cugraph_type_erased_device_array_t* d_start_end_times               = NULL;
  cugraph_type_erased_device_array_view_t* d_start_end_times_view     = NULL;
  cugraph_type_erased_device_array_t* d_start_label_offsets           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_label_offsets_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view               = NULL;

  size_t local_num_starts = (rank > 0) ? 0 : num_start_vertices;

  ret_code =
    cugraph_type_erased_device_array_create(handle, local_num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");
  d_start_view = cugraph_type_erased_device_array_view(d_start);
  ret_code     = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start copy failed.");

  if (h_start_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, local_num_starts, INT32, &d_start_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_times create failed.");
    d_start_times_view = cugraph_type_erased_device_array_view(d_start_times);
    ret_code           = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_times_view, (byte_t*)h_start_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start times copy failed.");
  }

  if (h_start_end_times != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, local_num_starts, INT32, &d_start_end_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_end_times create failed.");
    d_start_end_times_view = cugraph_type_erased_device_array_view(d_start_end_times);
    ret_code               = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_end_times_view, (byte_t*)h_start_end_times, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start end times copy failed.");
  }

  if (h_start_vertex_label_offsets != NULL) {
    size_t local_label_offsets_size = (rank > 0) ? 0 : (num_start_vertices + 1);
    ret_code                        = cugraph_type_erased_device_array_create(
      handle, local_label_offsets_size, SIZE_T, &d_start_label_offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");
    d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);
    ret_code                   = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_label_offsets_view, (byte_t*)h_start_vertex_label_offsets, &ret_error);
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels_offsets copy_from_host failed.");
  }

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t* sampling_options;
  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, FALSE);
  cugraph_sampling_set_return_hops(sampling_options, TRUE);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, DEFAULT);
  cugraph_sampling_set_dedupe_sources(sampling_options, FALSE);
  cugraph_sampling_set_renumber_results(sampling_options, FALSE);
  cugraph_sampling_set_temporal_sampling_comparison(sampling_options, temporal_sampling_comparison);
  cugraph_sampling_set_disjoint_sampling(sampling_options, TRUE);

  ret_code = cugraph_neighbor_sample(handle,
                                     rng_state,
                                     graph,
                                     NULL,
                                     d_start_view,
                                     d_start_times_view,
                                     d_start_end_times_view,
                                     d_start_label_offsets_view,
                                     NULL,
                                     h_fan_out_view,
                                     1,
                                     neighbor_selection,
                                     &temporal_sampling_comparison,
                                     sampling_options,
                                     FALSE,
                                     &result,
                                     &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_neighbor_sample failed.");

  if (test_ret_value == 0) {
    test_ret_value = compare_mg_temporal_neighbor_sample_to_expected(
      handle, result, expected_edges, num_expected_edges, fan_out_size);
    TEST_ASSERT(test_ret_value, test_ret_value == 0, "compare to expected failed.");
  }

  cugraph_sampling_options_free(sampling_options);
  if (result != NULL) { cugraph_sample_result_free(result); }
  cugraph_type_erased_device_array_view_free(d_start_view);
  if (d_start_times_view != NULL) {
    cugraph_type_erased_device_array_view_free(d_start_times_view);
  }
  if (d_start_end_times_view != NULL) {
    cugraph_type_erased_device_array_view_free(d_start_end_times_view);
  }
  if (d_start_label_offsets_view != NULL) {
    cugraph_type_erased_device_array_view_free(d_start_label_offsets_view);
  }
  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_type_erased_device_array_free(d_start);
  if (d_start_times != NULL) { cugraph_type_erased_device_array_free(d_start_times); }
  if (d_start_end_times != NULL) { cugraph_type_erased_device_array_free(d_start_end_times); }
  if (d_start_label_offsets != NULL) {
    cugraph_type_erased_device_array_free(d_start_label_offsets);
  }
  cugraph_rng_state_free(rng_state);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_mg_neighbor_sample_first_single_hop(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t fan_out_size = 1;
  size_t num_starts   = 1;

  vertex_t src[]                          = {0, 0, 0};
  vertex_t dst[]                          = {1, 2, 3};
  edge_t edge_ids[]                       = {0, 1, 2};
  weight_t weight[]                       = {0.1, 0.2, 0.3};
  int32_t edge_types[]                    = {0, 1, 2};
  time_stamp_t edge_start_times[]         = {10, 20, 30};
  time_stamp_t edge_end_times[]           = {11, 21, 31};
  vertex_t start[]                        = {0};
  time_stamp_t start_vertex_start_times[] = {0};
  time_stamp_t start_vertex_end_times[]   = {100};
  size_t start_vertex_label_offsets[]     = {0, 1};
  int fan_out[]                           = {1};

  expected_temporal_sample_edge_t expected_edges[] = {
    {0, 1, 10, 0},
  };

  return generic_neighbor_sample_expected_edges_test(
    handle,
    src,
    dst,
    weight,
    edge_ids,
    edge_types,
    edge_start_times,
    edge_end_times,
    num_edges,
    start,
    start_vertex_start_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    fan_out,
    fan_out_size,
    MONOTONICALLY_INCREASING,
    CUGRAPH_NEIGHBOR_SELECTION_FIRST,
    expected_edges,
    sizeof(expected_edges) / sizeof(expected_edges[0]));
}

int test_mg_neighbor_sample_last_single_hop(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t fan_out_size = 1;
  size_t num_starts   = 1;

  vertex_t src[]                          = {0, 0, 0};
  vertex_t dst[]                          = {1, 2, 3};
  edge_t edge_ids[]                       = {0, 1, 2};
  weight_t weight[]                       = {0.1, 0.2, 0.3};
  int32_t edge_types[]                    = {0, 1, 2};
  time_stamp_t edge_start_times[]         = {10, 20, 30};
  time_stamp_t edge_end_times[]           = {11, 21, 31};
  vertex_t start[]                        = {0};
  time_stamp_t start_vertex_start_times[] = {0};
  time_stamp_t start_vertex_end_times[]   = {100};
  size_t start_vertex_label_offsets[]     = {0, 1};
  int fan_out[]                           = {1};

  expected_temporal_sample_edge_t expected_edges[] = {
    {0, 3, 30, 0},
  };

  return generic_neighbor_sample_expected_edges_test(
    handle,
    src,
    dst,
    weight,
    edge_ids,
    edge_types,
    edge_start_times,
    edge_end_times,
    num_edges,
    start,
    start_vertex_start_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    fan_out,
    fan_out_size,
    MONOTONICALLY_INCREASING,
    CUGRAPH_NEIGHBOR_SELECTION_LAST,
    expected_edges,
    sizeof(expected_edges) / sizeof(expected_edges[0]));
}

int test_mg_neighbor_sample_fixed_window_multihop(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t fan_out_size = 2;
  size_t num_starts   = 1;

  // FIXED_WINDOW keeps seed window [10, 100] at hop 1, so both 1->2 (t=30) and 1->3 (t=80) remain
  // eligible.
  vertex_t src[]                          = {0, 1, 1};
  vertex_t dst[]                          = {1, 2, 3};
  edge_t edge_ids[]                       = {0, 1, 2};
  weight_t weight[]                       = {0.1, 0.2, 0.3};
  int32_t edge_types[]                    = {0, 0, 0};
  time_stamp_t edge_start_times[]         = {50, 30, 80};
  time_stamp_t edge_end_times[]           = {51, 31, 81};
  vertex_t start[]                        = {0};
  time_stamp_t start_vertex_start_times[] = {10};
  time_stamp_t start_vertex_end_times[]   = {100};
  size_t start_vertex_label_offsets[]     = {0, 1};
  int fan_out[]                           = {-1, -1};

  expected_temporal_sample_edge_t expected_edges[] = {
    {0, 1, 50, 0},
    {1, 2, 30, 1},
    {1, 3, 80, 1},
  };

  return generic_neighbor_sample_expected_edges_test(
    handle,
    src,
    dst,
    weight,
    edge_ids,
    edge_types,
    edge_start_times,
    edge_end_times,
    num_edges,
    start,
    start_vertex_start_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    fan_out,
    fan_out_size,
    FIXED_WINDOW,
    CUGRAPH_NEIGHBOR_SELECTION_RANDOM,
    expected_edges,
    sizeof(expected_edges) / sizeof(expected_edges[0]));
}

int test_mg_neighbor_sample_fixed_window_first(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t fan_out_size = 2;
  size_t num_starts   = 1;

  vertex_t src[]                          = {0, 1, 1};
  vertex_t dst[]                          = {1, 2, 3};
  edge_t edge_ids[]                       = {0, 1, 2};
  weight_t weight[]                       = {0.1, 0.2, 0.3};
  int32_t edge_types[]                    = {0, 0, 0};
  time_stamp_t edge_start_times[]         = {50, 30, 80};
  time_stamp_t edge_end_times[]           = {51, 31, 81};
  vertex_t start[]                        = {0};
  time_stamp_t start_vertex_start_times[] = {10};
  time_stamp_t start_vertex_end_times[]   = {100};
  size_t start_vertex_label_offsets[]     = {0, 1};
  int fan_out[]                           = {-1, 1};

  expected_temporal_sample_edge_t expected_edges[] = {
    {0, 1, 50, 0},
    {1, 2, 30, 1},
  };

  return generic_neighbor_sample_expected_edges_test(
    handle,
    src,
    dst,
    weight,
    edge_ids,
    edge_types,
    edge_start_times,
    edge_end_times,
    num_edges,
    start,
    start_vertex_start_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    fan_out,
    fan_out_size,
    FIXED_WINDOW,
    CUGRAPH_NEIGHBOR_SELECTION_FIRST,
    expected_edges,
    sizeof(expected_edges) / sizeof(expected_edges[0]));
}

int test_mg_neighbor_sample_fixed_window_last(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 3;
  size_t fan_out_size = 2;
  size_t num_starts   = 1;

  vertex_t src[]                          = {0, 1, 1};
  vertex_t dst[]                          = {1, 2, 3};
  edge_t edge_ids[]                       = {0, 1, 2};
  weight_t weight[]                       = {0.1, 0.2, 0.3};
  int32_t edge_types[]                    = {0, 0, 0};
  time_stamp_t edge_start_times[]         = {50, 30, 80};
  time_stamp_t edge_end_times[]           = {51, 31, 81};
  vertex_t start[]                        = {0};
  time_stamp_t start_vertex_start_times[] = {10};
  time_stamp_t start_vertex_end_times[]   = {100};
  size_t start_vertex_label_offsets[]     = {0, 1};
  int fan_out[]                           = {-1, 1};

  expected_temporal_sample_edge_t expected_edges[] = {
    {0, 1, 50, 0},
    {1, 3, 80, 1},
  };

  return generic_neighbor_sample_expected_edges_test(
    handle,
    src,
    dst,
    weight,
    edge_ids,
    edge_types,
    edge_start_times,
    edge_end_times,
    num_edges,
    start,
    start_vertex_start_times,
    start_vertex_end_times,
    start_vertex_label_offsets,
    num_starts,
    fan_out,
    fan_out_size,
    FIXED_WINDOW,
    CUGRAPH_NEIGHBOR_SELECTION_LAST,
    expected_edges,
    sizeof(expected_edges) / sizeof(expected_edges[0]));
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample, handle);
  result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_time_window, handle);
  result |= RUN_MG_TEST(test_uniform_temporal_neighbor_from_alex, handle);
  result |= RUN_MG_TEST(test_mg_neighbor_sample_first_single_hop, handle);
  result |= RUN_MG_TEST(test_mg_neighbor_sample_last_single_hop, handle);
  result |= RUN_MG_TEST(test_mg_neighbor_sample_fixed_window_multihop, handle);
  result |= RUN_MG_TEST(test_mg_neighbor_sample_fixed_window_first, handle);
  result |= RUN_MG_TEST(test_mg_neighbor_sample_fixed_window_last, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_dedupe_sources, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_unique_sources, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_carry_over_sources, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
