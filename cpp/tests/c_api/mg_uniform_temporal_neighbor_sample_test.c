/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdbool.h>
#include <unistd.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;
typedef int32_t edge_time_t;

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
  edge_time_t* h_edge_start_times,
  edge_time_t* h_edge_end_times,
  size_t num_vertices,
  size_t num_edges,
  vertex_t* h_start,
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
  bool_t is_multigraph)
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

  cugraph_sampling_options_free(sampling_options);
  cugraph_sample_result_free(result);
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

  vertex_t src[]            = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]            = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_time_t start_times[] = {0, 1, 2, 3, 4, 5, 6, 7};
  edge_time_t end_times[]   = {1, 2, 3, 4, 5, 6, 7, 8};
  edge_t idx[]              = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[]          = {2, 2};
  int fan_out[]             = {1, 2};

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
                                                       num_starts,
                                                       0,
                                                       fan_out,
                                                       fan_out_size,
                                                       with_replacement,
                                                       return_hops,
                                                       prior_sources_behavior,
                                                       dedupe_sources,
                                                       FALSE,
                                                       FALSE);
}

int test_uniform_temporal_neighbor_from_alex(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 12;
  size_t num_vertices = 5;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]                 = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
  vertex_t dst[]                 = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
  edge_t edge_ids[]              = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t edge_types[]           = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
  weight_t weights[]             = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
  edge_time_t edge_start_times[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  edge_time_t edge_end_times[]   = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  vertex_t start[]               = {0, 4};
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
                                                       TRUE);
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
                                                       FALSE);
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
                                                       FALSE);
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
                                                       FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample, handle);
  result |= RUN_MG_TEST(test_uniform_temporal_neighbor_from_alex, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_dedupe_sources, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_unique_sources, handle);
  // result |= RUN_MG_TEST(test_uniform_temporal_neighbor_sample_carry_over_sources, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
