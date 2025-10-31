/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <stdbool.h>
#include <stdlib.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

cugraph_data_type_id_t vertex_tid    = INT32;
cugraph_data_type_id_t edge_tid      = INT32;
cugraph_data_type_id_t weight_tid    = FLOAT32;
cugraph_data_type_id_t edge_id_tid   = INT32;
cugraph_data_type_id_t edge_type_tid = INT32;

int generic_uniform_neighbor_sample_test(const cugraph_resource_handle_t* handle,
                                         vertex_t* h_src,
                                         vertex_t* h_dst,
                                         weight_t* h_wgt,
                                         edge_t* h_edge_ids,
                                         int32_t* h_edge_types,
                                         size_t num_vertices,
                                         size_t num_edges,
                                         vertex_t* h_start,
                                         size_t* h_start_label_offsets,
                                         size_t num_start_vertices,
                                         size_t num_start_label_offsets,
                                         int* fan_out,
                                         size_t fan_out_size,
                                         bool_t with_replacement,
                                         bool_t return_hops,
                                         cugraph_prior_sources_behavior_t prior_sources_behavior,
                                         bool_t dedupe_sources,
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
                                  INT32,
                                  NULL,
                                  NULL,
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

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_start_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_start_label_offsets, SIZE_T, &d_start_label_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_label_offsets create failed.");

  d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_label_offsets_view, (byte_t*)h_start_label_offsets, &ret_error);

  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_label_offsets copy_from_host failed.");

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

  ret_code = cugraph_homogeneous_uniform_neighbor_sample(handle,
                                                         rng_state,
                                                         graph,
                                                         d_start_view,
                                                         d_start_label_offsets_view,
                                                         h_fan_out_view,
                                                         sampling_options,
                                                         FALSE,
                                                         &result,
                                                         &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  test_ret_value = validate_sample_result(handle,
                                          result,
                                          h_src,
                                          h_dst,
                                          h_wgt,
                                          h_edge_ids,
                                          h_edge_types,
                                          NULL,
                                          NULL,
                                          num_vertices,
                                          num_edges,
                                          h_start,
                                          num_start_vertices,
                                          h_start_label_offsets,
                                          num_start_label_offsets,
                                          fan_out,
                                          fan_out_size,
                                          sampling_options,
                                          FALSE);
  TEST_ASSERT(test_ret_value, test_ret_value == 0, "validate_sample_result failed.");

  cugraph_sampling_options_free(sampling_options);
  cugraph_rng_state_free(rng_state);
  cugraph_sample_result_free(result);
  cugraph_graph_free(graph);
  cugraph_type_erased_device_array_view_free(d_start_view);
  cugraph_type_erased_device_array_view_free(d_start_label_offsets_view);
  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_type_erased_device_array_free(d_start);
  cugraph_type_erased_device_array_free(d_start_label_offsets);

  return test_ret_value;
}

int create_test_graph_with_edge_ids(const cugraph_resource_handle_t* p_handle,
                                    vertex_t* h_src,
                                    vertex_t* h_dst,
                                    edge_t* h_ids,
                                    size_t num_edges,
                                    bool_t store_transposed,
                                    bool_t renumber,
                                    bool_t is_symmetric,
                                    cugraph_graph_t** p_graph,
                                    cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  cugraph_graph_properties_t properties;

  properties.is_symmetric  = is_symmetric;
  properties.is_multigraph = FALSE;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* ids;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* ids_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, edge_tid, &ids, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "ids create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  ids_view = cugraph_type_erased_device_array_view(ids);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, ids_view, (byte_t*)h_ids, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_as_type(ids, weight_tid, &wgt_view, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt cast from ids failed.");

  ret_code = cugraph_graph_create_with_times_sg(p_handle,
                                                &properties,
                                                NULL,
                                                src_view,
                                                dst_view,
                                                wgt_view,
                                                NULL,
                                                NULL,
                                                NULL,
                                                NULL,
                                                store_transposed,
                                                renumber,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                FALSE,
                                                p_graph,
                                                ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(ids_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(ids);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

int test_uniform_neighbor_sample_clean(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges               = 9;
  size_t num_vertices            = 6;
  size_t fan_out_size            = 3;
  size_t num_starts              = 2;
  size_t num_start_label_offsets = 3;

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
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              weight,
                                              edge_ids,
                                              edge_types,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_label_offsets,
                                              num_starts,
                                              num_start_label_offsets,
                                              fan_out,
                                              fan_out_size,
                                              with_replacement,
                                              return_hops,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              renumber_results);
}

int test_uniform_neighbor_sample_dedupe_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges               = 9;
  size_t num_vertices            = 6;
  size_t fan_out_size            = 3;
  size_t num_starts              = 2;
  size_t num_start_label_offsets = 3;

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
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              weight,
                                              edge_ids,
                                              edge_types,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_label_offsets,
                                              num_starts,
                                              num_start_label_offsets,
                                              fan_out,
                                              fan_out_size,
                                              with_replacement,
                                              return_hops,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              renumber_results);
}

int test_uniform_neighbor_sample_unique_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges               = 9;
  size_t num_vertices            = 6;
  size_t fan_out_size            = 3;
  size_t num_starts              = 2;
  size_t num_start_label_offsets = 3;

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
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              weight,
                                              edge_ids,
                                              edge_types,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_label_offsets,
                                              num_starts,
                                              num_start_label_offsets,
                                              fan_out,
                                              fan_out_size,
                                              with_replacement,
                                              return_hops,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              renumber_results);
}

int test_uniform_neighbor_sample_carry_over_sources(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges               = 9;
  size_t num_vertices            = 6;
  size_t fan_out_size            = 3;
  size_t num_starts              = 2;
  size_t num_start_label_offsets = 3;

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
  bool_t renumber_results                                 = FALSE;

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              weight,
                                              edge_ids,
                                              edge_types,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_label_offsets,
                                              num_starts,
                                              num_start_label_offsets,
                                              fan_out,
                                              fan_out_size,
                                              with_replacement,
                                              return_hops,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              renumber_results);
}

int test_uniform_neighbor_sample_renumber_results(const cugraph_resource_handle_t* handle)
{
  cugraph_data_type_id_t vertex_tid    = INT32;
  cugraph_data_type_id_t edge_tid      = INT32;
  cugraph_data_type_id_t weight_tid    = FLOAT32;
  cugraph_data_type_id_t edge_id_tid   = INT32;
  cugraph_data_type_id_t edge_type_tid = INT32;

  size_t num_edges               = 9;
  size_t num_vertices            = 6;
  size_t fan_out_size            = 3;
  size_t num_starts              = 2;
  size_t num_start_label_offsets = 3;

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
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = TRUE;

  return generic_uniform_neighbor_sample_test(handle,
                                              src,
                                              dst,
                                              weight,
                                              edge_ids,
                                              edge_types,
                                              num_vertices,
                                              num_edges,
                                              start,
                                              start_label_offsets,
                                              num_starts,
                                              num_start_label_offsets,
                                              fan_out,
                                              fan_out_size,
                                              with_replacement,
                                              return_hops,
                                              prior_sources_behavior,
                                              dedupe_sources,
                                              renumber_results);
}

int main(int argc, char** argv)
{
  cugraph_resource_handle_t* handle = NULL;

  handle = cugraph_create_resource_handle(NULL);

  int result = 0;
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_clean, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_dedupe_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_unique_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_carry_over_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_renumber_results, handle);

  cugraph_free_resource_handle(handle);

  return result;
}
