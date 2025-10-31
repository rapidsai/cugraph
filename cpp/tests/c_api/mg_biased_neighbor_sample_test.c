/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "mg_test_utils.h" /* RUN_MG_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <stdbool.h>
#include <unistd.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

cugraph_data_type_id_t vertex_tid    = INT32;
cugraph_data_type_id_t edge_tid      = INT32;
cugraph_data_type_id_t weight_tid    = FLOAT32;
cugraph_data_type_id_t edge_id_tid   = INT32;
cugraph_data_type_id_t edge_type_tid = INT32;

int generic_biased_neighbor_sample_test(const cugraph_resource_handle_t* handle,
                                        vertex_t* h_src,
                                        vertex_t* h_dst,
                                        weight_t* h_wgt,
                                        edge_t* h_edge_ids,
                                        int32_t* h_edge_types,
                                        int32_t* h_edge_start_times,
                                        int32_t* h_edge_end_times,
                                        size_t num_vertices,
                                        size_t num_edges,
                                        vertex_t* h_start,
                                        size_t num_start_vertices,
                                        size_t* h_start_label_offsets,
                                        size_t num_start_label_offsets,
                                        int* fan_out,
                                        size_t fan_out_size,
                                        bool_t with_replacement,
                                        bool_t return_hops,
                                        cugraph_prior_sources_behavior_t prior_sources_behavior,
                                        bool_t dedupe_sources,
                                        bool_t is_multi_graph,
                                        bool_t renumber_results)
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
                                      INT32,
                                      h_edge_start_times,
                                      h_edge_end_times,
                                      num_edges,
                                      FALSE,
                                      TRUE,
                                      FALSE,
                                      is_multi_graph,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_t* d_start                         = NULL;
    cugraph_type_erased_device_array_view_t* d_start_view               = NULL;
    cugraph_type_erased_device_array_t* d_start_label_offsets           = NULL;
    cugraph_type_erased_device_array_view_t* d_start_label_offsets_view = NULL;
    cugraph_type_erased_host_array_view_t* h_fan_out_view               = NULL;

    if (rank > 0) num_start_vertices = 0;

    h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices, INT32, &d_start, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

    d_start_view = cugraph_type_erased_device_array_view(d_start);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_view, (byte_t*)h_start, &ret_error);

    if (h_start_label_offsets != NULL) {
      ret_code = cugraph_type_erased_device_array_create(
        handle, num_start_label_offsets, SIZE_T, &d_start_label_offsets, &ret_error);
      TEST_ASSERT(
        test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_label_offsets create failed.");

      d_start_label_offsets_view = cugraph_type_erased_device_array_view(d_start_label_offsets);

      ret_code = cugraph_type_erased_device_array_view_copy_from_host(
        handle, d_start_label_offsets_view, (byte_t*)h_start_label_offsets, &ret_error);
      TEST_ASSERT(
        test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_label_offsets copy_from_host failed.");
    }

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

    ret_code = cugraph_homogeneous_biased_neighbor_sample(handle,
                                                          rng_state,
                                                          graph,
                                                          NULL,
                                                          d_start_view,
                                                          d_start_label_offsets_view,
                                                          h_fan_out_view,
                                                          sampling_options,
                                                          FALSE,
                                                          &result,
                                                          &ret_error);

    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "homogeneous_biased_neighbor_sample failed.");

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
                                               h_start_label_offsets,
                                               num_start_label_offsets,
                                               fan_out,
                                               fan_out_size,
                                               sampling_options,
                                               FALSE);
    TEST_ASSERT(test_ret_value, test_ret_value == 0, "validate_sample_result failed.");

    cugraph_sampling_options_free(sampling_options);
    cugraph_sample_result_free(result);
    cugraph_rng_state_free(rng_state);
    cugraph_type_erased_device_array_view_free(d_start_view);
    cugraph_type_erased_device_array_view_free(d_start_label_offsets_view);
    cugraph_type_erased_host_array_view_free(h_fan_out_view);
    cugraph_type_erased_device_array_free(d_start);
    cugraph_type_erased_device_array_free(d_start_label_offsets);
  }

  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_biased_neighbor_sample(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7};
  vertex_t start[] = {2, 2};
  int fan_out[]    = {1, 2};

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;

  return generic_biased_neighbor_sample_test(handle,
                                             src,
                                             dst,
                                             wgt,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             num_starts,
                                             NULL,
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

int test_biased_neighbor_from_alex(const cugraph_resource_handle_t* handle)
{
  size_t num_edges                = 12;
  size_t num_vertices             = 5;
  size_t fan_out_size             = 2;
  size_t num_starts               = 2;
  size_t start_label_offsets_size = 3;

  vertex_t src[]               = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
  vertex_t dst[]               = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
  edge_t idx[]                 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t typ[]                = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
  weight_t wgt[]               = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
  vertex_t start[]             = {0, 4};
  size_t start_label_offsets[] = {0, 1, 2};
  int fan_out[]                = {2, 2};

  bool_t store_transposed = FALSE;

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources                                   = FALSE;
  bool_t renumber_results                                 = FALSE;
  cugraph_compression_type_t compression                  = COO;
  bool_t compress_per_hop                                 = FALSE;

  return generic_biased_neighbor_sample_test(handle,
                                             src,
                                             dst,
                                             wgt,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             num_starts,
                                             start_label_offsets,
                                             start_label_offsets_size,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             TRUE,
                                             FALSE);
}

int test_biased_neighbor_sample_alex_bug(const cugraph_resource_handle_t* handle)
{
  size_t num_edges                = 156;
  size_t num_vertices             = 34;
  size_t fan_out_size             = 2;
  size_t num_starts               = 4;
  size_t start_label_offsets_size = 4;

  vertex_t src[] = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
                    17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
                    16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
                    32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
                    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
                    8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
                    24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  vertex_t dst[] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
                    1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
                    6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
                    23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
                    3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
                    21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
                    32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
                    25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  weight_t wgt[] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  edge_t edge_ids[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
    18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
    54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155};

  vertex_t start[]             = {0, 1, 2, 5};
  size_t start_label_offsets[] = {0, 2, 3, 4};
  int fan_out[]                = {2, 3};

  size_t expected_size[] = {3, 2, 1, 1, 1, 1, 1, 1};

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
  bool_t dedupe_sources                                   = TRUE;
  bool_t renumber_results                                 = FALSE;
  cugraph_compression_type_t compression                  = COO;
  bool_t compress_per_hop                                 = FALSE;

  return generic_biased_neighbor_sample_test(handle,
                                             src,
                                             dst,
                                             wgt,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             num_starts,
                                             start_label_offsets,
                                             start_label_offsets_size,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             FALSE,
                                             renumber_results);
}

int test_biased_neighbor_sample_sort_by_hop(const cugraph_resource_handle_t* handle)
{
  size_t num_edges                = 156;
  size_t num_vertices             = 34;
  size_t fan_out_size             = 2;
  size_t num_starts               = 4;
  size_t start_label_offsets_size = 4;

  vertex_t src[] = {1,  2,  3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13,
                    17, 19, 21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16,
                    16, 30, 32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29,
                    32, 33, 25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,
                    1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,  6,  8,
                    8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23, 23, 23,
                    24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32};
  vertex_t dst[] = {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,
                    1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  4,  4,  5,  5,  5,
                    6,  8,  8,  8,  9,  13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22, 23, 23, 23,
                    23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 1,  2,
                    3,  4,  5,  6,  7,  8,  10, 11, 12, 13, 17, 19, 21, 31, 2,  3,  7,  13, 17, 19,
                    21, 30, 3,  7,  8,  9,  13, 27, 28, 32, 7,  12, 13, 6,  10, 6,  10, 16, 16, 30,
                    32, 33, 33, 33, 32, 33, 32, 33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33,
                    25, 27, 31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33};
  weight_t wgt[] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  edge_t edge_ids[] = {
    0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,
    18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
    36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
    54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
    72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
    90,  91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107,
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155};

  vertex_t start[]             = {0, 1, 2, 5};
  size_t start_label_offsets[] = {0, 2, 3, 4};
  int fan_out[]                = {2, 3};

  size_t expected_size[] = {3, 2, 1, 1, 1, 1, 1, 1};

  bool_t with_replacement                                 = FALSE;
  bool_t return_hops                                      = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
  bool_t dedupe_sources                                   = TRUE;
  bool_t renumber_results                                 = FALSE;
  cugraph_compression_type_t compression                  = COO;
  bool_t compress_per_hop                                 = FALSE;

  return generic_biased_neighbor_sample_test(handle,
                                             src,
                                             dst,
                                             wgt,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             num_vertices,
                                             num_edges,
                                             start,
                                             num_starts,
                                             start_label_offsets,
                                             start_label_offsets_size,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             FALSE,
                                             renumber_results);
}

int test_biased_neighbor_sample_dedupe_sources(const cugraph_resource_handle_t* handle)
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

  return generic_biased_neighbor_sample_test(handle,
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
                                             num_starts,
                                             start_label_offsets,
                                             num_start_label_offsets,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             FALSE,
                                             FALSE);
}

int test_biased_neighbor_sample_unique_sources(const cugraph_resource_handle_t* handle)
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

  return generic_biased_neighbor_sample_test(handle,
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
                                             num_starts,
                                             start_label_offsets,
                                             num_start_label_offsets,
                                             fan_out,
                                             fan_out_size,
                                             with_replacement,
                                             return_hops,
                                             prior_sources_behavior,
                                             dedupe_sources,
                                             FALSE,
                                             FALSE);
}

int test_biased_neighbor_sample_carry_over_sources(const cugraph_resource_handle_t* handle)
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

  return generic_biased_neighbor_sample_test(handle,
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
                                             num_starts,
                                             start_label_offsets,
                                             num_start_label_offsets,
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
  result |= RUN_MG_TEST(test_biased_neighbor_sample, handle);
  result |= RUN_MG_TEST(test_biased_neighbor_from_alex, handle);
  // result |= RUN_MG_TEST(test_biased_neighbor_sample_alex_bug, handle);
  result |= RUN_MG_TEST(test_biased_neighbor_sample_sort_by_hop, handle);
  // result |= RUN_MG_TEST(test_biased_neighbor_sample_dedupe_sources, handle);
  // result |= RUN_MG_TEST(test_biased_neighbor_sample_unique_sources, handle);
  // result |= RUN_MG_TEST(test_biased_neighbor_sample_carry_over_sources, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
