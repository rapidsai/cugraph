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

#include <math.h>
#include <unistd.h>
#include <stdbool.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

data_type_id_t vertex_tid    = INT32;
data_type_id_t edge_tid      = INT32;
data_type_id_t weight_tid    = FLOAT32;
data_type_id_t edge_id_tid   = INT32;
data_type_id_t edge_type_tid = INT32;

int generic_uniform_neighbor_sample_test(const cugraph_resource_handle_t* handle,
                                         vertex_t *h_src,
                                         vertex_t *h_dst,
                                         weight_t *h_wgt,
                                         edge_t *h_edge_ids,
                                         int32_t *h_edge_types,
                                         size_t num_vertices,
                                         size_t num_edges,
                                         vertex_t *h_start,
                                         int *h_start_labels,
                                         size_t num_start_vertices,
                                         int *fan_out,
                                         size_t fan_out_size,
                                         bool_t with_replacement,
                                         bool_t return_hops,
                                         cugraph_prior_sources_behavior_t prior_sources_behavior,
                                         bool_t dedupe_sources)
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
                                      num_edges,
                                      FALSE,
                                      TRUE,
                                      FALSE,
                                      FALSE,
                                      &graph,
                                      &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_device_array_t* d_start_labels           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_labels_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  if (rank > 0) num_start_vertices = 0;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_start_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  if (h_start_labels != NULL) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_start_vertices, INT32, &d_start_labels, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

    d_start_labels_view = cugraph_type_erased_device_array_view(d_start_labels);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, d_start_labels_view, (byte_t*)h_start_labels, &ret_error);

    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels copy_from_host failed.");
  }

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t *sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);

  ret_code = cugraph_uniform_neighbor_sample(handle,
                                             graph,
                                             d_start_view,
                                             d_start_labels_view,
                                             NULL,
                                             NULL,
                                             h_fan_out_view,
                                             rng_state,
                                             sampling_options,
                                             FALSE,
                                             &result,
                                             &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed")
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_sampling_options_free(sampling_options);

  cugraph_type_erased_device_array_view_t* result_srcs;
  cugraph_type_erased_device_array_view_t* result_dsts;
  cugraph_type_erased_device_array_view_t* result_edge_id;
  cugraph_type_erased_device_array_view_t* result_weights;
  cugraph_type_erased_device_array_view_t* result_edge_types;
  cugraph_type_erased_device_array_view_t* result_hops;
  cugraph_type_erased_device_array_view_t* result_offsets = NULL;
  cugraph_type_erased_device_array_view_t* result_labels = NULL;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_edge_types = cugraph_sample_result_get_edge_type(result);
  result_hops       = cugraph_sample_result_get_hop(result);
  result_hops       = cugraph_sample_result_get_hop(result);

  size_t result_offsets_size = 2;

  if (d_start_labels != NULL) {
    result_offsets    = cugraph_sample_result_get_offsets(result);
    result_labels     = cugraph_sample_result_get_start_labels(result);
    result_offsets_size = 1 + cugraph_test_scalar_reduce(handle,
                                                         cugraph_type_erased_device_array_view_size(result_offsets) - 1);

  }

  size_t result_size = cugraph_test_device_gatherv_size(handle, result_srcs);

  vertex_t h_result_srcs[result_size];
  vertex_t h_result_dsts[result_size];
  edge_t h_result_edge_id[result_size];
  weight_t h_result_weight[result_size];
  int32_t h_result_edge_types[result_size];
  int32_t h_result_hops[result_size];
  size_t h_result_offsets[result_offsets_size];
  int h_result_labels[result_offsets_size-1];

  if (result_offsets_size == 2) {
    h_result_offsets[0] = 0;
    h_result_offsets[1] = result_size;
  }

  ret_code = cugraph_test_device_gatherv_fill(handle, result_srcs, h_result_srcs);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");

  ret_code = cugraph_test_device_gatherv_fill(handle, result_dsts, h_result_dsts);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");

  if (h_edge_ids != NULL) {
    ret_code = cugraph_test_device_gatherv_fill(handle, result_edge_id, h_result_edge_id);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");
  }

  if (h_wgt != NULL) {
    ret_code = cugraph_test_device_gatherv_fill(handle, result_weights, h_result_weight);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");
  }

  if (h_edge_types != NULL) {
    ret_code = cugraph_test_device_gatherv_fill(handle, result_edge_types, h_result_edge_types);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");
  }

  if (return_hops) {
    ret_code = cugraph_test_device_gatherv_fill(handle, result_hops, h_result_hops);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");
  }

  if (d_start_labels != NULL) {
    size_t sz = cugraph_type_erased_device_array_view_size(result_offsets);

    ret_code = cugraph_test_device_gatherv_fill(handle, result_labels, h_result_labels);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "gatherv_fill failed.");

    size_t tmp_result_offsets[sz];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)tmp_result_offsets, result_offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    // convert to size
    for (size_t i = 1 ; i < sz ; ++i) {
      tmp_result_offsets[i-1] = tmp_result_offsets[i] - tmp_result_offsets[i-1];
    }
    
    cugraph_test_host_gatherv_fill(handle, tmp_result_offsets, sz-1, SIZE_T, h_result_offsets + 1);

    h_result_offsets[0] = 0;
    for (size_t i = 1 ; i < result_offsets_size ; ++i) {
      h_result_offsets[i] += h_result_offsets[i-1];
    }
  }

  //  First, check that all edges are actually part of the graph
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
    if (h_wgt != NULL)
      M_w[h_src[i]][h_dst[i]] = h_wgt[i];
    else
      M_w[h_src[i]][h_dst[i]] = 1.0;
 
    if (h_edge_ids != NULL) M_edge_id[h_src[i]][h_dst[i]]   = h_edge_ids[i];
    if (h_edge_types != NULL) M_edge_type[h_src[i]][h_dst[i]] = h_edge_types[i];
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    if (h_wgt != NULL) {
      TEST_ASSERT(test_ret_value,
                  M_w[h_result_srcs[i]][h_result_dsts[i]] == h_result_weight[i],
                  "uniform_neighbor_sample got edge that doesn't exist");
    } else {
      TEST_ASSERT(test_ret_value,
                  M_w[h_result_srcs[i]][h_result_dsts[i]] == 1.0,
                  "uniform_neighbor_sample got edge that doesn't exist");
    }

    if (h_edge_ids != NULL) TEST_ASSERT(test_ret_value,
                                        M_edge_id[h_result_srcs[i]][h_result_dsts[i]] == h_result_edge_id[i],
                                        "uniform_neighbor_sample got edge that doesn't exist");
    if (h_edge_types != NULL) TEST_ASSERT(test_ret_value,
                                          M_edge_type[h_result_srcs[i]][h_result_dsts[i]] == h_result_edge_types[i],
                                          "uniform_neighbor_sample got edge that doesn't exist");
  }

  if ((return_hops) && (d_start_labels != NULL) && (result_offsets_size > 0)) {
    //
    // For the sampling result to make sense, all sources in hop 0 must be in the seeds,
    // all sources in hop 1 must be a result from hop 0, etc.
    //
    vertex_t check_v1[result_size];
    vertex_t check_v2[result_size];
    vertex_t *check_sources = check_v1;
    vertex_t *check_destinations = check_v2;

    size_t degree[num_vertices];
    for (size_t i = 0 ; i < num_vertices ; ++i)
      degree[i] = 0;

    for (size_t i = 0 ; i < num_edges ; ++i) {
      degree[h_src[i]]++;
    }

    for (size_t label_id = 0 ; label_id < (result_offsets_size - 1) ; ++label_id) {
      // Skip any labels we already processed
      bool already_processed = false;
      for (size_t i = 0 ; (i < label_id) && !already_processed ; ++i)
        already_processed = (h_result_labels[label_id] == h_result_labels[i]);

      if (already_processed) continue;

      size_t   sources_size = 0;
      size_t   destinations_size = 0;

      // Fill sources with the input sources
      for (size_t i = 0 ; i < num_start_vertices ; ++i) {
        if (h_start_labels[i] == h_result_labels[label_id]) {
          check_sources[sources_size] = h_start[i];
          ++sources_size;
        }
      }

      for (int hop = 0 ; hop < fan_out_size ; ++hop) {
        if (prior_sources_behavior == CARRY_OVER) {
          destinations_size = sources_size;
          for (size_t i = 0 ; i < sources_size ; ++i) {
            check_destinations[i] = check_sources[i];
          }
        }

        for (size_t current_label_id = label_id ; current_label_id < (result_offsets_size - 1) ; ++current_label_id) {
          if (h_result_labels[current_label_id] == h_result_labels[label_id]) {
            for (size_t i = h_result_offsets[current_label_id]; (i < h_result_offsets[current_label_id+1]) && (test_ret_value == 0) ; ++i) {
              if (h_result_hops[i] == hop) {
                bool found = false;
                for (size_t j = 0 ; (!found) && (j < sources_size) ; ++j) {
                  found = (h_result_srcs[i] == check_sources[j]);
                }

                TEST_ASSERT(test_ret_value, found, "encountered source vertex that was not part of previous frontier");
              }

              if (prior_sources_behavior == CARRY_OVER) {
                // Make sure destination isn't already in the source list
                bool found = false;
                for (size_t j = 0 ; (!found) && (j < destinations_size) ; ++j) {
                  found = (h_result_dsts[i] == check_destinations[j]);
                }

                if (!found) {
                  check_destinations[destinations_size] = h_result_dsts[i];
                  ++destinations_size;
                }
              } else {
                check_destinations[destinations_size] = h_result_dsts[i];
                ++destinations_size;
              }
            }
          }
        }

        vertex_t *tmp = check_sources;
        check_sources = check_destinations;
        check_destinations = tmp;
        sources_size = destinations_size;
        destinations_size = 0;
      }

      if (prior_sources_behavior == EXCLUDE) {
        // Make sure vertex v only appears as source in the first hop after it is encountered
        for (size_t current_label_id = label_id ; current_label_id < (result_offsets_size - 1) ; ++current_label_id) {
          if (h_result_labels[current_label_id] == h_result_labels[label_id]) {
            for (size_t i = h_result_offsets[current_label_id]; (i < h_result_offsets[current_label_id+1]) && (test_ret_value == 0) ; ++i) {
              for (size_t j = i + 1 ; (j < h_result_offsets[current_label_id+1]) && (test_ret_value == 0) ; ++j) {
                if (h_result_srcs[i] == h_result_srcs[j]) {
                  TEST_ASSERT(test_ret_value,
                              h_result_hops[i] == h_result_hops[j],
                              "source vertex should not have been used in diferent hops");
                }
              }
            }
          }
        }
      }

      if (dedupe_sources) {
        // Make sure vertex v only appears as source once for each edge after it appears as destination
        // Externally test this by verifying that vertex v only appears in <= hop size/degree
        for (size_t current_label_id = label_id ; current_label_id < (result_offsets_size - 1) ; ++current_label_id) {
          if (h_result_labels[current_label_id] == h_result_labels[label_id]) {
            for (size_t i = h_result_offsets[current_label_id]; (i < h_result_offsets[current_label_id+1]) && (test_ret_value == 0) ; ++i) {
              if (h_result_hops[i] > 0) {
                size_t num_occurrences = 1;
                for (size_t j = i + 1 ; j < h_result_offsets[current_label_id+1] ; ++j) {
                  if ((h_result_srcs[j] == h_result_srcs[i]) && (h_result_hops[j] == h_result_hops[i]))
                    num_occurrences++;
                }

                if (fan_out[h_result_hops[i]] < 0) {
                  TEST_ASSERT(test_ret_value,
                              num_occurrences <= degree[h_result_srcs[i]],
                              "source vertex used in too many return edges");
                } else {
                  TEST_ASSERT(test_ret_value,
                              num_occurrences < fan_out[h_result_hops[i]],
                              "source vertex used in too many return edges");
                }
              }
            }
          }
        }
      }
    }
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_uniform_neighbor_sample(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t idx[]     = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[] = {2, 2};
  int fan_out[]    = {1, 2};

  bool_t with_replacement = FALSE;
  bool_t return_hops = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, NULL, idx, NULL, num_vertices, num_edges,
                                              start, NULL, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources);
}

int test_uniform_neighbor_from_alex(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 12;
  size_t num_vertices = 5;
  size_t fan_out_size = 2;
  size_t num_starts   = 2;

  vertex_t src[]   = {0, 1, 2, 3, 4, 3, 4, 2, 0, 1, 0, 2};
  vertex_t dst[]   = {1, 2, 4, 2, 3, 4, 1, 1, 2, 3, 4, 4};
  edge_t idx[]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  int32_t typ[]    = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0};
  weight_t wgt[]   = {0.0, 0.1, 0.2, 3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11};
  vertex_t start[] = {0, 4};
  int32_t batch[]  = {0, 1};
  int fan_out[]    = {2, 2};

  bool_t with_replacement = TRUE;
  bool_t store_transposed = FALSE;

  int test_ret_value            = 0;
  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_t* d_label           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_device_array_view_t* d_label_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = create_mg_test_graph_with_properties(
    handle, src, dst, idx, typ, wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_label, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label create failed.");

  d_label_view = cugraph_type_erased_device_array_view(d_label);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_view, (byte_t*)batch, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  d_label_view,
                                                                  NULL,
                                                                  NULL,
                                                                  h_fan_out_view,
                                                                  rng_state,
                                                                  with_replacement,
                                                                  TRUE,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed");
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_src;
  cugraph_type_erased_device_array_view_t* result_dst;
  cugraph_type_erased_device_array_view_t* result_index;
  cugraph_type_erased_device_array_view_t* result_type;
  cugraph_type_erased_device_array_view_t* result_weight;
  cugraph_type_erased_device_array_view_t* result_labels;
  cugraph_type_erased_device_array_view_t* result_hops;

  result_src    = cugraph_sample_result_get_sources(result);
  result_dst    = cugraph_sample_result_get_destinations(result);
  result_index  = cugraph_sample_result_get_edge_id(result);
  result_type   = cugraph_sample_result_get_edge_type(result);
  result_weight = cugraph_sample_result_get_edge_weight(result);
  result_labels = cugraph_sample_result_get_start_labels(result);
  result_hops   = cugraph_sample_result_get_hop(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_src);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_index[result_size];
  int h_type[result_size];
  weight_t h_wgt[result_size];
  int h_labels[result_size];
  int h_hop[result_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_srcs, result_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_dsts, result_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_index, result_index, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_type, result_type, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_wgt, result_weight, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_labels, result_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_hop, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  edge_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[src[i]][dst[i]] = idx[i];

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M[h_srcs[i]][h_dsts[i]] >= 0,
                "uniform_neighbor_sample got edge that doesn't exist");
  }
#endif

  cugraph_sample_result_free(result);

  cugraph_type_erased_host_array_view_free(h_fan_out_view);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_neighbor_sample_alex_bug(const cugraph_resource_handle_t* handle)
{
  size_t num_edges = 156;
  size_t num_vertices = 34;
  size_t fan_out_size = 2;
  size_t num_starts   = 4;
  size_t num_labels   = 3;

  vertex_t src[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2,
                    3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12,
                    13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32,
                    33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27,
                    31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6,
                    8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22,
                    23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29,
                    29, 30, 30, 31, 31, 32};
  vertex_t dst[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,4,4,5,5,5,6,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32,1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,2,3,7,13,17,19,21,30,3,7,8,9,13,27,28,32,7,12,13,6,10,6,10,16,16,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33};
  weight_t wgt[] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};

  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                          50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                          60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                          70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                          80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                          100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                          120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                          140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                          150, 151, 152, 153, 154, 155};

  vertex_t start[]     = {0, 1, 2, 5};
  int32_t  start_labels[] = { 0, 0, 1, 2 };
  int32_t  label_list[] = { 0, 1, 2 };
  int32_t  label_to_output_comm_rank[] = { 0, 0, 1 };
  int fan_out[]        = {2, 3};

  size_t expected_size[] = { 3, 2, 1, 1, 1, 1, 1, 1 };

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  ret_code = create_mg_test_graph_with_properties(handle,
                                                  src,
                                                  dst,
                                                  edge_ids,
                                                  NULL,
                                                  wgt,
                                                  num_edges,
                                                  FALSE,
                                                  TRUE,
                                                  &graph,
                                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_device_array_t* d_start_labels           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_labels_view = NULL;
  cugraph_type_erased_device_array_t* d_label_list           = NULL;
  cugraph_type_erased_device_array_view_t* d_label_list_view = NULL;
  cugraph_type_erased_device_array_t* d_label_to_output_comm_rank           = NULL;
  cugraph_type_erased_device_array_view_t* d_label_to_output_comm_rank_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  if (rank > 0) {
    num_starts = 0;
  }

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

  d_start_labels_view = cugraph_type_erased_device_array_view(d_start_labels);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_labels_view, (byte_t*)start_labels, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_labels, INT32, &d_label_list, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label_list create failed.");

  d_label_list_view = cugraph_type_erased_device_array_view(d_label_list);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_list_view, (byte_t*)label_list, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "label_list copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_labels, INT32, &d_label_to_output_comm_rank, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label_to_output_comm_rank create failed.");

  d_label_to_output_comm_rank_view = cugraph_type_erased_device_array_view(d_label_to_output_comm_rank);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_to_output_comm_rank_view, (byte_t*)label_to_output_comm_rank, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "label_to_output_comm_rank copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  d_start_labels_view,
                                                                  d_label_list_view,
                                                                  d_label_to_output_comm_rank_view,
                                                                  h_fan_out_view,
                                                                  rng_state,
                                                                  FALSE,
                                                                  TRUE,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed")
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_srcs = NULL;
  cugraph_type_erased_device_array_view_t* result_dsts = NULL;
  cugraph_type_erased_device_array_view_t* result_edge_id = NULL;
  cugraph_type_erased_device_array_view_t* result_weights = NULL;
  cugraph_type_erased_device_array_view_t* result_hops = NULL;
  cugraph_type_erased_device_array_view_t* result_offsets = NULL;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_hops       = cugraph_sample_result_get_hop(result);
  result_offsets    = cugraph_sample_result_get_offsets(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);
  size_t result_offsets_size = cugraph_type_erased_device_array_view_size(result_offsets);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_edge_id[result_size];
  weight_t h_weight[result_size];
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
    handle, (byte_t*)h_hops, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_offsets, result_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M_w[num_vertices][num_vertices];
  edge_t M_edge_id[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M_w[i][j]         = 0.0;
      M_edge_id[i][j]   = -1;
    }

  for (int i = 0; i < num_edges; ++i) {
    M_w[src[i]][dst[i]]         = wgt[i];
    M_edge_id[src[i]][dst[i]]   = edge_ids[i];
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M_w[h_srcs[i]][h_dsts[i]] == h_weight[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  TEST_ASSERT(test_ret_value,
              result_offsets_size == expected_size[rank],
              "incorrect number of results");
              

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
}

int test_uniform_neighbor_sample_sort_by_hop(const cugraph_resource_handle_t* handle)
{
  size_t num_edges = 156;
  size_t num_vertices = 34;
  size_t fan_out_size = 2;
  size_t num_starts   = 4;
  size_t num_labels   = 3;

  vertex_t src[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2,
                    3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12,
                    13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32,
                    33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27,
                    31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6,
                    8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22,
                    23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29,
                    29, 30, 30, 31, 31, 32};
  vertex_t dst[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,4,4,5,5,5,6,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32,1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,2,3,7,13,17,19,21,30,3,7,8,9,13,27,28,32,7,12,13,6,10,6,10,16,16,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33};
  weight_t wgt[] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};

  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                          20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                          30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                          40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                          50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                          60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                          70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                          80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                          90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                          100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                          120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
                          130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                          140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                          150, 151, 152, 153, 154, 155};

  vertex_t start[]     = {0, 1, 2, 5};
  int32_t  start_labels[] = { 0, 0, 1, 2 };
  int32_t  label_list[] = { 0, 1, 2 };
  int32_t  label_to_output_comm_rank[] = { 0, 0, 1 };
  int fan_out[]        = {2, 3};

  size_t expected_size[] = { 3, 2, 1, 1, 1, 1, 1, 1 };

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

  ret_code = create_mg_test_graph_with_properties(handle,
                                                  src,
                                                  dst,
                                                  edge_ids,
                                                  NULL,
                                                  wgt,
                                                  num_edges,
                                                  FALSE,
                                                  TRUE,
                                                  &graph,
                                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;
  cugraph_type_erased_device_array_t* d_start_labels           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_labels_view = NULL;
  cugraph_type_erased_device_array_t* d_label_list           = NULL;
  cugraph_type_erased_device_array_view_t* d_label_list_view = NULL;
  cugraph_type_erased_device_array_t* d_label_to_output_comm_rank           = NULL;
  cugraph_type_erased_device_array_view_t* d_label_to_output_comm_rank_view = NULL;
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  if (rank > 0) {
    num_starts = 0;
  }

  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

  d_start_labels_view = cugraph_type_erased_device_array_view(d_start_labels);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_labels_view, (byte_t*)start_labels, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_labels, INT32, &d_label_list, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label_list create failed.");

  d_label_list_view = cugraph_type_erased_device_array_view(d_label_list);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_list_view, (byte_t*)label_list, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "label_list copy_from_host failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_labels, INT32, &d_label_to_output_comm_rank, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_label_to_output_comm_rank create failed.");

  d_label_to_output_comm_rank_view = cugraph_type_erased_device_array_view(d_label_to_output_comm_rank);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_label_to_output_comm_rank_view, (byte_t*)label_to_output_comm_rank, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "label_to_output_comm_rank copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  d_start_labels_view,
                                                                  d_label_list_view,
                                                                  d_label_to_output_comm_rank_view,
                                                                  h_fan_out_view,
                                                                  rng_state,
                                                                  FALSE,
                                                                  TRUE,
                                                                  FALSE,
                                                                  &result,
                                                                  &ret_error);

#ifdef NO_CUGRAPH_OPS
  TEST_ASSERT(
    test_ret_value, ret_code != CUGRAPH_SUCCESS, "uniform_neighbor_sample should have failed")
#else
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_neighbor_sample failed.");

  cugraph_type_erased_device_array_view_t* result_srcs = NULL;
  cugraph_type_erased_device_array_view_t* result_dsts = NULL;
  cugraph_type_erased_device_array_view_t* result_edge_id = NULL;
  cugraph_type_erased_device_array_view_t* result_weights = NULL;
  cugraph_type_erased_device_array_view_t* result_hops = NULL;
  cugraph_type_erased_device_array_view_t* result_offsets = NULL;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_hops       = cugraph_sample_result_get_hop(result);
  result_offsets    = cugraph_sample_result_get_offsets(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);
  size_t result_offsets_size = cugraph_type_erased_device_array_view_size(result_offsets);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_edge_id[result_size];
  weight_t h_weight[result_size];
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
    handle, (byte_t*)h_hops, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_offsets, result_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M_w[num_vertices][num_vertices];
  edge_t M_edge_id[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j) {
      M_w[i][j]         = 0.0;
      M_edge_id[i][j]   = -1;
    }

  for (int i = 0; i < num_edges; ++i) {
    M_w[src[i]][dst[i]]         = wgt[i];
    M_edge_id[src[i]][dst[i]]   = edge_ids[i];
  }

  for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                M_w[h_srcs[i]][h_dsts[i]] == h_weight[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  TEST_ASSERT(test_ret_value,
              result_offsets_size == expected_size[rank],
              "incorrect number of results");

  for (int i = 0 ; i < (result_offsets_size - 1) && (test_ret_value == 0) ; ++i) {
    for (int j = h_result_offsets[i] ; j < (h_result_offsets[i+1] - 1) && (test_ret_value == 0) ; ++j) {
      TEST_ASSERT(test_ret_value,
                  h_hops[j] <= h_hops[j+1],
                  "Results not sorted by hop id");
    }
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
}

int test_uniform_neighbor_sample_dedupe_sources(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]       = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 3, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2, 3};
  int start_labels[] = { 6, 12 };
  int fan_out[]        = {-1, -1, -1};

  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;

  bool_t with_replacement = FALSE;
  bool_t return_hops = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources = TRUE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources);
}

int test_uniform_neighbor_sample_unique_sources(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]       = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2, 3};
  int start_labels[] = { 6, 12 };
  int fan_out[]        = {-1, -1, -1};

  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;

  bool_t with_replacement = FALSE;
  bool_t return_hops = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = EXCLUDE;
  bool_t dedupe_sources = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources);
}

int test_uniform_neighbor_sample_carry_over_sources(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t fan_out_size = 3;
  size_t num_starts   = 2;

  vertex_t src[]       = {0, 0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 2, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  int32_t edge_types[] = {8, 7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2, 3};
  int start_labels[] = { 6, 12 };
  int fan_out[]        = {-1, -1, -1};

  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;

  bool_t with_replacement = FALSE;
  bool_t return_hops = TRUE;
  cugraph_prior_sources_behavior_t prior_sources_behavior = CARRY_OVER;
  bool_t dedupe_sources = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_uniform_neighbor_sample, handle);
  result |= RUN_MG_TEST(test_uniform_neighbor_from_alex, handle);
  //result |= RUN_MG_TEST(test_uniform_neighbor_sample_alex_bug, handle);
  result |= RUN_MG_TEST(test_uniform_neighbor_sample_sort_by_hop, handle);
  result |= RUN_MG_TEST(test_uniform_neighbor_sample_dedupe_sources, handle);
  result |= RUN_MG_TEST(test_uniform_neighbor_sample_unique_sources, handle);
  result |= RUN_MG_TEST(test_uniform_neighbor_sample_carry_over_sources, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
