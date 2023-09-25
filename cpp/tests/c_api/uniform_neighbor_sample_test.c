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

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

data_type_id_t vertex_tid    = INT32;
data_type_id_t edge_tid      = INT32;
data_type_id_t weight_tid    = FLOAT32;
data_type_id_t edge_id_tid   = INT32;
data_type_id_t edge_type_tid = INT32;

int vertex_id_compare_function(const void * a, const void * b) {
  if (*((vertex_t *) a) < *((vertex_t *) b))
    return -1;
  else if (*((vertex_t *) a) > *((vertex_t *) b))
    return 1;
  else
    return 0;
}

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

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_start_vertices, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_start_vertices, INT32, &d_start_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start_labels create failed.");

  d_start_labels_view = cugraph_type_erased_device_array_view(d_start_labels);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_labels_view, (byte_t*)h_start_labels, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start_labels copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, fan_out_size, INT32);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  cugraph_sampling_options_t *sampling_options;

  ret_code = cugraph_sampling_options_create(&sampling_options, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "sampling_options create failed.");

  cugraph_sampling_set_with_replacement(sampling_options, with_replacement);
  cugraph_sampling_set_return_hops(sampling_options, return_hops);
  cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior);
  cugraph_sampling_set_dedupe_sources(sampling_options, dedupe_sources);
  cugraph_sampling_set_renumber_results(sampling_options, renumber_results);

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
  cugraph_type_erased_device_array_view_t* result_offsets;
  cugraph_type_erased_device_array_view_t* result_labels;
  cugraph_type_erased_device_array_view_t* result_renumber_map;
  cugraph_type_erased_device_array_view_t* result_renumber_map_offsets;

  result_srcs                 = cugraph_sample_result_get_sources(result);
  result_dsts                 = cugraph_sample_result_get_destinations(result);
  result_edge_id              = cugraph_sample_result_get_edge_id(result);
  result_weights              = cugraph_sample_result_get_edge_weight(result);
  result_edge_types           = cugraph_sample_result_get_edge_type(result);
  result_hops                 = cugraph_sample_result_get_hop(result);
  result_hops                 = cugraph_sample_result_get_hop(result);
  result_offsets              = cugraph_sample_result_get_offsets(result);
  result_labels               = cugraph_sample_result_get_start_labels(result);
  result_renumber_map         = cugraph_sample_result_get_renumber_map(result);
  result_renumber_map_offsets = cugraph_sample_result_get_renumber_map_offsets(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);
  size_t result_offsets_size = cugraph_type_erased_device_array_view_size(result_offsets);
  size_t renumber_map_size = 0;

  if (renumber_results) {
    renumber_map_size = cugraph_type_erased_device_array_view_size(result_renumber_map);
  }

  vertex_t h_result_srcs[result_size];
  vertex_t h_result_dsts[result_size];
  edge_t h_result_edge_id[result_size];
  weight_t h_result_weight[result_size];
  int32_t h_result_edge_types[result_size];
  int32_t h_result_hops[result_size];
  size_t h_result_offsets[result_offsets_size];
  int h_result_labels[result_offsets_size-1];
  vertex_t h_renumber_map[renumber_map_size];
  size_t h_renumber_map_offsets[result_offsets_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_srcs, result_srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_dsts, result_dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_edge_id, result_edge_id, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_weight, result_weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_edge_types, result_edge_types, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_hops, result_hops, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_offsets, result_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_labels, result_labels, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  if (renumber_results) {
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_renumber_map, result_renumber_map, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_renumber_map_offsets, result_renumber_map_offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
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
    M_w[h_src[i]][h_dst[i]]         = h_wgt[i];
    M_edge_id[h_src[i]][h_dst[i]]   = h_edge_ids[i];
    M_edge_type[h_src[i]][h_dst[i]] = h_edge_types[i];
  }

  if (renumber_results) {
    for (int label_id = 0 ; label_id < (result_offsets_size - 1) ; ++label_id) {
      for (size_t i = h_result_offsets[label_id]; (i < h_result_offsets[label_id+1]) && (test_ret_value == 0) ; ++i) {
        vertex_t src = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_srcs[i]];
        vertex_t dst = h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]];

        TEST_ASSERT(test_ret_value,
                    M_w[src][dst] == h_result_weight[i],
                    "uniform_neighbor_sample got edge that doesn't exist");
        TEST_ASSERT(test_ret_value,
                    M_edge_id[src][dst] == h_result_edge_id[i],
                    "uniform_neighbor_sample got edge that doesn't exist");
        TEST_ASSERT(test_ret_value,
                    M_edge_type[src][dst] == h_result_edge_types[i],
                    "uniform_neighbor_sample got edge that doesn't exist");
      }
    }
  } else {
    for (int i = 0; (i < result_size) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(test_ret_value,
                  M_w[h_result_srcs[i]][h_result_dsts[i]] == h_result_weight[i],
                  "uniform_neighbor_sample got edge that doesn't exist");
      TEST_ASSERT(test_ret_value,
                  M_edge_id[h_result_srcs[i]][h_result_dsts[i]] == h_result_edge_id[i],
                  "uniform_neighbor_sample got edge that doesn't exist");
      TEST_ASSERT(test_ret_value,
                  M_edge_type[h_result_srcs[i]][h_result_dsts[i]] == h_result_edge_types[i],
                  "uniform_neighbor_sample got edge that doesn't exist");
    }
  }

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

  for (int label_id = 0 ; label_id < (result_offsets_size - 1) ; ++label_id) {
    size_t   sources_size = 0;
    size_t   destinations_size = 0;

    // Fill sources with the input sources
    for (size_t i = 0 ; i < num_start_vertices ; ++i) {
      if (h_start_labels[i] == h_result_labels[label_id]) {
        check_sources[sources_size] = h_start[i];
        ++sources_size;
      }
    }

    if (renumber_results) {
      size_t num_vertex_ids = 2 * (h_result_offsets[label_id+1] - h_result_offsets[label_id]);
      vertex_t vertex_ids[num_vertex_ids];
      
      for (size_t i = 0 ; (i < (h_result_offsets[label_id+1] - h_result_offsets[label_id])) && (test_ret_value == 0) ; ++i) {
        vertex_ids[2*i] = h_result_srcs[h_result_offsets[label_id] + i];
        vertex_ids[2*i+1] = h_result_dsts[h_result_offsets[label_id] + i];
      }

      qsort(vertex_ids, num_vertex_ids, sizeof(vertex_t), vertex_id_compare_function);

      vertex_t current_v = 0;
      for (size_t i = 0 ; (i < num_vertex_ids) && (test_ret_value == 0) ; ++i) {
        if (vertex_ids[i] == current_v)
          ++current_v;
        else 
            TEST_ASSERT(test_ret_value,
                        vertex_ids[i] == (current_v - 1),
                        "vertices are not properly renumbered");
      }
    }

    for (int hop = 0 ; hop < fan_out_size ; ++hop) {
      if (prior_sources_behavior == CARRY_OVER) {
        destinations_size = sources_size;
        for (size_t i = 0 ; i < sources_size ; ++i) {
          check_destinations[i] = check_sources[i];
        }
      }

      for (size_t i = h_result_offsets[label_id]; (i < h_result_offsets[label_id+1]) && (test_ret_value == 0) ; ++i) {
        if (h_result_hops[i] == hop) {
          bool found = false;
          for (size_t j = 0 ; (!found) && (j < sources_size) ; ++j) {
            found = renumber_results ? (h_renumber_map[h_renumber_map_offsets[label_id] + h_result_srcs[i]] == check_sources[j])
              : (h_result_srcs[i] == check_sources[j]);
          }

          TEST_ASSERT(test_ret_value, found, "encountered source vertex that was not part of previous frontier");
        }

        if (prior_sources_behavior == CARRY_OVER) {
          // Make sure destination isn't already in the source list
          bool found = false;
          for (size_t j = 0 ; (!found) && (j < destinations_size) ; ++j) {
            found = renumber_results ? (h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]] == check_destinations[j])
              : (h_result_dsts[i] == check_destinations[j]);
          }

          if (!found) {
            check_destinations[destinations_size] = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]] : h_result_dsts[i];
            ++destinations_size;
          }
        } else {
          check_destinations[destinations_size] = renumber_results ? h_renumber_map[h_renumber_map_offsets[label_id] + h_result_dsts[i]] : h_result_dsts[i];
          ++destinations_size;
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
      for (size_t i = h_result_offsets[label_id]; (i < h_result_offsets[label_id+1]) && (test_ret_value == 0) ; ++i) {
        for (size_t j = i + 1 ; (j < h_result_offsets[label_id+1]) && (test_ret_value == 0) ; ++j) {
          if (h_result_srcs[i] == h_result_srcs[j]) {
            TEST_ASSERT(test_ret_value,
                        h_result_hops[i] == h_result_hops[j],
                        "source vertex should not have been used in diferent hops");
          }
        }
      }
    }

    if (dedupe_sources) {
      // Make sure vertex v only appears as source once for each edge after it appears as destination
      // Externally test this by verifying that vertex v only appears in <= hop size/degree
      for (size_t i = h_result_offsets[label_id]; (i < h_result_offsets[label_id+1]) && (test_ret_value == 0) ; ++i) {
        if (h_result_hops[i] > 0) {
          size_t num_occurrences = 1;
          for (size_t j = i + 1 ; j < h_result_offsets[label_id+1] ; ++j) {
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

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
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

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

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

  ret_code = cugraph_sg_graph_create(p_handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     NULL,
                                     NULL,
                                     store_transposed,
                                     renumber,
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

int test_uniform_neighbor_sample_with_properties(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 1;
  size_t num_starts   = 1;

  vertex_t src[]       = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  int32_t edge_types[] = {7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2};
  int fan_out[]        = {-1};

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

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
  cugraph_type_erased_host_array_view_t* h_fan_out_view = NULL;

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, 1, INT32);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  NULL,
                                                                  NULL,
                                                                  NULL,
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

  cugraph_type_erased_device_array_view_t* result_srcs;
  cugraph_type_erased_device_array_view_t* result_dsts;
  cugraph_type_erased_device_array_view_t* result_edge_id;
  cugraph_type_erased_device_array_view_t* result_weights;
  cugraph_type_erased_device_array_view_t* result_edge_types;
  cugraph_type_erased_device_array_view_t* result_hops;

  result_srcs       = cugraph_sample_result_get_sources(result);
  result_dsts       = cugraph_sample_result_get_destinations(result);
  result_edge_id    = cugraph_sample_result_get_edge_id(result);
  result_weights    = cugraph_sample_result_get_edge_weight(result);
  result_edge_types = cugraph_sample_result_get_edge_type(result);
  result_hops       = cugraph_sample_result_get_hop(result);

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);

  vertex_t h_srcs[result_size];
  vertex_t h_dsts[result_size];
  edge_t h_edge_id[result_size];
  weight_t h_weight[result_size];
  int32_t h_edge_types[result_size];
  int32_t h_hops[result_size];

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

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_hops, result_hops, &ret_error);
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
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_type[h_srcs[i]][h_dsts[i]] == h_edge_types[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
}

int test_uniform_neighbor_sample_with_labels(const cugraph_resource_handle_t* handle)
{
  data_type_id_t vertex_tid    = INT32;
  data_type_id_t edge_tid      = INT32;
  data_type_id_t weight_tid    = FLOAT32;
  data_type_id_t edge_id_tid   = INT32;
  data_type_id_t edge_type_tid = INT32;

  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t fan_out_size = 1;
  size_t num_starts   = 2;

  vertex_t src[]       = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]       = {1, 3, 4, 0, 1, 3, 5, 5};
  edge_t edge_ids[]    = {0, 1, 2, 3, 4, 5, 6, 7};
  weight_t weight[]    = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  int32_t edge_types[] = {7, 6, 5, 4, 3, 2, 1, 0};
  vertex_t start[]     = {2, 3};
  size_t start_labels[] = { 6, 12 };
  int fan_out[]        = {-1};

  // Create graph
  int test_ret_value              = 0;
  cugraph_error_code_t ret_code   = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error      = NULL;
  cugraph_graph_t* graph          = NULL;
  cugraph_sample_result_t* result = NULL;

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

  h_fan_out_view = cugraph_type_erased_host_array_view_create(fan_out, 1, INT32);

  cugraph_rng_state_t *rng_state;
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");

  ret_code = cugraph_uniform_neighbor_sample_with_edge_properties(handle,
                                                                  graph,
                                                                  d_start_view,
                                                                  d_start_labels_view,
                                                                  NULL,
                                                                  NULL,
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

  size_t result_size = cugraph_type_erased_device_array_view_size(result_srcs);
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
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_id[h_srcs[i]][h_dsts[i]] == h_edge_id[i],
                "uniform_neighbor_sample got edge that doesn't exist");
    TEST_ASSERT(test_ret_value,
                M_edge_type[h_srcs[i]][h_dsts[i]] == h_edge_types[i],
                "uniform_neighbor_sample got edge that doesn't exist");
  }

  cugraph_sample_result_free(result);
#endif

  cugraph_sg_graph_free(graph);
  cugraph_error_free(ret_error);
}

int test_uniform_neighbor_sample_clean(const cugraph_resource_handle_t* handle)
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
  bool_t dedupe_sources = FALSE;
  bool_t renumber_results = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources, renumber_results);
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
  bool_t renumber_results = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources, renumber_results);
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
  bool_t renumber_results = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources, renumber_results);
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
  bool_t renumber_results = FALSE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources, renumber_results);
}

int test_uniform_neighbor_sample_renumber_results(const cugraph_resource_handle_t* handle)
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
  cugraph_prior_sources_behavior_t prior_sources_behavior = DEFAULT;
  bool_t dedupe_sources = FALSE;
  bool_t renumber_results = TRUE;

  return generic_uniform_neighbor_sample_test(handle, src, dst, weight, edge_ids, edge_types, num_vertices, num_edges,
                                              start, start_labels, num_starts,
                                              fan_out, fan_out_size, with_replacement,
                                              return_hops, prior_sources_behavior, dedupe_sources, renumber_results);
}

int main(int argc, char** argv)
{
  cugraph_resource_handle_t* handle = NULL;

  handle = cugraph_create_resource_handle(NULL);

  int result = 0;
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_with_properties, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_with_labels, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_clean, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_dedupe_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_unique_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_carry_over_sources, handle);
  result |= RUN_TEST_NEW(test_uniform_neighbor_sample_renumber_results, handle);

  cugraph_free_resource_handle(handle);

  return result;
}
