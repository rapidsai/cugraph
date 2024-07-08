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

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

typedef enum {
  JACCARD,
  SORENSEN,
  OVERLAP,
  COSINE,
  ALL_PAIRS_JACCARD,
  ALL_PAIRS_SORENSEN,
  ALL_PAIRS_OVERLAP,
  ALL_PAIRS_COSINE
} similarity_t;

int generic_similarity_test(const cugraph_resource_handle_t* handle,
                            vertex_t* h_src,
                            vertex_t* h_dst,
                            weight_t* h_wgt,
                            vertex_t* h_first,
                            vertex_t* h_second,
                            vertex_t* h_start_vertices,
                            weight_t* h_result,
                            size_t num_vertices,
                            size_t num_edges,
                            size_t num_pairs,
                            size_t num_start_vertices,
                            size_t topk,
                            bool_t store_transposed,
                            bool_t use_weight,
                            similarity_t test_type)
{
  int test_ret_value        = 0;
  data_type_id_t vertex_tid = INT32;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                                = NULL;
  cugraph_similarity_result_t* result                   = NULL;
  cugraph_vertex_pairs_t* vertex_pairs                  = NULL;
  cugraph_type_erased_device_array_t* v1                = NULL;
  cugraph_type_erased_device_array_t* v2                = NULL;
  cugraph_type_erased_device_array_t* start_v           = NULL;
  cugraph_type_erased_device_array_view_t* v1_view      = NULL;
  cugraph_type_erased_device_array_view_t* v2_view      = NULL;
  cugraph_type_erased_device_array_view_t* start_v_view = NULL;

  ret_code = create_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, TRUE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (topk == 0) { topk = SIZE_MAX; }

  if (cugraph_resource_handle_get_rank(handle) != 0) { num_pairs = 0; }

  if (h_first != NULL && h_second != NULL) {
    ret_code =
      cugraph_type_erased_device_array_create(handle, num_pairs, vertex_tid, &v1, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "v1 create failed.");

    ret_code =
      cugraph_type_erased_device_array_create(handle, num_pairs, vertex_tid, &v2, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "v2 create failed.");

    v1_view = cugraph_type_erased_device_array_view(v1);
    v2_view = cugraph_type_erased_device_array_view(v2);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, v1_view, (byte_t*)h_first, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "h_first copy_from_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, v2_view, (byte_t*)h_second, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "h_second copy_from_host failed.");

    ret_code =
      cugraph_create_vertex_pairs(handle, graph, v1_view, v2_view, &vertex_pairs, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create vertex pairs failed.");
  }

  if (h_start_vertices != NULL) {
    ret_code = cugraph_type_erased_device_array_create(
      handle, num_start_vertices, vertex_tid, &start_v, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "v1 create failed.");
    start_v_view = cugraph_type_erased_device_array_view(start_v);

    ret_code = cugraph_type_erased_device_array_view_copy_from_host(
      handle, start_v_view, (byte_t*)h_start_vertices, &ret_error);

    TEST_ASSERT(
      test_ret_value, ret_code == CUGRAPH_SUCCESS, "h_start_vertices copy_from_host failed.");
  }

  switch (test_type) {
    case JACCARD:
      ret_code = cugraph_jaccard_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case ALL_PAIRS_JACCARD:
      ret_code = cugraph_all_pairs_jaccard_coefficients(
        handle, graph, start_v_view, use_weight, topk, FALSE, &result, &ret_error);
      break;
    case SORENSEN:
      ret_code = cugraph_sorensen_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case ALL_PAIRS_SORENSEN:
      ret_code = cugraph_all_pairs_sorensen_coefficients(
        handle, graph, start_v_view, use_weight, topk, FALSE, &result, &ret_error);
      break;
    case OVERLAP:
      ret_code = cugraph_overlap_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case ALL_PAIRS_OVERLAP:
      ret_code = cugraph_all_pairs_overlap_coefficients(
        handle, graph, start_v_view, use_weight, topk, FALSE, &result, &ret_error);
      break;
    case COSINE:
      ret_code = cugraph_cosine_similarity_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case ALL_PAIRS_COSINE:
      ret_code = cugraph_all_pairs_cosine_similarity_coefficients(
        handle, graph, start_v_view, use_weight, topk, FALSE, &result, &ret_error);
      break;
  }

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph similarity failed.");

  cugraph_type_erased_device_array_view_t* similarity_coefficient;

  similarity_coefficient = cugraph_similarity_result_get_similarity(result);

  switch (test_type) {
    case ALL_PAIRS_JACCARD:
      num_pairs = cugraph_type_erased_device_array_view_size(similarity_coefficient);
      break;
    case ALL_PAIRS_SORENSEN:
      num_pairs = cugraph_type_erased_device_array_view_size(similarity_coefficient);
      break;
    case ALL_PAIRS_OVERLAP:
      num_pairs = cugraph_type_erased_device_array_view_size(similarity_coefficient);
      break;
    case ALL_PAIRS_COSINE:
      num_pairs = cugraph_type_erased_device_array_view_size(similarity_coefficient);
      break;
  }

  weight_t h_similarity_coefficient[num_pairs];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_similarity_coefficient, similarity_coefficient, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_pairs) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_similarity_coefficient[i], h_result[i], 0.001),
                "similarity results don't match");
  }

  if (result != NULL) cugraph_similarity_result_free(result);
  if (vertex_pairs != NULL) cugraph_vertex_pairs_free(vertex_pairs);
  cugraph_mg_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_jaccard(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 10;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]         = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[]        = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[] = {0.2, 0.666667, 0.333333, 0.4, 0.166667, 0.5, 0.2, 0.25, 0.25, 0.666667};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 JACCARD);
}

int test_weighted_jaccard(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 7;
  size_t num_pairs          = 3;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
  vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
  weight_t h_wgt[] = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

  vertex_t h_first[]         = {0, 0, 1};
  vertex_t h_second[]        = {1, 2, 3};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.357143, 0.208333, 0.0};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 TRUE,
                                 JACCARD);
}

int test_all_pairs_jaccard(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[] = {0.2,      0.25,      0.666667, 0.333333, 0.2,  0.4,      0.166667, 0.5,
                         0.25,     0.4,       0.2,      0.25,     0.25, 0.666667, 0.166667, 0.2,
                         0.666667, 0.3333333, 0.25,     0.666667, 0.5,  0.25};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_JACCARD);
}

int test_all_pairs_jaccard_with_start_vertices(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 0;

  vertex_t h_src[]            = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]            = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first           = NULL;
  vertex_t* h_second          = NULL;
  vertex_t h_start_vertices[] = {0, 1, 2};
  weight_t h_result[]         = {
    0.2, 0.25, 0.666667, 0.333333, 0.2, 0.4, 0.166667, 0.5, 0.25, 0.4, 0.2, 0.25, 0.25};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_JACCARD);
}

int test_all_pairs_jaccard_with_topk(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 5;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.666667, 0.666667, 0.666667, 0.666667, 0.5};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_JACCARD);
}

int test_sorensen(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 10;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]         = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[]        = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[] = {0.333333, 0.8, 0.5, 0.571429, 0.285714, 0.666667, 0.333333, 0.4, 0.4, 0.8};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 SORENSEN);
}

int test_weighted_sorensen(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 7;
  size_t num_pairs          = 3;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
  vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
  weight_t h_wgt[] = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

  vertex_t h_first[]         = {0, 0, 1};
  vertex_t h_second[]        = {1, 2, 3};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.526316, 0.344828, 0.000000};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 TRUE,
                                 SORENSEN);
}

int test_all_pairs_sorensen(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[] = {0.333333, 0.4,      0.8,      0.5, 0.333333, 0.571429, 0.285714, 0.666667,
                         0.4,      0.571429, 0.333333, 0.4, 0.4,      0.8,      0.285714, 0.333333,
                         0.8,      0.5,      0.4,      0.8, 0.666667, 0.4};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_SORENSEN);
}

int test_all_pairs_sorensen_with_start_vertices(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 0;

  vertex_t h_src[]            = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]            = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first           = NULL;
  vertex_t* h_second          = NULL;
  vertex_t h_start_vertices[] = {0, 1, 2};
  weight_t h_result[]         = {0.333333,
                                 0.4,
                                 0.8,
                                 0.5,
                                 0.333333,
                                 0.571429,
                                 0.285714,
                                 0.666667,
                                 0.4,
                                 0.571429,
                                 0.333333,
                                 0.4,
                                 0.4};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_SORENSEN);
}

int test_all_pairs_sorensen_with_topk(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 5;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.8, 0.8, 0.8, 0.8, 0.666667};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_SORENSEN);
}

int test_overlap(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 10;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]         = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[]        = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.5, 1, 0.5, 0.666667, 0.333333, 1, 0.333333, 0.5, 0.5, 1};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 OVERLAP);
}

int test_weighted_overlap(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 7;
  size_t num_pairs          = 3;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
  vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
  weight_t h_wgt[] = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

  vertex_t h_first[]         = {0, 0, 1};
  vertex_t h_second[]        = {1, 2, 3};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.714286, 0.416667, 0.000000};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 TRUE,
                                 OVERLAP);
}

int test_all_pairs_overlap(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.5, 0.5,      1.0,      0.5, 0.5, 0.666667, 0.333333, 1.0,
                                0.5, 0.666667, 0.333333, 0.5, 0.5, 1.0,      0.333333, 0.333333,
                                1.0, 0.5,      0.5,      1.0, 1.0, 0.5};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_OVERLAP);
}

int test_all_pairs_overlap_with_start_vertices(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 0;

  vertex_t h_src[]            = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]            = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first           = NULL;
  vertex_t* h_second          = NULL;
  vertex_t h_start_vertices[] = {0, 1, 2};
  weight_t h_result[]         = {
    0.5, 0.5, 1.0, 0.5, 0.5, 0.666667, 0.333333, 1.0, 0.5, 0.666667, 0.333333, 0.5, 0.5};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_OVERLAP);
}

int test_all_pairs_overlap_with_topk(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 5;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {1.0, 1.0, 1.0, 1.0, 1.0};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_OVERLAP);
}

int test_cosine(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 10;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]         = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[]        = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 COSINE);
}

int test_weighted_cosine(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 7;
  size_t num_pairs          = 2;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[] = {0, 1, 2, 0, 1, 2, 3, 3, 3, 4, 4, 4, 0, 5, 2, 6};
  vertex_t h_dst[] = {3, 3, 3, 4, 4, 4, 0, 1, 2, 0, 1, 2, 5, 0, 6, 2};
  weight_t h_wgt[] = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.5, 4.0, 4.0};

  vertex_t h_first[]         = {0, 0};
  vertex_t h_second[]        = {1, 2};
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.990830, 0.976187};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 TRUE,
                                 COSINE);
}

int test_all_pairs_cosine(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 0;
  size_t topk               = 0;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {0.5, 0.5,      1.0,      0.5, 0.5, 0.666667, 0.333333, 1.0,
                                0.5, 0.666667, 0.333333, 0.5, 0.5, 1.0,      0.333333, 0.333333,
                                1.0, 0.5,      0.5,      1.0, 1.0, 0.5};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_COSINE);
}

int test_all_pairs_cosine_with_start_vertices(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 0;

  vertex_t h_src[]            = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]            = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]            = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first           = NULL;
  vertex_t* h_second          = NULL;
  vertex_t h_start_vertices[] = {0, 1, 2};
  weight_t h_result[]         = {
    0.5, 0.5, 1.0, 0.5, 0.5, 0.666667, 0.333333, 1.0, 0.5, 0.666667, 0.333333, 0.5, 0.5};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_COSINE);
}

int test_all_pairs_cosine_with_topk(const cugraph_resource_handle_t* handle)
{
  size_t num_edges          = 16;
  size_t num_vertices       = 6;
  size_t num_pairs          = 0;
  size_t num_start_vertices = 3;
  size_t topk               = 5;

  vertex_t h_src[]           = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]           = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]           = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t* h_first          = NULL;
  vertex_t* h_second         = NULL;
  vertex_t* h_start_vertices = NULL;
  weight_t h_result[]        = {1.0, 1.0, 1.0, 1.0, 1.0};

  return generic_similarity_test(handle,
                                 h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_start_vertices,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 num_start_vertices,
                                 topk,
                                 FALSE,
                                 FALSE,
                                 ALL_PAIRS_COSINE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;

  result |= RUN_MG_TEST(test_jaccard, handle);
  result |= RUN_MG_TEST(test_weighted_jaccard, handle);
  result |= RUN_MG_TEST(test_all_pairs_jaccard, handle);
  result |= RUN_MG_TEST(test_all_pairs_jaccard_with_start_vertices, handle);
  result |= RUN_MG_TEST(test_all_pairs_jaccard_with_topk, handle);

  result |= RUN_MG_TEST(test_sorensen, handle);
  result |= RUN_MG_TEST(test_weighted_sorensen, handle);
  result |= RUN_MG_TEST(test_all_pairs_sorensen, handle);
  result |= RUN_MG_TEST(test_all_pairs_sorensen_with_start_vertices, handle);
  result |= RUN_MG_TEST(test_all_pairs_sorensen_with_topk, handle);

  result |= RUN_MG_TEST(test_overlap, handle);
  result |= RUN_MG_TEST(test_weighted_overlap, handle);
  result |= RUN_MG_TEST(test_all_pairs_overlap, handle);
  result |= RUN_MG_TEST(test_all_pairs_overlap_with_start_vertices, handle);
  result |= RUN_MG_TEST(test_all_pairs_overlap_with_topk, handle);

  result |= RUN_MG_TEST(test_cosine, handle);
  result |= RUN_MG_TEST(test_weighted_cosine, handle);
  result |= RUN_MG_TEST(test_all_pairs_cosine, handle);
  result |= RUN_MG_TEST(test_all_pairs_cosine_with_start_vertices, handle);
  result |= RUN_MG_TEST(test_all_pairs_cosine_with_topk, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
