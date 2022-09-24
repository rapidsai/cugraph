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
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

typedef enum { JACCARD, SORENSEN, OVERLAP } similarity_t;

int generic_similarity_test(vertex_t* h_src,
                            vertex_t* h_dst,
                            weight_t* h_wgt,
                            vertex_t* h_first,
                            vertex_t* h_second,
                            weight_t* h_result,
                            size_t num_vertices,
                            size_t num_edges,
                            size_t num_pairs,
                            bool_t store_transposed,
                            bool_t use_weight,
                            similarity_t test_type)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle    = NULL;
  cugraph_graph_t* graph               = NULL;
  cugraph_similarity_result_t* result  = NULL;
  cugraph_vertex_pairs_t* vertex_pairs = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

#if 0
  // FIXME: populate vertex_pairs

  switch (test_type) {
    case JACCARD:
      ret_code = cugraph_jaccard_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case SORENSEN:
      ret_code = cugraph_sorensen_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
    case OVERLAP:
      ret_code = cugraph_overlap_coefficients(
        handle, graph, vertex_pairs, use_weight, FALSE, &result, &ret_error);
      break;
  }

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_katz_centrality failed.");

  cugraph_type_erased_device_array_view_t* first;
  cugraph_type_erased_device_array_view_t* second;
  cugraph_type_erased_device_array_view_t* similarity_coefficient;

  similarity_coefficient = cugraph_similarity_result_get_similarity(result);

  vertex_t h_first[num_pairs];
  vertex_t h_second[num_pairs];
  weight_t h_similarity_coefficient[num_pairs];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_first, first, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_second, second, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_similarity_coefficient, similarity_coefficient, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_pairs) && (test_ret_value == 0); ++i) {
    /*
      // FIXME: Implement this properly
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_similarity_coefficient[h_vertices[i]], h_centralities[i], 0.001),
                "katz centrality results don't match");
    */
  }
#endif

  if (result != NULL) cugraph_similarity_result_free(result);
  if (vertex_pairs != NULL) cugraph_vertex_pairs_free(vertex_pairs);
  cugraph_sg_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_jaccard()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 FALSE,
                                 JACCARD);
}

int test_weighted_jaccard()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 TRUE,
                                 JACCARD);
}

int test_sorensen()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 FALSE,
                                 SORENSEN);
}

int test_weighted_sorensen()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 TRUE,
                                 SORENSEN);
}

int test_overlap()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 FALSE,
                                 OVERLAP);
}

int test_weighted_overlap()
{
  size_t num_edges    = 16;
  size_t num_vertices = 6;
  size_t num_pairs    = 10;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t h_first[]  = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3};
  vertex_t h_second[] = {1, 3, 4, 2, 3, 5, 3, 4, 5, 4};
  weight_t h_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // TODO: Fill in

  return generic_similarity_test(h_src,
                                 h_dst,
                                 h_wgt,
                                 h_first,
                                 h_second,
                                 h_result,
                                 num_vertices,
                                 num_edges,
                                 num_pairs,
                                 FALSE,
                                 TRUE,
                                 OVERLAP);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_jaccard);
  result |= RUN_TEST(test_sorensen);
  result |= RUN_TEST(test_overlap);
  result |= RUN_TEST(test_weighted_jaccard);
  result |= RUN_TEST(test_weighted_sorensen);
  result |= RUN_TEST(test_weighted_overlap);
  return result;
}
