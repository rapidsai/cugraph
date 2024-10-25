/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph_c/graph_generators.h>

#include <stdio.h>

/*
 * Simple rmat generator test
 */
int test_rmat_generation(const cugraph_resource_handle_t* handle)
{
// FIXME: this should be re-implemented (this test doesn't work for more than 8 GPUs, and this
// assumes that the underlying random number generator produces a known sequence)
#if 1
  return CUGRAPH_SUCCESS;
#else
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  vertex_t expected_src_0[] = {17, 18, 0,  16, 1,  24, 16, 1, 6, 4,  2, 1, 14, 2,  16,
                               2,  5,  23, 4,  10, 4,  3,  0, 4, 11, 0, 0, 2,  24, 0};
  vertex_t expected_dst_0[] = {0, 10, 23, 0,  26, 0, 2,  1, 27, 8,  1, 0, 21, 21, 0,
                               4, 8,  14, 10, 17, 0, 16, 0, 16, 25, 5, 8, 8,  4,  19};
  vertex_t expected_src_1[] = {1, 0, 2,  25, 0, 5, 18, 0, 0, 2, 4, 0, 8, 9, 2,
                               2, 9, 16, 4,  1, 0, 2,  0, 0, 9, 5, 4, 6, 4, 9};
  vertex_t expected_dst_1[] = {4,  1, 4, 0,  16, 16, 16, 3,  2,  0,  1,  0,  7,  0, 2,
                               13, 6, 8, 18, 16, 0,  4,  27, 16, 24, 17, 21, 25, 1, 0};
  vertex_t expected_src_2[] = {4,  4,  4, 0,  2, 8, 6,  0, 0, 0, 0, 16, 8,  18, 1,
                               19, 16, 0, 24, 4, 0, 17, 0, 8, 5, 8, 8,  12, 10, 1};
  vertex_t expected_dst_2[] = {9, 2,  3,  18, 0,  24, 2,  4, 0, 0, 25, 0, 0, 4, 20,
                               0, 16, 10, 17, 16, 25, 16, 1, 1, 4, 24, 6, 6, 0, 0};
  vertex_t expected_src_3[] = {9,  16, 0, 1,  4,  26, 7,  20, 0, 0, 0,  25, 2, 16, 4,
                               12, 18, 2, 16, 24, 5,  20, 1,  4, 1, 22, 9,  1, 2,  16};
  vertex_t expected_dst_3[] = {18, 0, 16, 2, 1, 2, 16, 0, 5, 2, 22, 8, 6, 11, 0,
                               17, 0, 2,  0, 6, 9, 1,  0, 8, 8, 9,  4, 2, 3,  18};
  vertex_t expected_src_4[] = {16, 24, 25, 0, 18, 0, 0, 0, 0, 0,  16, 0, 1,  8, 8,
                               0,  4,  0,  4, 0,  5, 8, 0, 2, 21, 11, 0, 24, 0, 4};
  vertex_t expected_dst_4[] = {8, 13, 16, 0, 0, 2, 17, 18, 16, 14, 4, 0,  0,  1, 4,
                               2, 1,  0,  1, 0, 0, 0,  26, 0,  20, 1, 14, 21, 2, 28};
  vertex_t expected_src_5[] = {0, 24, 8,  2,  6, 0, 10, 4,  4, 0, 10, 4,  17, 0, 17,
                               0, 10, 20, 26, 0, 6, 0,  11, 2, 1, 0,  17, 2,  4, 8};
  vertex_t expected_dst_5[] = {2,  24, 0, 18, 0, 0, 1,  0, 14, 18, 16, 19, 1, 0, 8,
                               29, 12, 1, 8,  8, 0, 22, 4, 12, 2,  1,  0,  0, 8, 8};
  vertex_t expected_src_6[] = {4,  4, 10, 8, 26, 0,  20, 2,  14, 1, 8, 1, 0, 16, 24,
                               28, 8, 18, 7, 5,  16, 2,  12, 22, 8, 4, 1, 1, 12, 8};
  vertex_t expected_dst_6[] = {4, 2, 6,  1, 3, 16, 5, 8, 5, 0, 19, 9, 0, 1, 0,
                               0, 8, 26, 0, 9, 16, 0, 3, 6, 2, 24, 8, 0, 4, 10};
  vertex_t expected_src_7[] = {5, 0, 16, 0, 16, 21, 1, 3, 8, 4, 0, 22, 25, 8, 0,
                               4, 0, 6,  0, 0,  0,  0, 2, 0, 0, 4, 8,  0,  0, 0};
  vertex_t expected_dst_7[] = {0, 4, 30, 10, 5, 0, 24, 0, 0, 4,  18, 8, 3,  10, 9,
                               1, 0, 10, 2,  0, 2, 16, 3, 0, 14, 16, 1, 26, 8,  8};

  vertex_t* expected_src[] = {expected_src_0,
                              expected_src_1,
                              expected_src_2,
                              expected_src_3,
                              expected_src_4,
                              expected_src_5,
                              expected_src_6,
                              expected_src_7};
  vertex_t* expected_dst[] = {expected_dst_0,
                              expected_dst_1,
                              expected_dst_2,
                              expected_dst_3,
                              expected_dst_4,
                              expected_dst_5,
                              expected_dst_6,
                              expected_dst_7};
  size_t expected_len[]    = {sizeof(expected_src_0) / sizeof(expected_src_0[0]),
                              sizeof(expected_src_1) / sizeof(expected_src_1[0]),
                              sizeof(expected_src_2) / sizeof(expected_src_2[0]),
                              sizeof(expected_src_3) / sizeof(expected_src_3[0]),
                              sizeof(expected_src_4) / sizeof(expected_src_4[0]),
                              sizeof(expected_src_5) / sizeof(expected_src_5[0]),
                              sizeof(expected_src_6) / sizeof(expected_src_6[0]),
                              sizeof(expected_src_7) / sizeof(expected_src_7[0])};

  size_t max_rank_to_validate = sizeof(expected_src) / sizeof(expected_src[0]);

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_rng_state_t* rng_state = NULL;
  ;
  cugraph_coo_t* coo = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  ret_code = cugraph_generate_rmat_edgelist(
    handle, rng_state, 5, 30, 0.57, 0.19, 0.19, FALSE, &coo, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "generate_rmat_edgelist failed.");

  if (rank < max_rank_to_validate) {
    cugraph_type_erased_device_array_view_t* src_view;
    cugraph_type_erased_device_array_view_t* dst_view;

    src_view = cugraph_coo_get_sources(coo);
    dst_view = cugraph_coo_get_destinations(coo);

    size_t src_size = cugraph_type_erased_device_array_view_size(src_view);

    vertex_t h_src[src_size];
    vertex_t h_dst[src_size];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_src, src_view, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_dst, dst_view, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_to_host failed.");

    TEST_ASSERT(test_ret_value, src_size == expected_len[rank], "incorrect number of edges");

    for (int i = 0; (i < src_size) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(test_ret_value, expected_src[rank][i] == h_src[i], "generated edges don't match");
      TEST_ASSERT(test_ret_value, expected_dst[rank][i] == h_dst[i], "generated edges don't match");
    }

    cugraph_type_erased_device_array_view_free(dst_view);
    cugraph_type_erased_device_array_view_free(src_view);
  }

  cugraph_coo_free(coo);
  cugraph_error_free(ret_error);

  return test_ret_value;
#endif
}

int test_rmat_list_generation(const cugraph_resource_handle_t* handle)
{
// FIXME: this should be re-implemented (this test doesn't work for more than 8 GPUs, and this
// assumes that the underlying random number generator produces a known sequence)
#if 1
  return CUGRAPH_SUCCESS;
#else
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  size_t num_lists = 3;

  vertex_t expected_src_0_rank_0[] = {29, 16, 16, 5, 6,  1, 20, 5,  22, 14,
                                      3,  12, 4,  0, 10, 0, 4,  16, 20, 16};
  vertex_t expected_dst_0_rank_0[] = {0, 18, 0, 8, 1,  0, 11, 8, 21, 16,
                                      0, 2,  2, 0, 17, 0, 9,  0, 8,  0};
  vertex_t expected_src_1_rank_0[] = {2, 6, 8, 0, 8, 0, 0, 2, 8, 0, 12, 0, 2, 9, 4, 8};
  vertex_t expected_dst_1_rank_0[] = {10, 5, 0, 4, 10, 5, 4, 1, 0, 1, 11, 0, 0, 6, 2, 14};
  vertex_t expected_src_2_rank_0[] = {5, 0, 24, 11, 2, 20, 2, 1,  0, 18,
                                      2, 0, 0,  0,  8, 4,  1, 22, 0, 1};
  vertex_t expected_dst_2_rank_0[] = {16, 4,  0, 1, 8, 0,  0,  0, 8, 7,
                                      3,  15, 0, 0, 4, 26, 16, 0, 8, 2};
  vertex_t expected_src_0_rank_1[] = {0, 21, 10, 17, 8, 0, 21, 24, 0, 0,
                                      0, 1,  2,  16, 0, 2, 2,  24, 0, 27};
  vertex_t expected_dst_0_rank_1[] = {0, 2, 4, 0, 19, 12, 0,  18, 4,  16,
                                      8, 0, 1, 4, 5,  4,  18, 16, 20, 24};
  vertex_t expected_src_1_rank_1[] = {12, 0, 0, 0, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
  vertex_t expected_dst_1_rank_1[] = {12, 8, 1, 0, 8, 0, 1, 0, 6, 13, 0, 0, 0, 0, 0, 2};
  vertex_t expected_src_2_rank_1[] = {4,  12, 0,  0, 1,  12, 2,  0, 16, 0,
                                      20, 8,  14, 0, 16, 9,  24, 8, 10, 1};
  vertex_t expected_dst_2_rank_1[] = {16, 0, 0, 4, 0, 13, 0, 1,  2, 0,
                                      0,  8, 8, 4, 1, 3,  5, 16, 2, 0};
  vertex_t expected_src_0_rank_2[] = {8, 4, 4, 4, 1, 6, 9, 4, 6, 0, 13, 3, 0, 0, 8, 13};
  vertex_t expected_dst_0_rank_2[] = {6, 0, 6, 2, 0, 0, 1, 0, 6, 0, 10, 0, 2, 2, 12, 0};
  vertex_t expected_src_1_rank_2[] = {18, 12, 2, 0, 12, 4,  1, 18, 0, 4,
                                      0,  3,  0, 0, 18, 22, 3, 4,  2, 8};
  vertex_t expected_dst_1_rank_2[] = {2,  1,  8, 22, 1, 28, 4, 11, 0, 0,
                                      12, 24, 8, 3,  8, 6,  1, 16, 0, 2};
  vertex_t expected_src_2_rank_2[] = {1, 2, 12, 3, 1, 0, 4, 5, 1, 3, 5, 3, 1, 2, 8, 0};
  vertex_t expected_dst_2_rank_2[] = {8, 0, 0, 4, 4, 4, 2, 10, 1, 13, 0, 0, 0, 4, 0, 4};
  vertex_t expected_src_0_rank_3[] = {0, 13, 0, 2, 16, 4, 10, 1,  20, 2,
                                      0, 0,  1, 0, 16, 3, 6,  10, 0,  0};
  vertex_t expected_dst_0_rank_3[] = {9, 16, 0, 8, 9,  1, 0,  1, 3,  0,
                                      0, 2,  8, 0, 12, 2, 20, 8, 29, 3};
  vertex_t expected_src_1_rank_3[] = {2,  13, 3, 2,  18, 8,  0, 0, 8,  10,
                                      17, 0,  2, 26, 9,  28, 1, 0, 10, 17};
  vertex_t expected_dst_1_rank_3[] = {2, 8, 0,  14, 16, 20, 8, 8, 0, 0,
                                      7, 0, 16, 4,  20, 16, 5, 3, 0, 3};
  vertex_t expected_src_2_rank_3[] = {0, 1, 3, 0, 9, 1, 5, 0, 4, 4, 0, 6, 0, 12, 0, 2};
  vertex_t expected_dst_2_rank_3[] = {4, 1, 0, 12, 6, 2, 2, 8, 2, 10, 8, 0, 4, 4, 0, 1};
  vertex_t expected_src_0_rank_4[] = {0, 1, 1, 0, 0, 1, 4, 3, 14, 3, 0, 4, 0, 2, 9, 0};
  vertex_t expected_dst_0_rank_4[] = {0, 2, 8, 0, 2, 0, 13, 0, 0, 8, 1, 12, 0, 9, 0, 0};
  vertex_t expected_src_1_rank_4[] = {2, 2, 2, 6, 4, 6, 4, 10, 0, 2, 2, 8, 4, 0, 4, 9};
  vertex_t expected_dst_1_rank_4[] = {0, 1, 6, 1, 1, 0, 10, 2, 0, 0, 0, 0, 0, 1, 0, 0};
  vertex_t expected_src_2_rank_4[] = {9,  2, 0,  20, 0,  3, 5, 22, 2, 8,
                                      26, 0, 18, 16, 12, 0, 0, 18, 8, 2};
  vertex_t expected_dst_2_rank_4[] = {2, 8, 6,  18, 16, 16, 19, 0, 16, 1,
                                      0, 0, 12, 0,  8,  0,  0,  8, 22, 4};
  vertex_t expected_src_0_rank_5[] = {8, 11, 4, 4, 8, 0, 8, 15, 12, 6, 1, 0, 7, 4, 3, 2};
  vertex_t expected_dst_0_rank_5[] = {5, 4, 12, 8, 4, 8, 8, 1, 0, 3, 0, 8, 8, 0, 1, 5};
  vertex_t expected_src_1_rank_5[] = {16, 0, 8, 17, 9,  8,  2,  1, 0, 24,
                                      8,  2, 0, 0,  16, 22, 16, 0, 0, 2};
  vertex_t expected_dst_1_rank_5[] = {8, 10, 2,  0,  0, 6, 8, 2, 8, 16,
                                      1, 0,  24, 26, 0, 0, 2, 3, 8, 24};
  vertex_t expected_src_2_rank_5[] = {21, 5, 10, 24, 1, 16, 10, 0, 2, 2,
                                      5,  4, 2,  0,  0, 8,  2,  0, 4, 4};
  vertex_t expected_dst_2_rank_5[] = {4,  14, 20, 0, 0,  8, 20, 1,  8, 21,
                                      16, 2,  0,  2, 14, 0, 10, 24, 9, 5};
  vertex_t expected_src_0_rank_6[] = {8, 16, 2,  8, 0,  24, 20, 10, 8,  0,
                                      0, 0,  20, 8, 10, 0,  4,  8,  12, 4};
  vertex_t expected_dst_0_rank_6[] = {21, 25, 6, 6, 2,  0, 12, 1, 8,  8,
                                      16, 28, 8, 1, 10, 0, 1,  0, 16, 2};
  vertex_t expected_src_1_rank_6[] = {4, 0, 1, 13, 0, 1, 10, 8, 2, 0, 9, 8, 3, 6, 4, 12};
  vertex_t expected_dst_1_rank_6[] = {9, 2, 5, 6, 8, 0, 4, 0, 0, 10, 2, 6, 14, 0, 0, 2};
  vertex_t expected_src_2_rank_6[] = {2,  4,  18, 18, 0,  24, 4,  16, 5,  18,
                                      17, 13, 16, 28, 24, 0,  28, 0,  13, 1};
  vertex_t expected_dst_2_rank_6[] = {8,  24, 24, 1, 16, 2,  0, 4,  1, 0,
                                      10, 10, 18, 0, 18, 18, 2, 26, 8, 0};
  vertex_t expected_src_0_rank_7[] = {0, 5, 0,  4,  2, 3, 4, 4, 20, 18,
                                      0, 0, 29, 17, 0, 0, 0, 1, 0,  8};
  vertex_t expected_dst_0_rank_7[] = {4, 0,  24, 0, 0,  18, 8, 0,  0,  9,
                                      3, 12, 4,  8, 17, 1,  0, 28, 22, 1};
  vertex_t expected_src_1_rank_7[] = {0, 8, 5, 2, 4, 0, 0, 5, 0, 0, 0, 9, 2, 0, 2, 8};
  vertex_t expected_dst_1_rank_7[] = {3, 0, 1, 9, 9, 8, 7, 12, 1, 12, 5, 0, 0, 8, 13, 8};
  vertex_t expected_src_2_rank_7[] = {8, 9, 2, 4, 4, 0, 25, 0, 0,  0,
                                      8, 6, 0, 0, 8, 4, 21, 0, 16, 0};
  vertex_t expected_dst_2_rank_7[] = {4, 4,  1,  1, 4,  0, 16, 1, 0, 16,
                                      5, 25, 24, 8, 16, 2, 2,  0, 4, 16};

  vertex_t* expected_src_0[] = {expected_src_0_rank_0,
                                expected_src_0_rank_1,
                                expected_src_0_rank_2,
                                expected_src_0_rank_3,
                                expected_src_0_rank_4,
                                expected_src_0_rank_5,
                                expected_src_0_rank_6,
                                expected_src_0_rank_7};
  vertex_t* expected_dst_0[] = {expected_dst_0_rank_0,
                                expected_dst_0_rank_1,
                                expected_dst_0_rank_2,
                                expected_dst_0_rank_3,
                                expected_dst_0_rank_4,
                                expected_dst_0_rank_5,
                                expected_dst_0_rank_6,
                                expected_dst_0_rank_7};
  vertex_t* expected_src_1[] = {expected_src_1_rank_0,
                                expected_src_1_rank_1,
                                expected_src_1_rank_2,
                                expected_src_1_rank_3,
                                expected_src_1_rank_4,
                                expected_src_1_rank_5,
                                expected_src_1_rank_6,
                                expected_src_1_rank_7};
  vertex_t* expected_dst_1[] = {expected_dst_1_rank_0,
                                expected_dst_1_rank_1,
                                expected_dst_1_rank_2,
                                expected_dst_1_rank_3,
                                expected_dst_1_rank_4,
                                expected_dst_1_rank_5,
                                expected_dst_1_rank_6,
                                expected_dst_1_rank_7};
  vertex_t* expected_src_2[] = {expected_src_2_rank_0,
                                expected_src_2_rank_1,
                                expected_src_2_rank_2,
                                expected_src_2_rank_3,
                                expected_src_2_rank_4,
                                expected_src_2_rank_5,
                                expected_src_2_rank_6,
                                expected_src_2_rank_7};
  vertex_t* expected_dst_2[] = {expected_dst_2_rank_0,
                                expected_dst_2_rank_1,
                                expected_dst_2_rank_2,
                                expected_dst_2_rank_3,
                                expected_dst_2_rank_4,
                                expected_dst_2_rank_5,
                                expected_dst_2_rank_6,
                                expected_dst_2_rank_7};

  size_t expected_len_0[] = {sizeof(expected_src_0_rank_0) / sizeof(expected_src_0_rank_0[0]),
                             sizeof(expected_src_0_rank_1) / sizeof(expected_src_0_rank_1[0]),
                             sizeof(expected_src_0_rank_2) / sizeof(expected_src_0_rank_2[0]),
                             sizeof(expected_src_0_rank_3) / sizeof(expected_src_0_rank_3[0]),
                             sizeof(expected_src_0_rank_4) / sizeof(expected_src_0_rank_4[0]),
                             sizeof(expected_src_0_rank_5) / sizeof(expected_src_0_rank_5[0]),
                             sizeof(expected_src_0_rank_6) / sizeof(expected_src_0_rank_6[0]),
                             sizeof(expected_src_0_rank_7) / sizeof(expected_src_0_rank_7[0])};

  size_t expected_len_1[] = {sizeof(expected_src_1_rank_0) / sizeof(expected_src_1_rank_0[0]),
                             sizeof(expected_src_1_rank_1) / sizeof(expected_src_1_rank_1[0]),
                             sizeof(expected_src_1_rank_2) / sizeof(expected_src_1_rank_2[0]),
                             sizeof(expected_src_1_rank_3) / sizeof(expected_src_1_rank_3[0]),
                             sizeof(expected_src_1_rank_4) / sizeof(expected_src_1_rank_4[0]),
                             sizeof(expected_src_1_rank_5) / sizeof(expected_src_1_rank_5[0]),
                             sizeof(expected_src_1_rank_6) / sizeof(expected_src_1_rank_6[0]),
                             sizeof(expected_src_1_rank_7) / sizeof(expected_src_1_rank_7[0])};

  size_t expected_len_2[] = {sizeof(expected_src_2_rank_0) / sizeof(expected_src_2_rank_0[0]),
                             sizeof(expected_src_2_rank_1) / sizeof(expected_src_2_rank_1[0]),
                             sizeof(expected_src_2_rank_2) / sizeof(expected_src_2_rank_2[0]),
                             sizeof(expected_src_2_rank_3) / sizeof(expected_src_2_rank_3[0]),
                             sizeof(expected_src_2_rank_4) / sizeof(expected_src_2_rank_4[0]),
                             sizeof(expected_src_2_rank_5) / sizeof(expected_src_2_rank_5[0]),
                             sizeof(expected_src_2_rank_6) / sizeof(expected_src_2_rank_6[0]),
                             sizeof(expected_src_2_rank_7) / sizeof(expected_src_2_rank_7[0])};

  vertex_t** expected_src[] = {expected_src_0, expected_src_1, expected_src_2};
  vertex_t** expected_dst[] = {expected_dst_0, expected_dst_1, expected_dst_2};
  size_t* expected_len[]    = {expected_len_0, expected_len_1, expected_len_2};

  size_t max_rank_to_validate = sizeof(expected_src_0) / sizeof(expected_src_0[0]);

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_rng_state_t* rng_state = NULL;
  ;
  cugraph_coo_list_t* coo_list = NULL;

  int rank = cugraph_resource_handle_get_rank(handle);

  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t edge_tid   = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  ret_code = cugraph_generate_rmat_edgelists(
    handle, rng_state, num_lists, 4, 6, 4, UNIFORM, POWER_LAW, FALSE, &coo_list, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "generate_rmat_edgelist failed.");

  TEST_ASSERT(test_ret_value,
              cugraph_coo_list_size(coo_list) == num_lists,
              "generated wrong number of results");

  if (rank < max_rank_to_validate) {
    for (size_t i = 0; (i < num_lists) && (test_ret_value == 0); i++) {
      cugraph_coo_t* coo = NULL;

      coo = cugraph_coo_list_element(coo_list, i);

      cugraph_type_erased_device_array_view_t* src_view;
      cugraph_type_erased_device_array_view_t* dst_view;

      src_view = cugraph_coo_get_sources(coo);
      dst_view = cugraph_coo_get_destinations(coo);

      size_t src_size = cugraph_type_erased_device_array_view_size(src_view);

      TEST_ASSERT(test_ret_value, src_size == expected_len[i][rank], "wrong number of edges");

      vertex_t h_src[src_size];
      vertex_t h_dst[src_size];

      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        handle, (byte_t*)h_src, src_view, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_to_host failed.");

      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        handle, (byte_t*)h_dst, dst_view, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_to_host failed.");

      for (size_t j = 0; (j < src_size) && (test_ret_value == 0); ++j) {
        TEST_ASSERT(
          test_ret_value, expected_src[i][rank][j] == h_src[j], "generated edges don't match");
        TEST_ASSERT(
          test_ret_value, expected_dst[i][rank][j] == h_dst[j], "generated edges don't match");
      }

      cugraph_type_erased_device_array_view_free(dst_view);
      cugraph_type_erased_device_array_view_free(src_view);
    }
  }

  cugraph_coo_list_free(coo_list);
  cugraph_error_free(ret_error);

  return test_ret_value;
#endif
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_rmat_generation, handle);
  result |= RUN_MG_TEST(test_rmat_list_generation, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
