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

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/graph_generators.h>

#include <stdio.h>

/*
 * Simple rmat generator test
 */
int test_rmat_generation()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  vertex_t expected_src[] =  { 17, 18, 0, 16, 1, 24, 16, 1, 6, 4, 2, 1, 14, 2, 16, 2, 5, 23, 4, 10, 4, 3, 0, 4, 11, 0, 0, 2, 24, 0};
  vertex_t expected_dst[] = { 0, 10, 23, 0, 26, 0, 2, 1, 27, 8, 1, 0, 21, 21, 0, 4, 8, 14, 10, 17, 0, 16, 0, 16, 25, 5, 8, 8, 4, 19}; 

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle = NULL;
  cugraph_rng_state_t *rng_state    = NULL;;
  cugraph_coo_t *coo = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
 
  ret_code = cugraph_generate_rmat_edgelist(handle,
                                            rng_state,
                                            5,
                                            30,
                                            0.57,
                                            0.19,
                                            0.19,
                                            FALSE,
                                            FALSE,
                                            &coo,
                                            &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "generate_rmat_edgelist failed.");

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

  for (int i = 0 ; i < src_size ; ++i) {
    TEST_ASSERT(test_ret_value,
                expected_src[i] == h_src[i],
                "generated edges don't match");
    TEST_ASSERT(test_ret_value,
                expected_dst[i] == h_dst[i],
                "generated edges don't match");
  }

  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_coo_free(coo);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_rmat_list_generation()
{
  int test_ret_value = 0;

  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* handle = NULL;
  cugraph_rng_state_t *rng_state    = NULL;;
  cugraph_coo_list_t *coo_list = NULL;

  handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");

  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  //
  // NOTE: We can't exactly compare results for functions that make multiple RNG calls
  // within them.  When the RNG state is advanced, it is advanced by a multiple of
  // the number of possible threads involved, not based on how many of the values
  // were actually used.  So different GPU versions will result in subtly different
  // random sequences.
  //
  size_t   num_lists = 3;
  vertex_t max_vertex_id[] = { 32, 16, 32 };
  size_t   expected_len[]  = { 20, 16, 20 };
  
  ret_code = cugraph_rng_state_create(handle, 0, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_generate_rmat_edgelists(handle,
                                             rng_state,
                                             num_lists,
                                             4,
                                             6,
                                             4,
                                             UNIFORM,
                                             POWER_LAW,
                                             FALSE,
                                             FALSE,
                                             &coo_list,
                                             &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "generate_rmat_edgelist failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  TEST_ASSERT(test_ret_value, cugraph_coo_list_size(coo_list) == num_lists, "generated wrong number of results");

  for (size_t i = 0 ; (i < num_lists) && (test_ret_value == 0) ; i++) {
    cugraph_coo_t *coo = NULL;

    coo = cugraph_coo_list_element(coo_list, i);

    cugraph_type_erased_device_array_view_t* src_view;
    cugraph_type_erased_device_array_view_t* dst_view;

    src_view = cugraph_coo_get_sources(coo);
    dst_view = cugraph_coo_get_destinations(coo);

    size_t src_size = cugraph_type_erased_device_array_view_size(src_view);

    TEST_ASSERT(test_ret_value, src_size == expected_len[i], "wrong number of edges");

    vertex_t h_src[src_size];
    vertex_t h_dst[src_size];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_src, src_view, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      handle, (byte_t*)h_dst, dst_view, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_to_host failed.");

    for (size_t j = 0 ; (j < src_size) && (test_ret_value == 0) ; ++j) {
      TEST_ASSERT(test_ret_value,
                  h_src[j] < max_vertex_id[i],
                  "generated edges don't match");
      TEST_ASSERT(test_ret_value,
                  h_dst[j] < max_vertex_id[i],
                  "generated edges don't match");
    }

    cugraph_type_erased_device_array_view_free(dst_view);
    cugraph_type_erased_device_array_view_free(src_view);
  }

  cugraph_coo_list_free(coo_list);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_rmat_generation);
  result |= RUN_TEST(test_rmat_list_generation);
  return result;
}
