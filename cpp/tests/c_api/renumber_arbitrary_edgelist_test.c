/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "cugraph_c/array.h"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;

int generic_renumber_arbitrary_edgelist_test(vertex_t* h_src,
                                             vertex_t* h_dst,
                                             vertex_t* h_renumber_map,
                                             size_t num_edges,
                                             size_t renumber_map_size)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  cugraph_type_erased_device_array_t* srcs;
  cugraph_type_erased_device_array_t* dsts;
  cugraph_type_erased_device_array_view_t* srcs_view;
  cugraph_type_erased_device_array_view_t* dsts_view;
  cugraph_type_erased_host_array_view_t* renumber_map_view;

  ret_code = cugraph_type_erased_device_array_create(p_handle, num_edges, INT32, &srcs, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "srcs create failed.");

  ret_code = cugraph_type_erased_device_array_create(p_handle, num_edges, INT32, &dsts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dsts create failed.");

  srcs_view = cugraph_type_erased_device_array_view(srcs);
  dsts_view = cugraph_type_erased_device_array_view(dsts);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, srcs_view, (byte_t*)h_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dsts_view, (byte_t*)h_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  renumber_map_view =
    cugraph_type_erased_host_array_view_create(h_renumber_map, renumber_map_size, INT32);

  ret_code = cugraph_renumber_arbitrary_edgelist(
    p_handle, renumber_map_view, srcs_view, dsts_view, &ret_error);

  vertex_t h_renumbered_srcs[num_edges];
  vertex_t h_renumbered_dsts[num_edges];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_renumbered_srcs, srcs_view, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_renumbered_dsts, dsts_view, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_edges) && (test_ret_value == 0); ++i) {
    vertex_t renumbered_src = -1;
    vertex_t renumbered_dst = -1;

    for (size_t j = 0; (j < renumber_map_size) && ((renumbered_src < 0) || (renumbered_dst < 0));
         ++j) {
      if (h_src[i] == h_renumber_map[j]) renumbered_src = (vertex_t)j;
      if (h_dst[i] == h_renumber_map[j]) renumbered_dst = (vertex_t)j;
    }

    TEST_ASSERT(test_ret_value, h_renumbered_srcs[i] == renumbered_src, "src results don't match");
    TEST_ASSERT(test_ret_value, h_renumbered_dsts[i] == renumbered_dst, "dst results don't match");
  }

  cugraph_type_erased_device_array_free(dsts);
  cugraph_type_erased_device_array_free(srcs);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_renumbering()
{
  size_t num_edges         = 8;
  size_t renumber_map_size = 6;

  vertex_t h_src[]          = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]          = {1, 3, 4, 0, 1, 3, 5, 5};
  vertex_t h_renumber_map[] = {5, 3, 1, 2, 4, 0};

  return generic_renumber_arbitrary_edgelist_test(
    h_src, h_dst, h_renumber_map, num_edges, renumber_map_size);
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_renumbering);
  return result;
}
