/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include "c_test_utils.h"

#include <math.h>

extern "C" int nearlyEqual(float a, float b, float epsilon)
{
  // FIXME:  There is a better test than this,
  //   perhaps use the gtest comparison for consistency
  //   with C++ and wrap it in a C wrapper.
  return (fabsf(a - b) <= (((fabsf(a) < fabsf(b)) ? fabs(b) : fabs(a)) * epsilon));
}

extern "C" int nearlyEqualDouble(double a, double b, double epsilon)
{
  // FIXME:  There is a better test than this,
  //   perhaps use the gtest comparison for consistency
  //   with C++ and wrap it in a C wrapper.
  return (fabsf(a - b) <= (((fabsf(a) < fabsf(b)) ? fabs(b) : fabs(a)) * epsilon));
}

/*
 * Simple check of creating a graph from a COO on device memory.
 */
extern "C" int create_test_graph(const cugraph_resource_handle_t* p_handle,
                                 int32_t* h_src,
                                 int32_t* h_dst,
                                 float* h_wgt,
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
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, weight_tid, &wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_sg_graph_create(p_handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     nullptr,
                                     nullptr,
                                     store_transposed,
                                     renumber,
                                     FALSE,
                                     p_graph,
                                     ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

extern "C" int create_test_graph_double(const cugraph_resource_handle_t* p_handle,
                                        int32_t* h_src,
                                        int32_t* h_dst,
                                        double* h_wgt,
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
  data_type_id_t weight_tid = FLOAT64;

  cugraph_type_erased_device_array_t* src;
  cugraph_type_erased_device_array_t* dst;
  cugraph_type_erased_device_array_t* wgt;
  cugraph_type_erased_device_array_view_t* src_view;
  cugraph_type_erased_device_array_view_t* dst_view;
  cugraph_type_erased_device_array_view_t* wgt_view;

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src create failed.");
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(*ret_error));

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, vertex_tid, &dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst create failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_edges, weight_tid, &wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt create failed.");

  src_view = cugraph_type_erased_device_array_view(src);
  dst_view = cugraph_type_erased_device_array_view(dst);
  wgt_view = cugraph_type_erased_device_array_view(wgt);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, src_view, (byte_t*)h_src, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, dst_view, (byte_t*)h_dst, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "dst copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, wgt_view, (byte_t*)h_wgt, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "wgt copy_from_host failed.");

  ret_code = cugraph_sg_graph_create(p_handle,
                                     &properties,
                                     src_view,
                                     dst_view,
                                     wgt_view,
                                     nullptr,
                                     nullptr,
                                     store_transposed,
                                     renumber,
                                     FALSE,
                                     p_graph,
                                     ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  cugraph_type_erased_device_array_view_free(wgt_view);
  cugraph_type_erased_device_array_view_free(dst_view);
  cugraph_type_erased_device_array_view_free(src_view);
  cugraph_type_erased_device_array_free(wgt);
  cugraph_type_erased_device_array_free(dst);
  cugraph_type_erased_device_array_free(src);

  return test_ret_value;
}

/*
 * Runs the function pointed to by "test" and returns the return code.  Also
 * prints reporting info (using "test_name"): pass/fail and run time, to stdout.
 *
 * Intended to be used by the RUN_TEST macro.
 */
extern "C" int run_sg_test(int (*test)(), const char* test_name)
{
  int ret_val = 0;
  time_t start_time, end_time;

  printf("RUNNING: %s...", test_name);
  fflush(stdout);

  time(&start_time);

  ret_val = test();

  time(&end_time);

  printf("done (%f seconds).", difftime(end_time, start_time));
  if (ret_val == 0) {
    printf(" - passed\n");
  } else {
    printf(" - FAILED\n");
  }
  fflush(stdout);

  return ret_val;
}
