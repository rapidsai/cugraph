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

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/cugraph_api.h>
#include <stdio.h>

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int test_create_sg_graph_simple()
{
  typedef int32_t vertex_t;
  typedef int32_t edge_t;
  typedef float weight_t;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  size_t num_edges         = 8;
  size_t num_vertices      = 6;

  vertex_t h_src[] = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[] = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[] = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_device_buffer_t dbuf_src;
  cugraph_device_buffer_t dbuf_dst;
  cugraph_device_buffer_t dbuf_wgt;
  cugraph_graph_envelope_t* p_graph_envelope = NULL;

  data_type_id_t vertex_tid = INT32;
  data_type_id_t edge_tid   = INT32;
  data_type_id_t weight_tid = FLOAT32;

  p_handle = cugraph_create_resource_handle(NULL);
  runtime_assert(p_handle != NULL, "resource handle creation failed.");

  ret_code = cugraph_make_device_buffer(p_handle, vertex_tid, num_edges, &dbuf_src);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "src device_buffer creation failed.");

  ret_code = cugraph_make_device_buffer(p_handle, vertex_tid, num_edges, &dbuf_dst);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "dst device_buffer creation failed.");

  ret_code = cugraph_make_device_buffer(p_handle, weight_tid, num_edges, &dbuf_wgt);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "weight device_buffer creation failed.");

  ret_code = cugraph_update_device_buffer(p_handle, vertex_tid, &dbuf_src, (byte_t*)h_src);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "src device_buffer update failed.");

  ret_code = cugraph_update_device_buffer(p_handle, vertex_tid, &dbuf_dst, (byte_t*)h_dst);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "dst device_buffer update failed.");

  ret_code = cugraph_update_device_buffer(p_handle, weight_tid, &dbuf_wgt, (byte_t*)h_wgt);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "weight device_buffer update failed.");

  p_graph_envelope = cugraph_make_sg_graph(p_handle,
                                           vertex_tid,
                                           edge_tid,
                                           weight_tid,
                                           FALSE,
                                           &dbuf_src,
                                           &dbuf_dst,
                                           &dbuf_wgt,
                                           num_vertices,
                                           num_edges,
                                           FALSE,
                                           FALSE,
                                           FALSE);
  runtime_assert(p_graph_envelope != NULL, "graph envelope creation failed.");

  cugraph_free_graph(p_graph_envelope);

  cugraph_free_device_buffer(&dbuf_wgt);

  cugraph_free_device_buffer(&dbuf_dst);

  cugraph_free_device_buffer(&dbuf_src);

  cugraph_free_resource_handle(p_handle);

  return 0;
}

/*
 * Since cugraph_make_sg_graph() can return NULL, this ensures
 * cugraph_free_graph() can accept NULL.
 */
int test_free_graph_NULL_ptr()
{
  /* Returns void, so check that the call does not crash. */
  cugraph_free_graph((cugraph_graph_envelope_t*)NULL);
  return 0;
}

/*
 * Test creating a graph with NULL device arrays and "expensive check" enabled.
 */
int test_create_sg_graph_bad_arrays()
{
  int test_failed = 0;

  cugraph_graph_envelope_t* G = NULL;
  cugraph_resource_handle_t handle;
  cugraph_device_buffer_t* src_ptr     = NULL;
  cugraph_device_buffer_t* dst_ptr     = NULL;
  cugraph_device_buffer_t* weights_ptr = NULL;
  size_t num_verts                     = 4;
  size_t num_edges                     = 3;
  bool_t do_expensive_check            = 1;
  bool_t store_transposed              = 0;
  bool_t is_symmetric                  = 0;
  bool_t is_multigraph                 = 0;

  G = cugraph_make_sg_graph(&handle,
                            INT32,
                            INT32,
                            INT32, /* vert, edge, weight types */
                            store_transposed,
                            src_ptr,
                            dst_ptr,
                            weights_ptr,
                            num_verts,
                            num_edges,
                            do_expensive_check,
                            is_symmetric,
                            is_multigraph);

  if (G != NULL) { test_failed = 1; }

  return test_failed;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_create_sg_graph_simple);
  result |= RUN_TEST(test_free_graph_NULL_ptr);
  result |= RUN_TEST(test_create_sg_graph_bad_arrays);
  return result;
}
