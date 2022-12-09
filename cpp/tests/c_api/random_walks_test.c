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

#define NUM_PATHS 2
#define MAX_DEPTH 4

/* sample graph:
    0 --(.1)--> 1 --(1.1)--> 4
   /|\       /\ |            |
    |       /   |            |
   (5.1) (3.1)(2.1)        (3.2)
    |   /       |            |
    | /        \|/          \|/
    2 --(4.1)-->3 --(7.2)--> 5
*/

/* positive test RW call flow*/
int test_random_walks_1()
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

  /* RW args:*/
  cugraph_rw_ret_t rw_ret;
  cugraph_device_buffer_t dbuf_start;
  size_t num_paths                          = 2;
  size_t max_depth                          = 4;
  bool_t flag_use_padding                   = FALSE;
  cugraph_unique_ptr_t* p_sampling_strategy = NULL;
  vertex_t h_start[]                        = {0, 2};

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

  ret_code = cugraph_make_device_buffer(p_handle, vertex_tid, num_paths, &dbuf_start);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "start device_buffer creation failed.");

  ret_code = cugraph_update_device_buffer(p_handle, vertex_tid, &dbuf_start, (byte_t*)h_start);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "start device_buffer update failed.");

  p_sampling_strategy = cugraph_create_sampling_strategy(0, 0.0, 0.0);
  runtime_assert(p_sampling_strategy != NULL, "start device_buffer update failed.");

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

  ret_code = cugraph_random_walks(p_handle,
                                  p_graph_envelope,
                                  &dbuf_start,
                                  num_paths,
                                  max_depth,
                                  flag_use_padding,
                                  p_sampling_strategy,
                                  &rw_ret);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "cugraph_random_walks() failed.");

  cugraph_free_rw_result(&rw_ret);

  cugraph_free_graph(p_graph_envelope);

  cugraph_free_sampling_strategy(p_sampling_strategy);

  cugraph_free_device_buffer(&dbuf_start);

  cugraph_free_device_buffer(&dbuf_wgt);

  cugraph_free_device_buffer(&dbuf_dst);

  cugraph_free_device_buffer(&dbuf_src);

  cugraph_free_resource_handle(p_handle);

  return 0;
}

/* array of compressed paths,
   using 1-based indexing for vertices,
   to avoid confusion between, for example,
   `012` and `12`, which result in same number*/
#define NUM_MAX_PATHS 30
static int32_t c_ps_array[NUM_MAX_PATHS] = {
  1,  2,  3,  4,   5,    6,    12,    124,   125, 1246, 1256, 24,   25,  246, 256,
  31, 32, 34, 312, 3124, 3125, 31246, 31256, 324, 325,  3246, 3256, 346, 46,  56};

/* linear search of `value` inside `p_cmprsd_path[max_num_paths]`*/
bool_t is_one_of(int32_t value, int32_t* p_cmprsd_path, int max_num_paths)
{
  int i = 0;
  for (; i < max_num_paths; ++i)
    if (value == p_cmprsd_path[i]) return 1;

  return 0;
}

/* check on host if all obtained paths are possible paths */
bool_t host_check_paths(int32_t* p_path_v, int32_t* p_path_sz, int num_paths)
{
  int i            = 0;
  int count_passed = 0;

  for (; i < num_paths; ++i) {
    int32_t crt_path_sz          = p_path_sz[i];
    int path_it                  = 0;
    int32_t crt_path_accumulator = 0;
    bool_t flag_passed           = 0;

    for (; path_it < crt_path_sz; ++path_it) {
      crt_path_accumulator =
        (*p_path_v + 1) + 10 * crt_path_accumulator; /* 1-based indexing for vertices is necessary
                                                        to avoid ambiguity, hence the `+1`*/
      ++p_path_v;                                    /* iterate p_path_v*/
    }

    flag_passed = is_one_of(crt_path_accumulator, c_ps_array, NUM_MAX_PATHS);
    if (flag_passed) ++count_passed;
  }

  return (count_passed == num_paths);
}

/* check RW call results*/
int test_random_walks_2()
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

  /* RW args:*/
  cugraph_rw_ret_t rw_ret;
  cugraph_device_buffer_t dbuf_start;
  cugraph_device_buffer_t dbuf_rw_ret_v;
  cugraph_device_buffer_t dbuf_rw_ret_sz;

  size_t num_paths                          = NUM_PATHS;
  size_t max_depth                          = MAX_DEPTH;
  bool_t flag_use_padding                   = FALSE;
  cugraph_unique_ptr_t* p_sampling_strategy = NULL;
  bool_t flag_passed                        = FALSE;
  vertex_t h_start[]                        = {0, 2};
  edge_t h_sizes[NUM_PATHS];
  vertex_t h_paths[NUM_PATHS * MAX_DEPTH];

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

  ret_code = cugraph_make_device_buffer(p_handle, vertex_tid, num_paths, &dbuf_start);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "start device_buffer creation failed.");

  ret_code = cugraph_update_device_buffer(p_handle, vertex_tid, &dbuf_start, (byte_t*)h_start);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "start device_buffer update failed.");

  p_sampling_strategy = cugraph_create_sampling_strategy(0, 0.0, 0.0);
  runtime_assert(p_sampling_strategy != NULL, "sampling strategy creation failed.");

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

  ret_code = cugraph_random_walks(p_handle,
                                  p_graph_envelope,
                                  &dbuf_start,
                                  num_paths,
                                  max_depth,
                                  flag_use_padding,
                                  p_sampling_strategy,
                                  &rw_ret);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "cugraph_random_walks() failed.");

  extract_size_rw_result(&rw_ret, &dbuf_rw_ret_sz);

  ret_code = cugraph_update_host_buffer(p_handle, edge_tid, (byte_t*)h_sizes, &dbuf_rw_ret_sz);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "size host buffer update failed.");

  extract_vertex_rw_result(&rw_ret, &dbuf_rw_ret_v);

  ret_code = cugraph_update_host_buffer(p_handle, vertex_tid, (byte_t*)h_paths, &dbuf_rw_ret_v);
  runtime_assert(ret_code == CUGRAPH_SUCCESS, "paths host buffer update failed.");

  flag_passed = host_check_paths(h_paths, h_sizes, num_paths);
  runtime_assert(flag_passed == TRUE, "paths check failed.");

  cugraph_free_rw_result(&rw_ret);

  cugraph_free_graph(p_graph_envelope);

  cugraph_free_sampling_strategy(p_sampling_strategy);

  cugraph_free_device_buffer(&dbuf_start);

  cugraph_free_device_buffer(&dbuf_wgt);

  cugraph_free_device_buffer(&dbuf_dst);

  cugraph_free_device_buffer(&dbuf_src);

  cugraph_free_resource_handle(p_handle);

  return 0;
}

/* negative test RW call flow*/
int test_random_walks_3()
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

  /* RW args:*/
  cugraph_rw_ret_t rw_ret;
  cugraph_device_buffer_t dbuf_start;
  size_t num_paths                          = 2;
  size_t max_depth                          = 4;
  bool_t flag_use_padding                   = FALSE;
  cugraph_unique_ptr_t* p_sampling_strategy = NULL;
  bool_t flag_failed                        = FALSE;

  /* purposely erroneous start buffer*/
  dbuf_start.data_ = NULL;
  dbuf_start.size_ = 0;

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

  p_sampling_strategy = cugraph_create_sampling_strategy(0, 0.0, 0.0);
  runtime_assert(p_sampling_strategy != NULL, "sampling strategy creation failed.");

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

  ret_code    = cugraph_random_walks(p_handle,
                                  p_graph_envelope,
                                  &dbuf_start,
                                  num_paths,
                                  max_depth,
                                  flag_use_padding,
                                  p_sampling_strategy,
                                  &rw_ret);
  flag_failed = (ret_code != CUGRAPH_SUCCESS);
  runtime_assert(flag_failed == TRUE, "cugraph_random_walks() should have failed.");

  cugraph_free_rw_result(&rw_ret);

  cugraph_free_graph(p_graph_envelope);

  cugraph_free_sampling_strategy(p_sampling_strategy);

  cugraph_free_device_buffer(&dbuf_start);

  cugraph_free_device_buffer(&dbuf_wgt);

  cugraph_free_device_buffer(&dbuf_dst);

  cugraph_free_device_buffer(&dbuf_src);

  cugraph_free_resource_handle(p_handle);

  /* because bools and err return codes from programs have opposite meaning...*/
  if (flag_failed)
    return 0;
  else
    return 1;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_random_walks_1);
  result |= RUN_TEST(test_random_walks_2);
  result |= RUN_TEST(test_random_walks_3);
  return result;
}
