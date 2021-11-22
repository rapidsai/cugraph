/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_bfs_test_with_extract_paths(vertex_t* h_src,
                                        vertex_t* h_dst,
                                        weight_t* h_wgt,
                                        vertex_t* h_seeds,
                                        vertex_t* h_destinations,
                                        vertex_t expected_max_path_length,
                                        vertex_t const* expected_paths,
                                        size_t num_vertices,
                                        size_t num_edges,
                                        size_t num_seeds,
                                        size_t num_destinations,
                                        size_t depth_limit,
                                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle                    = NULL;
  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* p_paths_result                 = NULL;
  cugraph_extract_paths_result_t* p_extract_paths_result = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_t* p_destinations     = NULL;

  p_handle = cugraph_create_handle();
  TEST_ASSERT(test_ret_value, p_handle != NULL, "raft handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, &p_graph, &ret_error);

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, INT32, num_seeds, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  ret_code = cugraph_type_erased_device_array_copy_from_host(
    p_handle, p_sources, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, INT32, num_destinations, &p_destinations, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_destinations create failed.");

  ret_code = cugraph_type_erased_device_array_copy_from_host(
    p_handle, p_destinations, (byte_t*)h_destinations, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_bfs(
    p_handle, p_graph, p_sources, FALSE, depth_limit, FALSE, TRUE, &p_paths_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs failed.");

  ret_code = cugraph_extract_paths(p_handle,
                                   p_graph,
                                   p_sources,
                                   p_paths_result,
                                   p_destinations,
                                   &p_extract_paths_result,
                                   &ret_error);

  size_t max_path_length = cugraph_extract_paths_result_get_max_path_length(p_extract_paths_result);
  cugraph_type_erased_device_array_t* paths =
    cugraph_extract_paths_result_get_paths(p_extract_paths_result);
  cugraph_extract_paths_result_free(p_extract_paths_result);

  size_t paths_size = cugraph_type_erased_device_array_size(paths);
  vertex_t h_paths[paths_size];

  ret_code =
    cugraph_type_erased_device_array_copy_to_host(p_handle, (byte_t*)h_paths, paths, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < paths_size) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value, expected_paths[i] == h_paths[i], "paths don't match");
  }

  cugraph_type_erased_device_array_free(p_sources);
  cugraph_paths_result_free(p_paths_result);
  cugraph_sg_graph_free(p_graph);
  cugraph_free_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs_with_extract_paths()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                  = {0};
  vertex_t destinations[]           = {2, 5};
  vertex_t expected_max_path_length = 4;
  vertex_t expected_paths[]         = {0, 1, 3, 5, 0, 1, 4, 5};

  // Bfs wants store_transposed = FALSE
  return generic_bfs_test_with_extract_paths(src,
                                             dst,
                                             wgt,
                                             seeds,
                                             destinations,
                                             expected_max_path_length,
                                             expected_paths,
                                             num_vertices,
                                             num_edges,
                                             1,
                                             2,
                                             10,
                                             FALSE);
}

int test_bfs_with_extract_paths_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                  = {5};
  vertex_t destinations[]           = {0, 2};
  vertex_t expected_max_path_length = 4;
  vertex_t expected_paths[]         = {5, 3, 1, 0, 5, 4, 1, 0};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_bfs to transpose the graph
  return generic_bfs_test_with_extract_paths(src,
                                             dst,
                                             wgt,
                                             seeds,
                                             destinations,
                                             expected_max_path_length,
                                             expected_paths,
                                             num_vertices,
                                             num_edges,
                                             1,
                                             2,
                                             10,
                                             TRUE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_bfs_with_extract_paths);
  result |= RUN_TEST(test_bfs_with_extract_paths_with_transpose);
  return result;
}
