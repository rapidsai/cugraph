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
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

const weight_t EPSILON = 0.001;

int generic_node2vec_test(vertex_t* h_src,
                          vertex_t* h_dst,
                          weight_t* h_wgt,
                          vertex_t* h_seeds,
                          size_t num_vertices,
                          size_t num_edges,
                          size_t num_seeds,
                          size_t max_depth,
                          bool_t compressed_result,
                          double p,
                          double q,
                          bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                    = NULL;
  cugraph_graph_t* p_graph                               = NULL;
  cugraph_random_walk_result_t* p_result                 = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_seeds, INT32, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  p_source_view = cugraph_type_erased_device_array_view(p_sources);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, p_source_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_node2vec(
    p_handle, p_graph, p_source_view, max_depth, compressed_result, p, q, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "node2vec failed failed.");

  cugraph_type_erased_device_array_view_t* paths;
  cugraph_type_erased_device_array_view_t* path_sizes;
  cugraph_type_erased_device_array_view_t* weights;
  size_t max_path_length;

  max_path_length = cugraph_random_walk_result_get_max_path_length(p_result);
  paths           = cugraph_random_walk_result_get_paths(p_result);
  weights         = cugraph_random_walk_result_get_weights(p_result);

  vertex_t h_paths[max_path_length * num_seeds];
  weight_t h_weights[max_path_length * num_seeds];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_paths, paths, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_weights, weights, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  TEST_ASSERT(test_ret_value,
              cugraph_type_erased_device_array_view_size(paths) ==
                (cugraph_type_erased_device_array_view_size(weights) + num_seeds),
              "paths and weights sizes are not consistent");

  //  We can easily validate that the results of node2vec
  //  are feasible by converting the sparse (h_src,h_dst,h_wgt)
  //  into a dense host matrix and check each path.
  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = 0.0;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_wgt[i];

  if (compressed_result) {
    path_sizes = cugraph_random_walk_result_get_path_sizes(p_result);

    edge_t h_path_sizes[num_seeds];
    edge_t h_path_offsets[num_seeds + 1];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_path_sizes, path_sizes, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    edge_t path_size = 0;
    for (int i = 0; i < num_seeds; ++i)
      path_size += h_path_sizes[i];

    TEST_ASSERT(test_ret_value,
                cugraph_type_erased_device_array_view_size(paths) == path_size,
                "compressed paths size does not match expected size");

    h_path_offsets[0] = 0;
    for (int i = 0; i < num_seeds; ++i)
      h_path_offsets[i + 1] = h_path_offsets[i] + h_path_sizes[i];

    for (int i = 0; (i < num_seeds) && (test_ret_value == 0); ++i) {
      for (int j = h_path_offsets[i]; j < (h_path_offsets[i + 1] - 1); ++j) {
        TEST_ASSERT(test_ret_value,
                    nearlyEqual(h_weights[j - i], M[h_paths[j]][h_paths[j + 1]], EPSILON),
                    "node2vec weights don't match");
      }
    }
  } else {
    for (int i = 0; (i < num_seeds) && (test_ret_value == 0); ++i) {
      for (int j = 0; (j < (max_path_length - 1)) && (test_ret_value == 0); ++j) {
        if (h_paths[i * max_path_length + j + 1] != num_vertices) {
          TEST_ASSERT(
            test_ret_value,
            nearlyEqual(h_weights[i * (max_path_length - 1) + j],
                        M[h_paths[i * max_path_length + j]][h_paths[i * max_path_length + j + 1]],
                        EPSILON),
            "node2vec weights don't match");
        }
      }
    }
  }

  return test_ret_value;
}

int test_node2vec()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[] = {0, 0};
  size_t max_depth = 4;

  return generic_node2vec_test(
    src, dst, wgt, seeds, num_vertices, num_edges, 2, max_depth, FALSE, 0.8, 0.5, FALSE);
}

int test_node2vec_short_dense()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[] = {2, 3};
  size_t max_depth = 4;

  return generic_node2vec_test(
    src, dst, wgt, seeds, num_vertices, num_edges, 2, max_depth, FALSE, 0.8, 0.5, FALSE);
}

int test_node2vec_short_sparse()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[] = {2, 3};
  size_t max_depth = 4;

  // FIXME:  max_depth seems to be off by 1.  It's counting vertices
  //         instead of edges.
  return generic_node2vec_test(
    src, dst, wgt, seeds, num_vertices, num_edges, 2, max_depth, TRUE, 0.8, 0.5, FALSE);
}

int test_node2vec_karate()
{
  size_t num_edges = 156;
  size_t num_vertices = 34;

  vertex_t src[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 2,
                    3, 7, 13, 17, 19, 21, 30, 3, 7, 8, 9, 13, 27, 28, 32, 7, 12,
                    13, 6, 10, 6, 10, 16, 16, 30, 32, 33, 33, 33, 32, 33, 32,
                    33, 32, 33, 33, 32, 33, 32, 33, 25, 27, 29, 32, 33, 25, 27,
                    31, 31, 29, 33, 33, 31, 33, 32, 33, 32, 33, 32, 33, 33, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                    1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6,
                    8, 8, 8, 9, 13, 14, 14, 15, 15, 18, 18, 19, 20, 20, 22, 22,
                    23, 23, 23, 23, 23, 24, 24, 24, 25, 26, 26, 27, 28, 28, 29,
                    29, 30, 30, 31, 31, 32};
  vertex_t dst[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,4,4,5,5,5,6,8,8,8,9,13,14,14,15,15,18,18,19,20,20,22,22,23,23,23,23,23,24,24,24,25,26,26,27,28,28,29,29,30,30,31,31,32,1,2,3,4,5,6,7,8,10,11,12,13,17,19,21,31,2,3,7,13,17,19,21,30,3,7,8,9,13,27,28,32,7,12,13,6,10,6,10,16,16,30,32,33,33,33,32,33,32,33,32,33,33,32,33,32,33,25,27,29,32,33,25,27,31,31,29,33,33,31,33,32,33,32,33,32,33,33};
  weight_t wgt[] = {1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f,1.0f};
  vertex_t seeds[] = {12, 28, 20, 23, 15, 26};
  size_t max_depth = 5;

  return generic_node2vec_test(
    src, dst, wgt, seeds, num_vertices, num_edges, 6, max_depth, TRUE, 0.8, 0.5, FALSE);
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_node2vec);
  result |= RUN_TEST(test_node2vec_short_dense);
  result |= RUN_TEST(test_node2vec_short_sparse);
  result |= RUN_TEST(test_node2vec_karate);
  return result;
}
