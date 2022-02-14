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

int generic_node2vec_test(vertex_t* h_src,
			  vertex_t* h_dst,
			  weight_t* h_wgt,
			  vertex_t* h_seeds,
                          size_t num_edges,
                          size_t num_seeds,
			  size_t max_depth,
			  bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error                    = NULL;

  cugraph_resource_handle_t* p_handle           = NULL;
  cugraph_graph_t* p_graph                      = NULL;

  p_handle = cugraph_create_resource_handle();
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, &p_graph, &ret_error);

  // Populate this test.
  //  We can easily validate that the results of node2vec
  //  are feasible by converting the sparse (h_src,h_dst,h_wgt)
  //  into a dense host matrix and check each path.

  cugraph_sg_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_node2vec()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                  = {0};
  size_t   max_depth                = 4;

  return generic_node2vec_test(src, dst, wgt, seeds, max_depth, num_edges, 1, FALSE);
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_node2vec);
  return result;
}
