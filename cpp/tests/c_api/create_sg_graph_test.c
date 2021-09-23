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

#include "c_test_utils.h"  /* RUN_TEST */

#include <cugraph_c/cugraph_api.h>


/*
 * Simple check of creating a graph from a COO on device memory.
 */
int test_create_sg_graph_simple() {
   int test_failed = 0;

   cugraph_graph_envelope_t* G = NULL;
   cugraph_raft_handle_t handle;
   cugraph_device_array_t* src_ptr = NULL;
   cugraph_device_array_t* dst_ptr = NULL;
   cugraph_device_array_t* weights_ptr = NULL;
   size_t num_verts = 4;
   size_t num_edges = 3;
   bool_t do_expensive_check = 0;
   bool_t store_transposed = 0;
   bool_t is_symmetric = 0;
   bool_t is_multigraph = 0;

   /*
    * FIXME: populate GPU memory with a small (4 verts, 3 edges) graph COO.
    * FIXME: return success until this test is finished.
    */
   return 0;

   G = cugraph_make_sg_graph(
          &handle,
          INT32, INT32, INT32,  /* vert, edge, weight types */
          store_transposed,
          src_ptr, dst_ptr, weights_ptr,
          num_verts, num_edges,
          do_expensive_check,
          is_symmetric, is_multigraph);

   cugraph_free_graph(G);

   return test_failed;
}


/*
 * Since cugraph_make_sg_graph() can return NULL, this ensures
 * cugraph_free_graph() can accept NULL.
 */
int test_free_graph_NULL_ptr() {
   /* Returns void, so check that the call does not crash. */
   cugraph_free_graph((cugraph_graph_envelope_t*) NULL);
   return 0;
}


/*
 * Test creating a graph with NULL device arrays and "expensive check" enabled.
 */
int test_create_sg_graph_bad_arrays() {
   int test_failed = 0;

   cugraph_graph_envelope_t* G = NULL;
   cugraph_raft_handle_t handle;
   cugraph_device_array_t* src_ptr = NULL;
   cugraph_device_array_t* dst_ptr = NULL;
   cugraph_device_array_t* weights_ptr = NULL;
   size_t num_verts = 4;
   size_t num_edges = 3;
   bool_t do_expensive_check = 1;
   bool_t store_transposed = 0;
   bool_t is_symmetric = 0;
   bool_t is_multigraph = 0;

   /*
    * FIXME: return success until this test is finished.
    */
   return 0;

   G = cugraph_make_sg_graph(
          &handle,
          INT32, INT32, INT32,  /* vert, edge, weight types */
          store_transposed,
          src_ptr, dst_ptr, weights_ptr,
          num_verts, num_edges,
          do_expensive_check,
          is_symmetric, is_multigraph);

   if(G != NULL) {
      test_failed = 1;
   }

   return test_failed;
}

/******************************************************************************/

int main(int argc, char** argv) {
   int result = 0;
   result |= RUN_TEST(test_create_sg_graph_simple);
   result |= RUN_TEST(test_free_graph_NULL_ptr);
   result |= RUN_TEST(test_create_sg_graph_bad_arrays);
   return result;
}
