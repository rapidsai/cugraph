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

#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#include <stdio.h>
#include <time.h>

#define TEST_ASSERT(RETURN_VALUE, STATEMENT, MESSAGE)                      \
  {                                                                        \
    if (!(RETURN_VALUE)) {                                                 \
      (RETURN_VALUE) = !(STATEMENT);                                       \
      if ((RETURN_VALUE)) { printf("ASSERTION FAILED: %s\n", (MESSAGE)); } \
    }                                                                      \
  }

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Runs the function pointed to by "test" and returns the return code.  Also
 * prints reporting info (using "test_name"): pass/fail and run time, to stdout.
 *
 * Intended to be used by the RUN_TEST macro.
 */
int run_sg_test(int (*test)(), const char* test_name);

#define RUN_TEST(test_name) run_sg_test(test_name, #test_name)

int nearlyEqual(float a, float b, float epsilon);
int create_test_graph(const cugraph_resource_handle_t* p_handle,
                      int32_t* h_src,
                      int32_t* h_dst,
                      float* h_wgt,
                      size_t num_edges,
                      bool_t store_transposed,
                      bool_t renumber,
                      cugraph_graph_t** p_graph,
                      cugraph_error_t** ret_error);

void* create_raft_handle(int prows);
void free_raft_handle(void* raft_handle);

#ifdef __cplusplus
}
#endif
