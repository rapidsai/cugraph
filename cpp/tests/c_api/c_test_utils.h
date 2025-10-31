/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>
#include <cugraph_c/sampling_algorithms.h>

#include <cuda_runtime_api.h>

#include <stdbool.h>
#include <stdio.h>
#include <time.h>

#define TEST_ASSERT(RETURN_VALUE, STATEMENT, MESSAGE)                      \
  {                                                                        \
    if (!(RETURN_VALUE)) {                                                 \
      (RETURN_VALUE) = !(STATEMENT);                                       \
      if ((RETURN_VALUE)) { printf("ASSERTION FAILED: %s\n", (MESSAGE)); } \
    }                                                                      \
  }

#define TEST_ALWAYS_ASSERT(STATEMENT, MESSAGE)                \
  {                                                           \
    int tmp = !(STATEMENT);                                   \
    if (tmp) { printf("ASSERTION FAILED: %s\n", (MESSAGE)); } \
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

int run_sg_test_new(int (*test)(const cugraph_resource_handle_t*),
                    const char* test_name,
                    const cugraph_resource_handle_t* handle);

#define RUN_TEST_NEW(test_name, handle) run_sg_test_new(test_name, #test_name, handle)

int nearlyEqual(float a, float b, float epsilon);
int nearlyEqualDouble(double a, double b, double epsilon);

int create_test_graph(const cugraph_resource_handle_t* p_handle,
                      int32_t* h_src,
                      int32_t* h_dst,
                      float* h_wgt,
                      size_t num_edges,
                      bool_t store_transposed,
                      bool_t renumber,
                      bool_t is_symmetric,
                      cugraph_graph_t** p_graph,
                      cugraph_error_t** ret_error);

int create_test_graph_double(const cugraph_resource_handle_t* p_handle,
                             int32_t* h_src,
                             int32_t* h_dst,
                             double* h_wgt,
                             size_t num_edges,
                             bool_t store_transposed,
                             bool_t renumber,
                             bool_t is_symmetric,
                             cugraph_graph_t** p_graph,
                             cugraph_error_t** ret_error);

int create_sg_test_graph(const cugraph_resource_handle_t* handle,
                         cugraph_data_type_id_t vertex_tid,
                         cugraph_data_type_id_t edge_tid,
                         void* h_src,
                         void* h_dst,
                         cugraph_data_type_id_t weight_tid,
                         void* h_wgt,
                         cugraph_data_type_id_t edge_type_tid,
                         void* h_edge_type,
                         cugraph_data_type_id_t edge_id_tid,
                         void* h_edge_id,
                         cugraph_data_type_id_t edge_time_tid,
                         void* h_edge_start_times,
                         void* h_edge_end_times,
                         size_t num_edges,
                         bool_t store_transposed,
                         bool_t renumber,
                         bool_t is_symmetric,
                         bool_t is_multigraph,
                         cugraph_graph_t** graph,
                         cugraph_error_t** ret_error);

size_t cugraph_size_t_allreduce(const cugraph_resource_handle_t* handle, size_t value);

int validate_sample_result(const cugraph_resource_handle_t* handle,
                           const cugraph_sample_result_t* result,
                           int32_t* h_src,
                           int32_t* h_dst,
                           float* h_wgt,
                           int32_t* h_edge_ids,
                           int32_t* h_edge_types,
                           int32_t* h_edge_start_times,
                           int32_t* h_edge_end_times,
                           size_t num_vertices,
                           size_t num_edge,
                           int32_t* h_start_vertices,
                           size_t num_start_vertices,
                           size_t* h_start_label_offsets,
                           size_t num_start_label_offsets,
                           int32_t* h_fan_out,
                           size_t fan_out_size,
                           cugraph_sampling_options_t* sampling_options,
                           bool validate_edge_times);

#ifdef __cplusplus
}
#endif
