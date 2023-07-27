/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#pragma once

#include "c_test_utils.h"

#include <mpi.h>
#include <stdlib.h>

#define C_MPI_TRY(call)                                                     \
  do {                                                                      \
    int status = call;                                                      \
    if (MPI_SUCCESS != status) {                                            \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      printf("MPI call='%s' at file=%s line=%d failed with %s ",            \
             #call,                                                         \
             __FILE__,                                                      \
             __LINE__,                                                      \
             mpi_error_string);                                             \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

#define C_CUDA_TRY(call)              \
  do {                                \
    cudaError_t const status = call;  \
    if (status != cudaSuccess) {      \
      cudaGetLastError();             \
      printf(                         \
        "CUDA error encountered at: " \
        "call='%s', Reason=%s:%s",    \
        #call,                        \
        cudaGetErrorName(status),     \
        cudaGetErrorString(status));  \
      exit(1);                        \
    }                                 \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

int run_mg_test(int (*test)(const cugraph_resource_handle_t*),
                const char* test_name,
                const cugraph_resource_handle_t* rank);

#define RUN_MG_TEST(test_name, handle) run_mg_test(test_name, #test_name, handle)

void* create_mg_raft_handle(int argc, char** argv);
void free_mg_raft_handle(void* raft_handle);

int create_mg_test_graph(const cugraph_resource_handle_t* p_handle,
                         int32_t* h_src,
                         int32_t* h_dst,
                         float* h_wgt,
                         size_t num_edges,
                         bool_t store_transposed,
                         bool_t is_symmetric,
                         cugraph_graph_t** p_graph,
                         cugraph_error_t** ret_error);

int create_mg_test_graph_double(const cugraph_resource_handle_t* p_handle,
                                int32_t* h_src,
                                int32_t* h_dst,
                                double* h_wgt,
                                size_t num_edges,
                                bool_t store_transposed,
                                bool_t is_symmetric,
                                cugraph_graph_t** p_graph,
                                cugraph_error_t** ret_error);

int create_mg_test_graph_with_edge_ids(const cugraph_resource_handle_t* p_handle,
                                       int32_t* h_src,
                                       int32_t* h_dst,
                                       int32_t* h_idx,
                                       size_t num_edges,
                                       bool_t store_transposed,
                                       bool_t is_symmetric,
                                       cugraph_graph_t** p_graph,
                                       cugraph_error_t** ret_error);

int create_mg_test_graph_with_properties(const cugraph_resource_handle_t* p_handle,
                                         int32_t* h_src,
                                         int32_t* h_dst,
                                         int32_t* h_idx,
                                         int32_t* h_type,
                                         float* h_wgt,
                                         size_t num_edges,
                                         bool_t store_transposed,
                                         bool_t is_symmetric,
                                         cugraph_graph_t** p_graph,
                                         cugraph_error_t** ret_error);

int create_mg_test_graph_new(const cugraph_resource_handle_t* handle,
                             data_type_id_t vertex_tid,
                             data_type_id_t edge_tid,
                             void* h_src,
                             void* h_dst,
                             data_type_id_t weight_tid,
                             void* h_wgt,
                             data_type_id_t edge_type_tid,
                             void* h_edge_type,
                             data_type_id_t edge_id_tid,
                             void* h_edge_id,
                             size_t num_edges,
                             bool_t store_transposed,
                             bool_t renumber,
                             bool_t is_symmetric,
                             bool_t is_multigraph,
                             cugraph_graph_t** graph,
                             cugraph_error_t** ret_error);

size_t cugraph_test_device_gatherv_size(const cugraph_resource_handle_t* handle,
                                        const cugraph_type_erased_device_array_view_t *array);

int cugraph_test_device_gatherv_fill(const cugraph_resource_handle_t* handle,
                                     const cugraph_type_erased_device_array_view_t *array,
                                     void *fill_array);

size_t cugraph_test_scalar_reduce(const cugraph_resource_handle_t* handle, size_t value);

int cugraph_test_host_gatherv_fill(const cugraph_resource_handle_t* handle,
                                   void *input,
                                   size_t input_size,
                                   cugraph_data_type_id_t input_type,
                                   void *output);

#ifdef __cplusplus
}
#endif
