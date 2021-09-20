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

#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cugraph_error_ {
  CUGRAPH_SUCCESS,
  CUGRAPH_ERROR_UNKNOWN,
  CUGRAPH_INVALID_HANDLE
} cugraph_error_t;

typedef int bool_t;

/* C stub declarations */

typedef struct c_raft_handle_ {
  int allign_;
} c_raft_handle_t;

typedef struct c_graph_envelope_ {
  int allign_;
} c_graph_envelope_t;

typedef struct c_erased_device_array_ {
  int allign_;
} c_device_array_t;

typedef struct c_erased_unique_ptr_ {
  int allign_;
} c_unique_ptr_t;

typedef struct c_device_buffer_ {
  void* data_;
  size_t size_; /* in bytes */
} c_device_buffer_t;

/* C algorithm specific stubs: should go into separate corresponding headers */

typedef struct c_rw_ret_tuple_ {
  c_device_buffer_t vertex_paths_;
  c_device_buffer_t weight_paths_;
  c_device_buffer_t sizes_;
} c_rw_ret_tuple_t;

typedef struct c_rw_ret_ {
  void* p_erased_ret;
} c_rw_ret_t;

/* TODO:
 * (1.) graph_envelope "cnstr" / "destr":
 *
 *      c_graph_envelope_t* make_graph_envelope(...);
 *      free_graph_envelope(c_graph_envelope_t* ptr);
 *
 * (2.) type reconstruction extractors;
 *      e.g., for `return_t`: different extractors
 *      interpret return in a different (typed) way;
 *
 *      Example:
 *      c_device_buffer_t* extract_rw_ret_vertex_path(c_type vertex_t_id, c_rw_ret_t* rw_result);
 *      c_device_buffer_t* extract_rw_ret_weight_path(c_type vertex_t_id, c_rw_ret_t* rw_result);
 */

/* C algorithm specific wrapper declarations: : should go into separate corresponding headers */

cugraph_error_t c_random_walks(const c_raft_handle_t* ptr_handle,
                               c_graph_envelope_t* ptr_graph_envelope,
                               c_device_array_t* ptr_d_start,
                               size_t num_paths,
                               size_t max_depth,
                               bool_t flag_use_padding,
                               c_unique_ptr_t* ptr_sampling_strategy,
                               c_rw_ret_t* ret);

#ifdef __cplusplus
}
#endif
