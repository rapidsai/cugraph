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

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/resource_handle.h>

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C stub declarations */

typedef struct cugraph_graph_envelope_ {
  int align_;
} cugraph_graph_envelope_t;

typedef struct cugraph_erased_unique_ptr_ {
  int align_;
} cugraph_unique_ptr_t;

typedef struct cugraph_device_buffer_ {
  void* data_;  /* (rmm::device_buffer*) */
  size_t size_; /* rmm::device_buffer::size()  */
} cugraph_device_buffer_t;

/* C algorithm specific stubs: should go into separate corresponding headers */

typedef struct cugraph_rw_ret_tuple_ {
  cugraph_device_buffer_t vertex_paths_;
  cugraph_device_buffer_t weight_paths_;
  cugraph_device_buffer_t sizes_;
} cugraph_rw_ret_tuple_t;

typedef struct cugraph_rw_ret_ {
  void* p_erased_ret;
} cugraph_rw_ret_t;

/* Runtime production (not just DEBUG) assert */
bool_t runtime_assert(bool_t statement_truth_value, const char* error_msg);

/* Functionality:
 * (1.) graph_envelope "cnstr" / "destr":
 *
 *      cugraph_graph_envelope_t* make_graph_envelope(...);
 *      free_graph_envelope(cugraph_graph_envelope_t* ptr);
 *
 * (2.) type reconstruction extractors;
 *      e.g., for `return_t`: different extractors
 *      interpret return in a different (typed) way;
 *
 *      Example:
 *      cugraph_device_buffer_t* extract_rw_ret_vertex_path(cugraph_type vertex_t_id,
 * cugraph_rw_ret_t* rw_result); cugraph_device_buffer_t* extract_rw_ret_weight_path(cugraph_type
 * vertex_t_id, cugraph_rw_ret_t* rw_result);
 */

/* C algorithm specific wrapper declarations; FIXME: should go into separate corresponding headers
 */

/* Random Walks functionality*/

/* Sampling data allocator*/
cugraph_unique_ptr_t* cugraph_create_sampling_strategy(int sampling_type_id, double p, double q);

/* Sampling data deallocator*/
void cugraph_free_sampling_strategy(cugraph_unique_ptr_t* p_sampling);

/* deallocate result returned by RW wrapper*/
void cugraph_free_rw_result(cugraph_rw_ret_t* p_rw_ret);

/* RW result vertex extractor*/
cugraph_error_code_t extract_vertex_rw_result(cugraph_rw_ret_t* p_rw_ret,
                                              cugraph_device_buffer_t* p_d_buf_v);

/* RW result weights extractor*/
cugraph_error_code_t extract_weight_rw_result(cugraph_rw_ret_t* p_rw_ret,
                                              cugraph_device_buffer_t* p_d_buf_w);

/* RW result size extractor*/
cugraph_error_code_t extract_size_rw_result(cugraph_rw_ret_t* p_rw_ret,
                                            cugraph_device_buffer_t* p_d_buf_sz);

/* algorithm wrapper*/
cugraph_error_code_t cugraph_random_walks(const cugraph_resource_handle_t* ptr_handle,
                                          cugraph_graph_envelope_t* ptr_graph_envelope,
                                          cugraph_device_buffer_t* ptr_d_start,
                                          size_t num_paths,
                                          size_t max_depth,
                                          bool_t flag_use_padding,
                                          cugraph_unique_ptr_t* ptr_sampling_strategy,
                                          cugraph_rw_ret_t* ret);

/* SG graph allocator*/
cugraph_graph_envelope_t* cugraph_make_sg_graph(const cugraph_resource_handle_t* p_handle,
                                                data_type_id_t vertex_tid,
                                                data_type_id_t edge_tid,
                                                data_type_id_t weight_tid,
                                                bool_t st,
                                                cugraph_device_buffer_t* p_src,
                                                cugraph_device_buffer_t* p_dst,
                                                cugraph_device_buffer_t* p_weights,
                                                size_t num_vertices,
                                                size_t num_edges,
                                                bool_t check,
                                                bool_t is_symmetric,
                                                bool_t is_multigraph);

/* graph deallocator*/
void cugraph_free_graph(cugraph_graph_envelope_t* graph);

/* rmm::device buffer allocator: fill pointer semantics*/
cugraph_error_code_t cugraph_make_device_buffer(const cugraph_resource_handle_t* handle,
                                                data_type_id_t dtype,
                                                size_t n_elems,
                                                cugraph_device_buffer_t* ptr_buffer);

/* rmm::device buffer de-allocator*/
void cugraph_free_device_buffer(cugraph_device_buffer_t* ptr_buffer);

/* update dst device buffer from host src*/
cugraph_error_code_t cugraph_update_device_buffer(const cugraph_resource_handle_t* handle,
                                                  data_type_id_t dtype,
                                                  cugraph_device_buffer_t* ptr_dst,
                                                  const byte_t* ptr_h_src);

/* update src host buffer device src*/
cugraph_error_code_t cugraph_update_host_buffer(const cugraph_resource_handle_t* handle,
                                                data_type_id_t dtype,
                                                byte_t* ptr_h_dst,
                                                const cugraph_device_buffer_t* ptr_src);

#ifdef __cplusplus
}
#endif
