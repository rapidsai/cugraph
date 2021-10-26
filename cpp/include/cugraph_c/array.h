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

#include <cugraph_c/cugraph_api.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int align_;
} cugraph_type_erased_device_array_t;

typedef struct {
  int align_;
} cugraph_type_erased_host_array_t;

cugraph_type_erased_device_array_t* cugraph_type_erased_device_array_create(
  const cugraph_raft_handle_t* raft_handle, data_type_id_t dtype, size_t n_elems);

void cugraph_type_erased_device_array_free(cugraph_type_erased_device_array_t* p);

size_t cugraph_type_erased_device_array_size(const cugraph_type_erased_device_array_t* p);
data_type_id_t cugraph_type_erased_device_array_type(const cugraph_type_erased_device_array_t* p);
void* cugraph_type_erased_device_array_pointer(const cugraph_type_erased_device_array_t* p);

cugraph_type_erased_host_array_t* cugraph_type_erased_host_array_create(
  const cugraph_raft_handle_t* raft_handle, data_type_id_t dtype, size_t n_elems);

void cugraph_type_erased_host_array_free(cugraph_type_erased_host_array_t* p);

size_t cugraph_type_erased_host_array_size(const cugraph_type_erased_host_array_t* p);
data_type_id_t cugraph_type_erased_host_array_type(const cugraph_type_erased_host_array_t* p);
void* cugraph_type_erased_host_array_pointer(const cugraph_type_erased_host_array_t* p);

cugraph_error_t cugraph_type_erased_device_array_copy_from_host(
  const cugraph_raft_handle_t* raft_handle,
  cugraph_type_erased_device_array_t* dst,
  const byte_t* h_src);

cugraph_error_t cugraph_type_erased_device_array_copy_to_host(
  const cugraph_raft_handle_t* raft_handle,
  byte_t* h_dst,
  const cugraph_type_erased_device_array_t* src);

#ifdef __cplusplus
}
#endif
