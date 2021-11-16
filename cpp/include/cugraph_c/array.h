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

/**
 * @brief     Create a type erased device array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  dtype       The type of array to create
 * @param [in]  n_elems     The number of elements in the array
 * @param [out] array       Pointer to the location to store the pointer to the device array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_create(
  const cugraph_resource_handle_t* handle,
  data_type_id_t dtype,
  size_t n_elems,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error);

/**
 * @brief    Destroy a type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 */
void cugraph_type_erased_device_array_free(cugraph_type_erased_device_array_t* p);

/**
 * @brief    Get the size of a type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 * @return The number of elements in the array
 */
size_t cugraph_type_erased_device_array_size(const cugraph_type_erased_device_array_t* p);

/**
 * @brief    Get the type of a type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 * @return The type of the elements in the array
 */
data_type_id_t cugraph_type_erased_device_array_type(const cugraph_type_erased_device_array_t* p);

/**
 * @brief    Get the raw pointer of the type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 * @return Pointer (device memory) for the data in the array
 */
const void* cugraph_type_erased_device_array_pointer(const cugraph_type_erased_device_array_t* p);

/**
 * @brief     Create a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  dtype       The type of array to create
 * @param [in]  n_elems     The number of elements in the array
 * @param [out] array       Pointer to the location to store the pointer to the host array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_host_array_create(const cugraph_resource_handle_t* handle,
                                                           data_type_id_t dtype,
                                                           size_t n_elems,
                                                           cugraph_type_erased_host_array_t** array,
                                                           cugraph_error_t** error);

/**
 * @brief    Destroy a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 */
void cugraph_type_erased_host_array_free(cugraph_type_erased_host_array_t* p);

/**
 * @brief    Get the size of a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 * @return The number of elements in the array
 */
size_t cugraph_type_erased_host_array_size(const cugraph_type_erased_host_array_t* p);

/**
 * @brief    Get the type of a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 * @return The type of the elements in the array
 */
data_type_id_t cugraph_type_erased_host_array_type(const cugraph_type_erased_host_array_t* p);

/**
 * @brief    Get the raw pointer of the type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 * @return Pointer (host memory) for the data in the array
 */
void* cugraph_type_erased_host_array_pointer(const cugraph_type_erased_host_array_t* p);

/**
 * @brief    Copy data from host to a type erased device array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to the type erased device array
 * @param [in]  h_src       Pointer to host array to copy into device memory
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_copy_from_host(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_t* dst,
  const byte_t* h_src,
  cugraph_error_t** error);

/**
 * @brief    Copy data from device to a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] h_dst       Pointer to host array
 * @param [in]  src         Pointer to the type erased device array to copy from
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_copy_to_host(
  const cugraph_resource_handle_t* handle,
  byte_t* h_dst,
  const cugraph_type_erased_device_array_t* src,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
