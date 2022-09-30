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

#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_type_erased_device_array_t;

typedef struct {
  int32_t align_;
} cugraph_type_erased_device_array_view_t;

typedef struct {
  int32_t align_;
} cugraph_type_erased_host_array_t;

typedef struct {
  int32_t align_;
} cugraph_type_erased_host_array_view_t;

/**
 * @brief     Create a type erased device array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @param [out] array       Pointer to the location to store the pointer to the device array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_create(
  const cugraph_resource_handle_t* handle,
  size_t n_elems,
  data_type_id_t dtype,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error);

/**
 * @brief     Create a type erased device array from a view
 *
 * Copies the data from the view into the new device array
 *
 * @param [in]  handle Handle for accessing resources
 * @param [in]  view   Type erased device array view to copy from
 * @param [out] array  Pointer to the location to store the pointer to the device array
 * @param [out] error  Pointer to an error object storing details of any error.  Will
 *                     be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_create_from_view(
  const cugraph_resource_handle_t* handle,
  const cugraph_type_erased_device_array_view_t* view,
  cugraph_type_erased_device_array_t** array,
  cugraph_error_t** error);

/**
 * @brief    Destroy a type erased device array
 *
 * @param [in]  p    Pointer to the type erased device array
 */
void cugraph_type_erased_device_array_free(cugraph_type_erased_device_array_t* p);

#if 0
// FIXME: Not implemented, need to discuss if this can work.  We will either implement
//        this later or delete it from the interface once we resolve how to handle this
/**
 * @brief    Release the raw pointer of the type erased device array
 *
 * The caller is now responsible for freeing the device pointer
 *
 * @param [in]  p    Pointer to the type erased device array
 * @return Pointer (device memory) for the data in the array
 */
void* cugraph_type_erased_device_array_release(cugraph_type_erased_device_array_t* p);
#endif

/**
 * @brief    Create a type erased device array view from
 *           a type erased device array
 *
 * @param [in]  array       Pointer to the type erased device array
 * @return Pointer to the view of the host array
 */
cugraph_type_erased_device_array_view_t* cugraph_type_erased_device_array_view(
  cugraph_type_erased_device_array_t* array);

/**
 * @brief Create a type erased device array view with a different type
 *
 *    Create a type erased device array view from
 *    a type erased device array treating the underlying
 *    pointer as a different type.
 *
 *    Note: This is only viable when the underlying types are the same size.  That
 *    is, you can switch between INT32 and FLOAT32, or between INT64 and FLOAT64.
 *    But if the types are different sizes this will be an error.
 *
 * @param [in]  array        Pointer to the type erased device array
 * @param [in]  dtype        The type to cast the pointer to
 * @param [out] result_view  Address where to put the allocated device view
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_view_as_type(
  cugraph_type_erased_device_array_t* array,
  data_type_id_t dtype,
  cugraph_type_erased_device_array_view_t** result_view,
  cugraph_error_t** error);

/**
 * @brief    Create a type erased device array view from
 *           a raw device pointer.
 *
 * @param [in]  pointer     Raw device pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return Pointer to the view of the host array
 */
cugraph_type_erased_device_array_view_t* cugraph_type_erased_device_array_view_create(
  void* pointer, size_t n_elems, data_type_id_t dtype);

/**
 * @brief    Destroy a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 */
void cugraph_type_erased_device_array_view_free(cugraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the size of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The number of elements in the array
 */
size_t cugraph_type_erased_device_array_view_size(const cugraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the type of a type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return The type of the elements in the array
 */
data_type_id_t cugraph_type_erased_device_array_view_type(
  const cugraph_type_erased_device_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased device array view
 *
 * @param [in]  p    Pointer to the type erased device array view
 * @return Pointer (device memory) for the data in the array
 */
const void* cugraph_type_erased_device_array_view_pointer(
  const cugraph_type_erased_device_array_view_t* p);

/**
 * @brief     Create a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @param [out] array       Pointer to the location to store the pointer to the host array
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_host_array_create(const cugraph_resource_handle_t* handle,
                                                           size_t n_elems,
                                                           data_type_id_t dtype,
                                                           cugraph_type_erased_host_array_t** array,
                                                           cugraph_error_t** error);

/**
 * @brief    Destroy a type erased host array
 *
 * @param [in]  p    Pointer to the type erased host array
 */
void cugraph_type_erased_host_array_free(cugraph_type_erased_host_array_t* p);

#if 0
// FIXME: Not implemented, need to discuss if this can work.  We will either implement
//        this later or delete it from the interface once we resolve how to handle this
/**
 * @brief    Release the raw pointer of the type erased host array
 *
 * The caller is now responsible for freeing the host pointer
 *
 * @param [in]  p    Pointer to the type erased host array
 * @return Pointer (host memory) for the data in the array
 */
void* cugraph_type_erased_host_array_release(cugraph_type_erased_host_array_t* p);
#endif

/**
 * @brief    Create a type erased host array view from
 *           a type erased host array
 *
 * @param [in]  array       Pointer to the type erased host array
 * @return Pointer to the view of the host array
 */
cugraph_type_erased_host_array_view_t* cugraph_type_erased_host_array_view(
  cugraph_type_erased_host_array_t* array);

/**
 * @brief    Create a type erased host array view from
 *           a raw host pointer.
 *
 * @param [in]  pointer     Raw host pointer
 * @param [in]  n_elems     The number of elements in the array
 * @param [in]  dtype       The type of array to create
 * @return pointer to the view of the host array
 */
cugraph_type_erased_host_array_view_t* cugraph_type_erased_host_array_view_create(
  void* pointer, size_t n_elems, data_type_id_t dtype);

/**
 * @brief    Destroy a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 */
void cugraph_type_erased_host_array_view_free(cugraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the size of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The number of elements in the array
 */
size_t cugraph_type_erased_host_array_size(const cugraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the type of a type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return The type of the elements in the array
 */
data_type_id_t cugraph_type_erased_host_array_type(const cugraph_type_erased_host_array_view_t* p);

/**
 * @brief    Get the raw pointer of the type erased host array view
 *
 * @param [in]  p    Pointer to the type erased host array view
 * @return Pointer (host memory) for the data in the array
 */
void* cugraph_type_erased_host_array_pointer(const cugraph_type_erased_host_array_view_t* p);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased host array view destination
 * @param [in]  src         Pointer to type erased host array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_host_array_view_copy(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_host_array_view_t* dst,
  const cugraph_type_erased_host_array_view_t* src,
  cugraph_error_t** error);

/**
 * @brief    Copy data from host to a type erased device array view
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to the type erased device array view
 * @param [in]  h_src       Pointer to host array to copy into device memory
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_view_copy_from_host(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_view_t* dst,
  const byte_t* h_src,
  cugraph_error_t** error);

/**
 * @brief    Copy data from device to a type erased host array
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] h_dst       Pointer to host array
 * @param [in]  src         Pointer to the type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_view_copy_to_host(
  const cugraph_resource_handle_t* handle,
  byte_t* h_dst,
  const cugraph_type_erased_device_array_view_t* src,
  cugraph_error_t** error);

/**
 * @brief    Copy data between two type erased device array views
 *
 * @param [in]  handle      Handle for accessing resources
 * @param [out] dst         Pointer to type erased device array view destination
 * @param [in]  src         Pointer to type erased device array view source
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_type_erased_device_array_view_copy(
  const cugraph_resource_handle_t* handle,
  cugraph_type_erased_device_array_view_t* dst,
  const cugraph_type_erased_device_array_view_t* src,
  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
