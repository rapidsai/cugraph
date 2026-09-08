/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/export.h>
#include <cugraph_c/types.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Determine whether a DLPack managed tensor is accessible from a CUDA device.
 *
 * @param [in]  managed_tensor  Opaque pointer to either a DLPack
 *                              `DLManagedTensor` or `DLManagedTensorVersioned`.
 * @param [in]  versioned       TRUE when @p managed_tensor points to a
 *                              `DLManagedTensorVersioned`; FALSE for a legacy
 *                              `DLManagedTensor`.
 * @param [out] result          TRUE for CUDA, CUDA host, or CUDA managed memory.
 * @param [out] error           Error details on failure.
 * @return CUGRAPH_SUCCESS on success, or an error code on failure.
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_dlpack_is_device_accessible(const void* managed_tensor,
                                                                        bool_t versioned,
                                                                        bool_t* result,
                                                                        cugraph_error_t** error);

/**
 * @brief Determine whether a DLPack managed tensor is accessible from the host.
 *
 * @param [in]  managed_tensor  Opaque pointer to either a DLPack
 *                              `DLManagedTensor` or `DLManagedTensorVersioned`.
 * @param [in]  versioned       TRUE when @p managed_tensor points to a
 *                              `DLManagedTensorVersioned`; FALSE for a legacy
 *                              `DLManagedTensor`.
 * @param [out] result          TRUE for CPU, CUDA host, or CUDA managed memory.
 * @param [out] error           Error details on failure.
 * @return CUGRAPH_SUCCESS on success, or an error code on failure.
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_dlpack_is_host_accessible(const void* managed_tensor,
                                                                      bool_t versioned,
                                                                      bool_t* result,
                                                                      cugraph_error_t** error);

/**
 * @brief Extract array metadata from a DLPack managed tensor.
 *
 * The tensor must describe a one-dimensional, contiguous array. The returned
 * pointer includes the DLPack byte offset.
 *
 * @param [in]  managed_tensor  Opaque pointer to either a DLPack
 *                              `DLManagedTensor` or `DLManagedTensorVersioned`.
 * @param [in]  versioned       TRUE when @p managed_tensor points to a
 *                              `DLManagedTensorVersioned`; FALSE for a legacy
 *                              `DLManagedTensor`.
 * @param [out] data            Pointer to the first array element.
 * @param [out] size            Number of array elements.
 * @param [out] dtype           Corresponding cuGraph data type.
 * @param [out] error           Error details on failure.
 * @return CUGRAPH_SUCCESS on success, or an error code on failure.
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_dlpack_get_array_info(const void* managed_tensor,
                                                                  bool_t versioned,
                                                                  void** data,
                                                                  size_t* size,
                                                                  cugraph_data_type_id_t* dtype,
                                                                  cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
