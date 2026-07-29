/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/export.h>
#include <cugraph_c/types.h>

#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert a DLPack DLDataType to a cugraph_data_type_id_t.
 *
 * Converts the dtype field from a DLPack tensor (e.g. DLTensor::dtype or
 * __dlpack__ metadata) into the cugraph_data_type_id_t required by
 * cugraph_type_erased_device_array_view_create() and related array APIs.
 *
 * Only scalar types (lanes == 1) are supported. Vectorized DLPack types,
 * bfloat16, complex, and opaque handle types are rejected.
 *
 * @param [in]  dlpack_dtype  Pointer to the DLPack data type to convert.
 *                             Must not be NULL.
 * @param [out] dtype         Pointer to store the resulting cuGraph data type.
 *                             Must not be NULL.
 * @param [out] error         Pointer to an error object storing details of any
 *                             error. Will be populated if the return code is
 *                             not CUGRAPH_SUCCESS. Must not be NULL.
 * @return CUGRAPH_SUCCESS on success, or an error code on failure.
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_data_type_id_from_dlpack(const DLDataType* dlpack_dtype,
                                                                     cugraph_data_type_id_t* dtype,
                                                                     cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
