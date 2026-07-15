/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/export.h>
#include <cugraph_c/types.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DLPack-compatible device types used by the cuGraph C API.
 *
 * The numeric values are part of the DLPack ABI. These cuGraph-prefixed
 * declarations keep this public header self-contained and avoid requiring C
 * API consumers to install or include DLPack headers.
 */
typedef enum {
  CUGRAPH_DL_DEVICE_TYPE_CPU          = 1,
  CUGRAPH_DL_DEVICE_TYPE_CUDA         = 2,
  CUGRAPH_DL_DEVICE_TYPE_CUDA_HOST    = 3,
  CUGRAPH_DL_DEVICE_TYPE_CUDA_MANAGED = 13,
} cugraph_dlpack_device_type_t;

/** @brief DLPack-compatible device descriptor. */
typedef struct {
  cugraph_dlpack_device_type_t device_type;
  int32_t device_id;
} cugraph_dlpack_device_t;

/** @brief DLPack-compatible data type codes used by the cuGraph C API. */
typedef enum {
  CUGRAPH_DL_DATA_TYPE_CODE_INT     = 0,
  CUGRAPH_DL_DATA_TYPE_CODE_UINT    = 1,
  CUGRAPH_DL_DATA_TYPE_CODE_FLOAT   = 2,
  CUGRAPH_DL_DATA_TYPE_CODE_BFLOAT  = 4,
  CUGRAPH_DL_DATA_TYPE_CODE_COMPLEX = 5,
  CUGRAPH_DL_DATA_TYPE_CODE_BOOL    = 6,
} cugraph_dlpack_data_type_code_t;

/** @brief DLPack-compatible scalar data type descriptor. */
typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} cugraph_dlpack_data_type_t;

/** @brief DLPack-compatible tensor descriptor. */
typedef struct {
  void* data;
  cugraph_dlpack_device_t device;
  int32_t ndim;
  cugraph_dlpack_data_type_t dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} cugraph_dlpack_tensor_t;

/** @brief DLPack-compatible legacy managed tensor. */
typedef struct cugraph_dlpack_managed_tensor_t {
  cugraph_dlpack_tensor_t dl_tensor;
  void* manager_ctx;
  void (*deleter)(struct cugraph_dlpack_managed_tensor_t* self);
} cugraph_dlpack_managed_tensor_t;

/** @brief DLPack protocol version. */
typedef struct {
  uint32_t major;
  uint32_t minor;
} cugraph_dlpack_version_t;

/** @brief DLPack-compatible versioned managed tensor. */
typedef struct cugraph_dlpack_managed_tensor_versioned_t {
  cugraph_dlpack_version_t version;
  void* manager_ctx;
  void (*deleter)(struct cugraph_dlpack_managed_tensor_versioned_t* self);
  uint64_t flags;
  cugraph_dlpack_tensor_t dl_tensor;
} cugraph_dlpack_managed_tensor_versioned_t;

/**
 * @brief Convert a DLPack-compatible data type to a cugraph_data_type_id_t.
 *
 * Converts the dtype field from a DLPack tensor or __dlpack__ metadata into
 * the cugraph_data_type_id_t required by
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
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_data_type_id_from_dlpack(const cugraph_dlpack_data_type_t* dlpack_dtype,
                                 cugraph_data_type_id_t* dtype,
                                 cugraph_error_t** error);

#ifdef __cplusplus
}
#endif
