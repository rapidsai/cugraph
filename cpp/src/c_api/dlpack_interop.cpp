/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/dlpack_interop.h>

#include <dlpack/dlpack.h>

#include <cstddef>

namespace {

#define CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(cugraph_value, dlpack_value) \
  static_assert(static_cast<int>(cugraph_value) == static_cast<int>(dlpack_value))

CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DEVICE_TYPE_CPU, kDLCPU);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DEVICE_TYPE_CUDA, kDLCUDA);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DEVICE_TYPE_CUDA_HOST, kDLCUDAHost);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DEVICE_TYPE_CUDA_MANAGED, kDLCUDAManaged);

CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_INT, kDLInt);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_UINT, kDLUInt);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_FLOAT, kDLFloat);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_BFLOAT, kDLBfloat);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_COMPLEX, kDLComplex);
CUGRAPH_DLPACK_ASSERT_ENUM_VALUE(CUGRAPH_DL_DATA_TYPE_CODE_BOOL, kDLBool);

#define CUGRAPH_DLPACK_ASSERT_ABI(cugraph_type, dlpack_type)  \
  static_assert(sizeof(cugraph_type) == sizeof(dlpack_type)); \
  static_assert(alignof(cugraph_type) == alignof(dlpack_type))

#define CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_type, dlpack_type, member) \
  static_assert(offsetof(cugraph_type, member) == offsetof(dlpack_type, member))

CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_device_t, DLDevice);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_device_t, DLDevice, device_type);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_device_t, DLDevice, device_id);

CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_data_type_t, DLDataType);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_data_type_t, DLDataType, code);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_data_type_t, DLDataType, bits);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_data_type_t, DLDataType, lanes);

CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_tensor_t, DLTensor);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, data);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, device);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, ndim);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, dtype);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, shape);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, strides);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_tensor_t, DLTensor, byte_offset);

CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_managed_tensor_t, DLManagedTensor);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_t, DLManagedTensor, dl_tensor);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_t, DLManagedTensor, manager_ctx);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_t, DLManagedTensor, deleter);

#if DLPACK_VERSION >= 100
CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_version_t, DLPackVersion);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_version_t, DLPackVersion, major);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_version_t, DLPackVersion, minor);

CUGRAPH_DLPACK_ASSERT_ABI(cugraph_dlpack_managed_tensor_versioned_t, DLManagedTensorVersioned);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_versioned_t,
                             DLManagedTensorVersioned,
                             version);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_versioned_t,
                             DLManagedTensorVersioned,
                             manager_ctx);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_versioned_t,
                             DLManagedTensorVersioned,
                             deleter);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_versioned_t,
                             DLManagedTensorVersioned,
                             flags);
CUGRAPH_DLPACK_ASSERT_OFFSET(cugraph_dlpack_managed_tensor_versioned_t,
                             DLManagedTensorVersioned,
                             dl_tensor);
#endif

#undef CUGRAPH_DLPACK_ASSERT_OFFSET
#undef CUGRAPH_DLPACK_ASSERT_ABI
#undef CUGRAPH_DLPACK_ASSERT_ENUM_VALUE

bool is_supported_dlpack_type_code(DLDataTypeCode code)
{
  switch (code) {
    case kDLInt:
    case kDLUInt:
    case kDLFloat:
    case kDLBool: return true;
    default: return false;
  }
}

}  // namespace

extern "C" cugraph_error_code_t cugraph_data_type_id_from_dlpack(
  const cugraph_dlpack_data_type_t* dlpack_dtype,
  cugraph_data_type_id_t* dtype,
  cugraph_error_t** error)
{
  *error = nullptr;

  CAPI_EXPECTS(
    dlpack_dtype != nullptr, CUGRAPH_INVALID_INPUT, "dlpack_dtype cannot be NULL", *error);
  CAPI_EXPECTS(dtype != nullptr, CUGRAPH_INVALID_INPUT, "dtype cannot be NULL", *error);

  if (dlpack_dtype->lanes != 1) {
    CAPI_EXPECTS(false,
                 CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                 "vectorized DLPack types (lanes > 1) are not supported",
                 *error);
  }

  auto const code = static_cast<DLDataTypeCode>(dlpack_dtype->code);
  if (!is_supported_dlpack_type_code(code)) {
    CAPI_EXPECTS(
      false, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION, "unsupported DLPack type code", *error);
  }

  switch (code) {
    case kDLInt:
      switch (dlpack_dtype->bits) {
        case 8: *dtype = INT8; return CUGRAPH_SUCCESS;
        case 16: *dtype = INT16; return CUGRAPH_SUCCESS;
        case 32: *dtype = INT32; return CUGRAPH_SUCCESS;
        case 64: *dtype = INT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLUInt:
      switch (dlpack_dtype->bits) {
        case 8: *dtype = UINT8; return CUGRAPH_SUCCESS;
        case 16: *dtype = UINT16; return CUGRAPH_SUCCESS;
        case 32: *dtype = UINT32; return CUGRAPH_SUCCESS;
        case 64: *dtype = UINT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLFloat:
      switch (dlpack_dtype->bits) {
        case 32: *dtype = FLOAT32; return CUGRAPH_SUCCESS;
        case 64: *dtype = FLOAT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLBool:
      if (dlpack_dtype->bits == 8) {
        *dtype = BOOL;
        return CUGRAPH_SUCCESS;
      }
      break;
    default: break;
  }

  CAPI_EXPECTS(false,
               CUGRAPH_INVALID_INPUT,
               "unsupported DLPack dtype bit width for the given type code",
               *error);
}
