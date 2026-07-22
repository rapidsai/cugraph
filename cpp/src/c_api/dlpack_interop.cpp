/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/dlpack_interop.h>

#include <dlpack/dlpack.h>

#include <cstddef>
#include <cstdint>

namespace {

cugraph_error_code_t get_dlpack_tensor(void const* managed_tensor,
                                       bool_t versioned,
                                       DLTensor const*& result,
                                       cugraph_error_t*& error)
{
  CAPI_EXPECTS(
    managed_tensor != nullptr, CUGRAPH_INVALID_INPUT, "managed_tensor cannot be NULL", error);

  if (versioned == TRUE) {
    auto const* tensor = static_cast<DLManagedTensorVersioned const*>(managed_tensor);
    CAPI_EXPECTS(tensor->version.major == DLPACK_MAJOR_VERSION,
                 CUGRAPH_INVALID_INPUT,
                 "unsupported DLPack major version",
                 error);
    result = &tensor->dl_tensor;
  } else {
    result = &static_cast<DLManagedTensor const*>(managed_tensor)->dl_tensor;
  }
  return CUGRAPH_SUCCESS;
}

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

cugraph_error_code_t data_type_id_from_dlpack(DLDataType const& dlpack_dtype,
                                              cugraph_data_type_id_t& dtype,
                                              cugraph_error_t*& error)
{
  CAPI_EXPECTS(dlpack_dtype.lanes == 1,
               CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
               "vectorized DLPack types (lanes > 1) are not supported",
               error);

  auto const code = static_cast<DLDataTypeCode>(dlpack_dtype.code);
  CAPI_EXPECTS(is_supported_dlpack_type_code(code),
               CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
               "unsupported DLPack type code",
               error);

  switch (code) {
    case kDLInt:
      switch (dlpack_dtype.bits) {
        case 8: dtype = INT8; return CUGRAPH_SUCCESS;
        case 16: dtype = INT16; return CUGRAPH_SUCCESS;
        case 32: dtype = INT32; return CUGRAPH_SUCCESS;
        case 64: dtype = INT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLUInt:
      switch (dlpack_dtype.bits) {
        case 8: dtype = UINT8; return CUGRAPH_SUCCESS;
        case 16: dtype = UINT16; return CUGRAPH_SUCCESS;
        case 32: dtype = UINT32; return CUGRAPH_SUCCESS;
        case 64: dtype = UINT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLFloat:
      switch (dlpack_dtype.bits) {
        case 32: dtype = FLOAT32; return CUGRAPH_SUCCESS;
        case 64: dtype = FLOAT64; return CUGRAPH_SUCCESS;
        default: break;
      }
      break;
    case kDLBool:
      if (dlpack_dtype.bits == 8) {
        dtype = BOOL;
        return CUGRAPH_SUCCESS;
      }
      break;
    default: break;
  }

  CAPI_EXPECTS(false,
               CUGRAPH_INVALID_INPUT,
               "unsupported DLPack dtype bit width for the given type code",
               error);
}

}  // namespace

extern "C" cugraph_error_code_t cugraph_dlpack_is_device_accessible(void const* managed_tensor,
                                                                    bool_t versioned,
                                                                    bool_t* result,
                                                                    cugraph_error_t** error)
{
  *error = nullptr;
  CAPI_EXPECTS(result != nullptr, CUGRAPH_INVALID_INPUT, "result cannot be NULL", *error);

  DLTensor const* tensor{nullptr};
  auto code = get_dlpack_tensor(managed_tensor, versioned, tensor, *error);
  if (code != CUGRAPH_SUCCESS) { return code; }
  auto const device = tensor->device.device_type;
  *result = (device == kDLCUDA || device == kDLCUDAHost || device == kDLCUDAManaged) ? TRUE : FALSE;
  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_dlpack_is_host_accessible(void const* managed_tensor,
                                                                  bool_t versioned,
                                                                  bool_t* result,
                                                                  cugraph_error_t** error)
{
  *error = nullptr;
  CAPI_EXPECTS(result != nullptr, CUGRAPH_INVALID_INPUT, "result cannot be NULL", *error);

  DLTensor const* tensor{nullptr};
  auto code = get_dlpack_tensor(managed_tensor, versioned, tensor, *error);
  if (code != CUGRAPH_SUCCESS) { return code; }
  auto const device = tensor->device.device_type;
  *result = (device == kDLCPU || device == kDLCUDAHost || device == kDLCUDAManaged) ? TRUE : FALSE;
  return CUGRAPH_SUCCESS;
}

extern "C" cugraph_error_code_t cugraph_dlpack_get_array_info(void const* managed_tensor,
                                                              bool_t versioned,
                                                              void** data,
                                                              size_t* size,
                                                              cugraph_data_type_id_t* dtype,
                                                              cugraph_error_t** error)
{
  *error = nullptr;
  CAPI_EXPECTS(data != nullptr, CUGRAPH_INVALID_INPUT, "data cannot be NULL", *error);
  CAPI_EXPECTS(size != nullptr, CUGRAPH_INVALID_INPUT, "size cannot be NULL", *error);
  CAPI_EXPECTS(dtype != nullptr, CUGRAPH_INVALID_INPUT, "dtype cannot be NULL", *error);

  DLTensor const* tensor{nullptr};
  auto code = get_dlpack_tensor(managed_tensor, versioned, tensor, *error);
  if (code != CUGRAPH_SUCCESS) { return code; }
  CAPI_EXPECTS(tensor->ndim == 1,
               CUGRAPH_INVALID_INPUT,
               "pylibcugraph array inputs must be one-dimensional",
               *error);
  CAPI_EXPECTS(
    tensor->shape != nullptr, CUGRAPH_INVALID_INPUT, "DLPack tensor shape cannot be NULL", *error);
  CAPI_EXPECTS(tensor->shape[0] >= 0,
               CUGRAPH_INVALID_INPUT,
               "DLPack tensor dimensions cannot be negative",
               *error);
  CAPI_EXPECTS(tensor->strides == nullptr || tensor->strides[0] == 1,
               CUGRAPH_INVALID_INPUT,
               "pylibcugraph array inputs must be contiguous",
               *error);

  code = data_type_id_from_dlpack(tensor->dtype, *dtype, *error);
  if (code != CUGRAPH_SUCCESS) { return code; }

  CAPI_EXPECTS(tensor->data != nullptr || tensor->byte_offset == 0,
               CUGRAPH_INVALID_INPUT,
               "DLPack tensor with NULL data cannot have a byte offset",
               *error);
  *data = tensor->data == nullptr
            ? nullptr
            : static_cast<void*>(static_cast<std::uint8_t*>(tensor->data) + tensor->byte_offset);
  *size = static_cast<size_t>(tensor->shape[0]);
  return CUGRAPH_SUCCESS;
}
