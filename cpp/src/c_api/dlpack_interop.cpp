/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/dlpack_interop.h>

namespace {

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

extern "C" cugraph_error_code_t cugraph_data_type_id_from_dlpack(const DLDataType* dlpack_dtype,
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
