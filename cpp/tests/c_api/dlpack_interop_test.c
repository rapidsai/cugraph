/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h"

#include <cugraph_c/dlpack_interop.h>

#include <dlpack/dlpack.h>
#include <stdint.h>
#include <stdio.h>

static DLManagedTensor make_tensor(void* data,
                                   DLDeviceType device_type,
                                   DLDataType dtype,
                                   int64_t* shape,
                                   int64_t* strides,
                                   uint64_t byte_offset)
{
  DLManagedTensor result       = {0};
  result.dl_tensor.data        = data;
  result.dl_tensor.device      = (DLDevice){device_type, 0};
  result.dl_tensor.ndim        = 1;
  result.dl_tensor.dtype       = dtype;
  result.dl_tensor.shape       = shape;
  result.dl_tensor.strides     = strides;
  result.dl_tensor.byte_offset = byte_offset;
  return result;
}

static int check_array_info(DLDataType dlpack_dtype,
                            cugraph_data_type_id_t expected_dtype,
                            cugraph_error_code_t expected_code)
{
  int32_t values[] = {1, 2, 3};
  int64_t shape    = 2;
  DLManagedTensor tensor =
    make_tensor(values, kDLCUDA, dlpack_dtype, &shape, NULL, sizeof(int32_t));
  void* data  = NULL;
  size_t size = 0;
  cugraph_data_type_id_t dtype;
  cugraph_error_t* error = NULL;
  cugraph_error_code_t code =
    cugraph_dlpack_get_array_info(&tensor, FALSE, &data, &size, &dtype, &error);

  if (code != expected_code) {
    printf("unexpected return code %d (expected %d)\n", code, expected_code);
    cugraph_error_free(error);
    return 1;
  }

  if (expected_code == CUGRAPH_SUCCESS) {
    if (dtype != expected_dtype || size != 2 || data != &values[1]) {
      printf("unexpected array metadata\n");
      cugraph_error_free(error);
      return 1;
    }
  } else if (error == NULL) {
    printf("expected error object on failure\n");
    return 1;
  }

  cugraph_error_free(error);
  return 0;
}

int test_dlpack_array_info()
{
  int test_ret_value = 0;

  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLInt, 8, 1}, INT8, CUGRAPH_SUCCESS) == 0,
              "INT8 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLInt, 16, 1}, INT16, CUGRAPH_SUCCESS) == 0,
              "INT16 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLInt, 32, 1}, INT32, CUGRAPH_SUCCESS) == 0,
              "INT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLInt, 64, 1}, INT64, CUGRAPH_SUCCESS) == 0,
              "INT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLUInt, 8, 1}, UINT8, CUGRAPH_SUCCESS) == 0,
              "UINT8 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLUInt, 16, 1}, UINT16, CUGRAPH_SUCCESS) == 0,
              "UINT16 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLUInt, 32, 1}, UINT32, CUGRAPH_SUCCESS) == 0,
              "UINT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLUInt, 64, 1}, UINT64, CUGRAPH_SUCCESS) == 0,
              "UINT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLFloat, 32, 1}, FLOAT32, CUGRAPH_SUCCESS) == 0,
              "FLOAT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLFloat, 64, 1}, FLOAT64, CUGRAPH_SUCCESS) == 0,
              "FLOAT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLBool, 8, 1}, BOOL, CUGRAPH_SUCCESS) == 0,
              "BOOL conversion failed");
  TEST_ASSERT(
    test_ret_value,
    check_array_info((DLDataType){kDLFloat, 32, 4}, 0, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
    "vectorized dtype should be rejected");
  TEST_ASSERT(
    test_ret_value,
    check_array_info((DLDataType){kDLBfloat, 16, 1}, 0, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
    "bfloat16 should be rejected");
  TEST_ASSERT(
    test_ret_value,
    check_array_info((DLDataType){kDLComplex, 64, 1}, 0, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
    "complex should be rejected");
  TEST_ASSERT(test_ret_value,
              check_array_info((DLDataType){kDLInt, 24, 1}, 0, CUGRAPH_INVALID_INPUT) == 0,
              "unsupported bit width should be rejected");

  return test_ret_value;
}

int test_dlpack_array_info_contiguity()
{
  int test_ret_value = 0;
  int32_t values[]   = {1, 2, 3};
  int64_t stride     = 2;
  void* data         = NULL;
  size_t size        = 0;
  cugraph_data_type_id_t dtype;
  cugraph_error_t* error = NULL;

  int64_t shape = 0;
  DLManagedTensor tensor =
    make_tensor(values, kDLCUDA, (DLDataType){kDLInt, 32, 1}, &shape, &stride, 0);
  cugraph_error_code_t code =
    cugraph_dlpack_get_array_info(&tensor, FALSE, &data, &size, &dtype, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && size == 0,
              "zero-size tensor with explicit stride should be accepted");
  cugraph_error_free(error);

  shape = 1;
  error = NULL;
  code  = cugraph_dlpack_get_array_info(&tensor, FALSE, &data, &size, &dtype, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && size == 1,
              "single-element tensor with explicit stride should be accepted");
  cugraph_error_free(error);

  shape = 2;
  error = NULL;
  code  = cugraph_dlpack_get_array_info(&tensor, FALSE, &data, &size, &dtype, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_INVALID_INPUT && error != NULL,
              "multi-element tensor with non-unit stride should be rejected");
  cugraph_error_free(error);

  return test_ret_value;
}

int test_dlpack_accessibility()
{
  int test_ret_value = 0;
  int32_t value      = 1;
  int64_t shape      = 1;
  DLManagedTensor tensor =
    make_tensor(&value, kDLCPU, (DLDataType){kDLInt, 32, 1}, &shape, NULL, 0);
  bool_t result          = FALSE;
  cugraph_error_t* error = NULL;

  cugraph_error_code_t code = cugraph_dlpack_is_host_accessible(&tensor, FALSE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && result == TRUE,
              "CPU tensor should be host accessible");
  cugraph_error_free(error);

  error = NULL;
  code  = cugraph_dlpack_is_device_accessible(&tensor, FALSE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && result == FALSE,
              "CPU tensor should not be device accessible");
  cugraph_error_free(error);

  tensor.dl_tensor.device.device_type = kDLCUDAHost;
  error                               = NULL;
  code = cugraph_dlpack_is_host_accessible(&tensor, FALSE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && result == TRUE,
              "CUDA host tensor should be host accessible");
  cugraph_error_free(error);

  error = NULL;
  code  = cugraph_dlpack_is_device_accessible(&tensor, FALSE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && result == TRUE,
              "CUDA host tensor should be device accessible");
  cugraph_error_free(error);

  return test_ret_value;
}

int test_versioned_dlpack_tensor()
{
  int test_ret_value                     = 0;
  int32_t value                          = 1;
  int64_t shape                          = 1;
  struct DLManagedTensorVersioned tensor = {0};
  tensor.version = (DLPackVersion){DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
  tensor.dl_tensor =
    make_tensor(&value, kDLCUDA, (DLDataType){kDLInt, 32, 1}, &shape, NULL, 0).dl_tensor;
  bool_t result          = FALSE;
  cugraph_error_t* error = NULL;

  cugraph_error_code_t code = cugraph_dlpack_is_device_accessible(&tensor, TRUE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_SUCCESS && result == TRUE,
              "versioned CUDA tensor should be device accessible");
  cugraph_error_free(error);

  tensor.version.major += 1;
  error = NULL;
  code  = cugraph_dlpack_is_device_accessible(&tensor, TRUE, &result, &error);
  TEST_ASSERT(test_ret_value,
              code == CUGRAPH_INVALID_INPUT && error != NULL,
              "unsupported DLPack major version should be rejected");
  cugraph_error_free(error);

  return test_ret_value;
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_dlpack_array_info);
  result |= RUN_TEST(test_dlpack_array_info_contiguity);
  result |= RUN_TEST(test_dlpack_accessibility);
  result |= RUN_TEST(test_versioned_dlpack_tensor);
  return result;
}
