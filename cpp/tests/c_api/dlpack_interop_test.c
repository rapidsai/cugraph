/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h"

#include <cugraph_c/dlpack_interop.h>

#include <stdio.h>

static int check_conversion(cugraph_dlpack_data_type_t dlpack_dtype,
                            cugraph_data_type_id_t expected_dtype,
                            cugraph_error_code_t expected_code)
{
  cugraph_data_type_id_t dtype;
  cugraph_error_t* error    = NULL;
  cugraph_error_code_t code = cugraph_data_type_id_from_dlpack(&dlpack_dtype, &dtype, &error);

  if (code != expected_code) {
    printf("unexpected return code %d (expected %d)\n", code, expected_code);
    cugraph_error_free(error);
    return 1;
  }

  if (expected_code == CUGRAPH_SUCCESS) {
    if (dtype != expected_dtype) {
      printf("unexpected dtype %d (expected %d)\n", dtype, expected_dtype);
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

int test_dlpack_interop_conversions()
{
  int test_ret_value = 0;

  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_INT, 8, 1},
                               INT8,
                               CUGRAPH_SUCCESS) == 0,
              "INT8 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_INT, 16, 1},
                               INT16,
                               CUGRAPH_SUCCESS) == 0,
              "INT16 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_INT, 32, 1},
                               INT32,
                               CUGRAPH_SUCCESS) == 0,
              "INT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_INT, 64, 1},
                               INT64,
                               CUGRAPH_SUCCESS) == 0,
              "INT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_UINT, 8, 1},
                               UINT8,
                               CUGRAPH_SUCCESS) == 0,
              "UINT8 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_UINT, 16, 1},
                               UINT16,
                               CUGRAPH_SUCCESS) == 0,
              "UINT16 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_UINT, 32, 1},
                               UINT32,
                               CUGRAPH_SUCCESS) == 0,
              "UINT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_UINT, 64, 1},
                               UINT64,
                               CUGRAPH_SUCCESS) == 0,
              "UINT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_FLOAT, 32, 1},
                               FLOAT32,
                               CUGRAPH_SUCCESS) == 0,
              "FLOAT32 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_FLOAT, 64, 1},
                               FLOAT64,
                               CUGRAPH_SUCCESS) == 0,
              "FLOAT64 conversion failed");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_BOOL, 8, 1},
                               BOOL,
                               CUGRAPH_SUCCESS) == 0,
              "BOOL conversion failed");

  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_FLOAT, 32, 4},
                               0,
                               CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
              "vectorized dtype should be rejected");
  TEST_ASSERT(
    test_ret_value,
    check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_BFLOAT, 16, 1},
                     0,
                     CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
    "bfloat16 should be rejected");
  TEST_ASSERT(
    test_ret_value,
    check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_COMPLEX, 64, 1},
                     0,
                     CUGRAPH_UNSUPPORTED_TYPE_COMBINATION) == 0,
    "complex should be rejected");
  TEST_ASSERT(test_ret_value,
              check_conversion((cugraph_dlpack_data_type_t){CUGRAPH_DL_DATA_TYPE_CODE_INT, 24, 1},
                               0,
                               CUGRAPH_INVALID_INPUT) == 0,
              "unsupported bit width should be rejected");

  return test_ret_value;
}

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_dlpack_interop_conversions);
  return result;
}
