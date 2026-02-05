/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bool_ { FALSE = 0, TRUE = 1 } bool_t;

typedef int8_t byte_t;

typedef enum data_type_id_ {
  INT8 = 0,
  INT16,
  INT32,
  INT64,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  FLOAT32,
  FLOAT64,
  SIZE_T,
  BOOL,
  NTYPES
} cugraph_data_type_id_t;

#ifdef __cplusplus
}
#endif
