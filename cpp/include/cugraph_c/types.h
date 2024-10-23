/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bool_ { FALSE = 0, TRUE = 1 } bool_t;

typedef int8_t byte_t;

typedef enum data_type_id_ {
  INT32 = 0,
  INT64,
  FLOAT32,
  FLOAT64,
  SIZE_T,
  BOOL,
  NTYPES
} cugraph_data_type_id_t;

#ifdef __cplusplus
}
#endif
