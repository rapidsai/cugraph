/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

typedef enum cugraph_error_code_ {
  CUGRAPH_SUCCESS = 0,
  CUGRAPH_UNKNOWN_ERROR,
  CUGRAPH_INVALID_HANDLE,
  CUGRAPH_ALLOC_ERROR,
  CUGRAPH_INVALID_INPUT,
  CUGRAPH_NOT_IMPLEMENTED,
  CUGRAPH_UNSUPPORTED_TYPE_COMBINATION
} cugraph_error_code_t;

typedef struct cugraph_error_ {
  int32_t align_;
} cugraph_error_t;

/**
 * @brief     Return an error message
 *
 * @param [in]  error       The error object from some cugraph function call
 * @return a C-style string that provides detail for the error
 */
const char* cugraph_error_message(const cugraph_error_t* error);

/**
 * @brief    Destroy an error message
 *
 * @param [in]  error       The error object from some cugraph function call
 */
void cugraph_error_free(cugraph_error_t* error);

#ifdef __cplusplus
}
#endif
