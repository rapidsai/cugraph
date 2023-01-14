/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph_c/resource_handle.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int32_t align_;
} cugraph_rng_state_t;

/**
 * @brief     Create a Random Number Generator State
 *
 * @param [in]  seed        Initial value for seed.  In MG this should be different
 *                          on each GPU
 * @param [out] state       Pointer to the location to store the pointer to the RngState
 * @param [out] error       Pointer to an error object storing details of any error.  Will
 *                          be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_rng_state_create(const cugraph_resource_handle_t* handle,
                                              uint64_t seed,
                                              cugraph_rng_state_t** state,
                                              cugraph_error_t** error);

/**
 * @brief    Destroy a Random Number Generator State
 *
 * @param [in]  p    Pointer to the Random Number Generator State
 */
void cugraph_rng_state_free(cugraph_rng_state_t* p);

#ifdef __cplusplus
}
#endif
