/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
