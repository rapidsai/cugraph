/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/error.h>

extern "C" const char* cugraph_error_message(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->error_message_.c_str();
  } else {
    return nullptr;
  }
}

extern "C" void cugraph_error_free(cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    delete internal_pointer;
  }
}
