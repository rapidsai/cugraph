/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph_c/error.h>

#include <string>

#define CAPI_EXPECTS(STATEMENT, ERROR_CODE, ERROR_MESSAGE, ERROR_OBJECT)                        \
  {                                                                                             \
    if (!(STATEMENT)) {                                                                         \
      (ERROR_OBJECT) =                                                                          \
        reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ERROR_MESSAGE}); \
      return (ERROR_CODE);                                                                      \
    }                                                                                           \
  }

namespace cugraph {
namespace c_api {

struct cugraph_error_t {
  std::string error_message_{};

  cugraph_error_t(const char* what) : error_message_(what) {}
};

}  // namespace c_api
}  // namespace cugraph
