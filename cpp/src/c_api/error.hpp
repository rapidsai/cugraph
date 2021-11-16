/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
