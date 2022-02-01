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

#include <c_api/error.hpp>
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
