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

#include <c_api/error.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

struct abstract_functor {
  // Move to abstract functor... make operator a void, add cugraph_graph_t * result to functor
  // try that with instantiation questions
  std::unique_ptr<cugraph_error_t> error_{std::make_unique<cugraph_error_t>("")};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};

  void unsupported()
  {
    error_code_            = CUGRAPH_UNSUPPORTED_TYPE_COMBINATION;
    error_->error_message_ = "Type Dispatcher executing unsupported combination of types";
  }
};

}  // namespace c_api
}  // namespace cugraph
