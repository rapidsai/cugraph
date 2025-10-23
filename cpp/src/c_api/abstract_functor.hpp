/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/error.hpp"

#include <cugraph/utilities/graph_traits.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

struct abstract_functor {
  // Move to abstract functor... make operator a void, add cugraph_graph_t * result to functor
  // try that with instantiation questions
  std::unique_ptr<cugraph_error_t> error_ = {std::make_unique<cugraph_error_t>("")};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};

  void unsupported()
  {
    mark_error(CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
               "Type Dispatcher executing unsupported combination of types");
  }

  void mark_error(cugraph_error_code_t error_code, std::string const& error_message)
  {
    error_code_            = error_code;
    error_->error_message_ = error_message;
  }
};

}  // namespace c_api
}  // namespace cugraph
