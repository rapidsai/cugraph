/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>

namespace cugraph {

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUGRAPH_EXPECTS and  CUGRAPH_FAIL macros.
 *
 */
struct logic_error : public raft::exception {
  explicit logic_error(char const* const message) : raft::exception(message) {}
  explicit logic_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace cugraph

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optinal format tagas
 * @throw cugraph::logic_error if the condition evaluates to false.
 */
#define CUGRAPH_EXPECTS(cond, fmt, ...)                              \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::string msg{};                                             \
      SET_ERROR_MSG(msg, "cuGraph failure at ", fmt, ##__VA_ARGS__); \
      throw cugraph::logic_error(msg);                               \
    }                                                                \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optinal format tagas
 * @throw always throws cugraph::logic_error
 */
#define CUGRAPH_FAIL(fmt, ...)                                     \
  do {                                                             \
    std::string msg{};                                             \
    SET_ERROR_MSG(msg, "cuGraph failure at ", fmt, ##__VA_ARGS__); \
    throw cugraph::logic_error(msg);                               \
  } while (0)
