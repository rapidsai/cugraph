/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#ifndef ERRORUTILS_HPP
#define ERRORUTILS_HPP

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>

#include <rmm/rmm.h>

#include "nvgraph_error_utils.h"

#include <cudf/types.h>

namespace cugraph {
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUGRAPH_EXPECTS macro.
 *
 *---------------------------------------------------------------------------**/
struct logic_error : public std::logic_error {
  logic_error(char const* const message) : std::logic_error(message) {}

  logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API
};
/**---------------------------------------------------------------------------*
 * @brief Exception thrown when a CUDA error is encountered.
 *
 *---------------------------------------------------------------------------**/
struct cuda_error : public std::runtime_error {
  cuda_error(std::string const& message) : std::runtime_error(message) {}
};
}  // namespace cugraph

#define STRINGIFY_DETAIL(x) #x
#define CUGRAPH_STRINGIFY(x) STRINGIFY_DETAIL(x)

/**---------------------------------------------------------------------------*
 * @brief Macro for checking (pre-)conditions that throws an exception when  
 * a condition is violated.
 * 
 * Example usage:
 * 
 * @code
 * CUGRAPH_EXPECTS(lhs->dtype == rhs->dtype, "Column type mismatch");
 * @endcode
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] reason String literal description of the reason that cond is
 * expected to be true
 * @throw cugraph::logic_error if the condition evaluates to false.
 *---------------------------------------------------------------------------**/
#define CUGRAPH_EXPECTS(cond, reason)                           \
  (!!(cond))                                                 \
      ? static_cast<void>(0)                                 \
      : throw cugraph::logic_error("CUGRAPH failure at: " __FILE__ \
                                ":" CUGRAPH_STRINGIFY(__LINE__) ": " reason)

/**---------------------------------------------------------------------------*
 * @brief Try evaluation an expression with a gdf_error type,
 * and throw an appropriate exception if it fails.
 *---------------------------------------------------------------------------**/
#define CUGRAPH_TRY(_gdf_error_expression) do { \
    auto _evaluated = _gdf_error_expression; \
    if (_evaluated == GDF_SUCCESS) { break; } \
    throw cugraph::logic_error( \
        ("CUGRAPH error " + std::string(gdf_error_get_name(_evaluated)) + " at " \
       __FILE__ ":"  \
        CUGRAPH_STRINGIFY(__LINE__) " evaluating " CUGRAPH_STRINGIFY(#_gdf_error_expression)).c_str() ); \
} while(0)

/**---------------------------------------------------------------------------*
 * @brief Indicates that an erroneous code path has been taken.
 *
 * In host code, throws a `cugraph::logic_error`.
 *
 *
 * Example usage:
 * ```
 * CUGRAPH_FAIL("Non-arithmetic operation is not supported");
 * ```
 * 
 * @param[in] reason String literal description of the reason
 *---------------------------------------------------------------------------**/
#define CUGRAPH_FAIL(reason)                              \
  throw cugraph::logic_error("cuGraph failure at: " __FILE__ \
                          ":" CUGRAPH_STRINGIFY(__LINE__) ": " reason)

namespace cugraph {
namespace detail {

inline void throw_rmm_error(rmmError_t error, const char* file,
                             unsigned int line) {
  // todo: throw cuda_error if the error is from cuda
  throw cugraph::logic_error(
      std::string{"RMM error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  rmmGetErrorString(error)});
}

inline void throw_cuda_error(cudaError_t error, const char* file,
                             unsigned int line) {
  throw cugraph::cuda_error(
      std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                  std::to_string(line) + ": " + std::to_string(error) + " " +
                  cudaGetErrorName(error) + " " + cudaGetErrorString(error)});
}

inline void check_stream(cudaStream_t stream, const char* file,
                         unsigned int line) {
  cudaError_t error{cudaSuccess};
  error = cudaStreamSynchronize(stream);
  if (cudaSuccess != error) {
    throw_cuda_error(error, file, line);
  }

  error = cudaGetLastError();
  if (cudaSuccess != error) {
    throw_cuda_error(error, file, line);
  }
}
}  // namespace detail
}  // namespace cugraph

/**---------------------------------------------------------------------------*
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, throws an exception detailing the CUDA error that occurred.
 *
 * This macro supersedes GDF_REQUIRE and should be preferred in all instances.
 * GDF_REQUIRE should be considered deprecated.
 *
 *---------------------------------------------------------------------------**/
#define CUDA_TRY(call)                                            \
  do {                                                            \
    cudaError_t const status = (call);                            \
    if (cudaSuccess != status) {                                  \
      cugraph::detail::throw_cuda_error(status, __FILE__, __LINE__); \
    }                                                             \
  } while (0);
#endif

#define CUDA_CHECK_LAST() {                                       \
  cudaError_t const status = cudaGetLastError();                  \
  if(status != cudaSuccess) {                                     \
   cugraph::detail::throw_cuda_error(status, __FILE__, __LINE__); \
  }                                                               \
}

/**---------------------------------------------------------------------------*
 * @brief Debug macro to synchronize a stream and check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream, and
 * check for any CUDA errors returned from cudaGetLastError. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 *
 * Similar to assert(), it is only present in non-Release builds.
 *
 *---------------------------------------------------------------------------**/
#ifndef NDEBUG
#define CHECK_STREAM(stream) \
  cugraph::detail::check_stream((stream), __FILE__, __LINE__)
#else
#define CHECK_STREAM(stream) static_cast<void>(0)
#endif