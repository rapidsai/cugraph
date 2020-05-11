/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <stdio.h>
#include <string>
#include <sstream>
#include <time.h>
 
#include "stacktrace.h"

namespace nvgraph {

#if defined(DEBUG) || defined(VERBOSE_DIAG)
#define STACKTRACE "\nStack trace:\n" + std::string(e.trace())
#define WHERE " at: " << __FILE__ << ':' << __LINE__
#else
#define STACKTRACE ""
#define WHERE ""
#endif 


enum NVGRAPH_ERROR { 
/*********************************************************
 * Flags for status reporting
 *********************************************************/
    NVGRAPH_OK=0, 
    NVGRAPH_ERR_BAD_PARAMETERS=1,
    NVGRAPH_ERR_UNKNOWN=2,
    NVGRAPH_ERR_CUDA_FAILURE=3,
    NVGRAPH_ERR_THRUST_FAILURE=4,
    NVGRAPH_ERR_IO=5,
    NVGRAPH_ERR_NOT_IMPLEMENTED=6,
    NVGRAPH_ERR_NO_MEMORY=7,
    NVGRAPH_ERR_NOT_CONVERGED=8
};

// define our own bad_alloc so we can set its .what()
class nvgraph_exception: public std::exception
{
  public:
    inline nvgraph_exception(const std::string &w, const std::string &where, const std::string &trace, NVGRAPH_ERROR reason) : m_trace(trace), m_what(w), m_reason(reason), m_where(where)
    {
    }

    inline virtual ~nvgraph_exception(void) throw () {};

    inline virtual const char *what(void) const throw()
    {
      return m_what.c_str();
    }
    inline virtual const char *where(void) const throw()
    {
      return m_where.c_str();
    }
    inline virtual const char *trace(void) const throw()
    {
      return m_trace.c_str();
    }
    inline virtual NVGRAPH_ERROR reason(void) const throw()
    {
      return m_reason;
    }


  private:
    std::string  m_trace;
    std::string  m_what;
    NVGRAPH_ERROR m_reason;
    std::string  m_where;
}; // end bad_alloc
  

int NVGRAPH_GetErrorString( NVGRAPH_ERROR error, char* buffer, int buf_len);

/********************************************************
 * Prints the error message, the stack trace, and exits
 * ******************************************************/
#define FatalError(s, reason) {                                                 \
  std::stringstream _where;                                                     \
  _where << WHERE ;                                                             \
  std::stringstream _trace;                                                     \
  printStackTrace(_trace);                                                      \
  throw nvgraph_exception(std::string(s) + "\n", _where.str(), _trace.str(), reason); \
}

#undef cudaCheckError
#if defined(DEBUG) || defined(VERBOSE_DIAG)
#define cudaCheckError() {                                              \
  cudaError_t e=cudaGetLastError();                                     \
  if(e!=cudaSuccess) {                                                  \
    std::stringstream _error;                                           \
    _error << "Cuda failure: '" << cudaGetErrorString(e) << "'";        \
    FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);                 \
  }                                                                     \
}
#else // NO DEBUG
#define cudaCheckError()                                                      \
    {                                                                         \
        cudaError_t __e = cudaGetLastError();                                 \
        if (__e != cudaSuccess) {                                             \
            FatalError("", NVGRAPH_ERR_CUDA_FAILURE);                         \
        }                                                                     \
    }
#endif

#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess)                                                \
        {                                                                     \
            std::stringstream _error;                                         \
            _error << "CUDA Runtime failure: '#" << _e << "'";                \
            FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);               \
        }                                                                     \
    }

#define CHECK_CURAND(call)                                                    \
    {                                                                         \
        curandStatus_t _e = (call);                                           \
        if (_e != CURAND_STATUS_SUCCESS)                                      \
        {                                                                     \
            std::stringstream _error;                                         \
            _error << "CURAND failure: '#" << _e << "'";                      \
            FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);               \
        }                                                                     \
    }

#define CHECK_CUBLAS(call)                                                    \
    {                                                                         \
        cublasStatus_t _e = (call);                                           \
        if (_e != CUBLAS_STATUS_SUCCESS)                                      \
        {                                                                     \
            std::stringstream _error;                                         \
            _error << "CUBLAS failure: '#" << _e << "'";                      \
            FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);               \
        }                                                                     \
    }

#define CHECK_CUSPARSE(call)                                                  \
    {                                                                         \
        cusparseStatus_t _e = (call);                                         \
        if (_e != CUSPARSE_STATUS_SUCCESS)                                    \
        {                                                                     \
            std::stringstream _error;                                         \
            _error << "CURAND failure: '#" << _e << "'";                      \
            FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);               \
        }                                                                     \
    }

#define CHECK_CUSOLVER(call)                                                  \
    {                                                                         \
        cusolverStatus_t _e = (call);                                         \
        if (_e != CUSOLVER_STATUS_SUCCESS)                                    \
        {                                                                     \
            std::stringstream _error;                                         \
            _error << "CURAND failure: '#" << _e << "'";                      \
            FatalError(_error.str(), NVGRAPH_ERR_CUDA_FAILURE);               \
        }                                                                     \
    }
} // namespace nvgraph

