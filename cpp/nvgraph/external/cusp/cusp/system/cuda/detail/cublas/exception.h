/*
 *  Copyright 2008-2014 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file exception.h
 *  \brief Cusp exceptions
 */

#pragma once

#include <cusp/detail/config.h>

#include <cublas_v2.h>

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace cublas
{

class cublas_exception : public cusp::exception
{
public:
    cublas_exception(const std::string name,
                     const cublasStatus_t stat)
                    : exception(name + ": ")
    {
        if(stat == CUBLAS_STATUS_NOT_INITIALIZED)
          message += "CUBLAS_STATUS_NOT_INITIALIZED";
        else if(stat == CUBLAS_STATUS_ALLOC_FAILED)
          message += "CUBLAS_STATUS_ALLOC_FAILED";
        else if(stat == CUBLAS_STATUS_INVALID_VALUE)
          message += "CUBLAS_STATUS_INVALID_VALUE";
        else if(stat == CUBLAS_STATUS_ARCH_MISMATCH)
          message += "CUBLAS_STATUS_ARCH_MISMATCH";
        else if(stat == CUBLAS_STATUS_MAPPING_ERROR)
          message += "CUBLAS_STATUS_MAPPING_ERROR";
        else if(stat == CUBLAS_STATUS_EXECUTION_FAILED)
          message += "CUBLAS_STATUS_EXECUTION_FAILED";
        else if(stat == CUBLAS_STATUS_INTERNAL_ERROR)
          message += "CUBLAS_STATUS_INTERNAL_ERROR";
        else if(stat == CUBLAS_STATUS_NOT_SUPPORTED)
          message += "CUBLAS_STATUS_NOT_SUPPORTED";
        else if(stat == CUBLAS_STATUS_LICENSE_ERROR)
          message += "CUBLAS_STATUS_LICENSE_ERROR";
        else
          message += "Unknown cublasStatus_t";
    }
};

} // end namespace cublas
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

