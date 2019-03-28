/*
 *  Copyright 2008-2012 NVIDIA Corporation
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

#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/cuda/detail/execution_policy.h>

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

template<typename DerivedPolicy>
class execute_with_cublas_base
  : public cusp::cuda::execution_policy<DerivedPolicy>
{
  public:

    execute_with_cublas_base(void) {}

    execute_with_cublas_base(const cublasHandle_t& handle)
      : m_handle(handle)
    {}

    __host__ __device__
    DerivedPolicy with(const cublasHandle_t& h) const
    {
      // create a copy of *this to return
      // make sure it is the derived type
      DerivedPolicy result = thrust::detail::derived_cast(*this);

      // change the result's cublas handle to h
      result.set_handle(h);

      return result;
    }

  private:

    __host__ __device__
    friend inline cublasHandle_t const& handle(const execute_with_cublas_base &exec)
    {
      return exec.m_handle;
    }

    __host__ __device__
    inline void set_handle(const cublasHandle_t &h)
    {
      m_handle = h;
    }

    cublasHandle_t m_handle;
};

class execute_with_cublas
  : public execute_with_cublas_base<execute_with_cublas>
{
    typedef execute_with_cublas_base<execute_with_cublas> super_t;

  public:

    __host__ __device__
    inline execute_with_cublas(const cublasHandle_t& h)
      : super_t(h)
    {}
};

} // end namespace cublas
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

