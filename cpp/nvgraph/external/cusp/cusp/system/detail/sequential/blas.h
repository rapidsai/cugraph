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

#pragma once

#include <cusp/detail/config.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3,
          typename ScalarType1,
          typename ScalarType2>
void gemm(thrust::cpp::execution_policy<DerivedPolicy>& exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const ScalarType1 alpha,
          const ScalarType2 beta)
{
    typedef typename Array2d1::value_type ValueType;

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = ValueType(0);

            for(size_t k = 0; k < A.num_cols; k++)
                v += A(i,k) * B(k,j);

            C(i,j) = v;
        }
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

