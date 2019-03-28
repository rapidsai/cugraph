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
#include <cusp/detail/format.h>

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
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(thrust::cpp::execution_policy<DerivedPolicy>& exec,
              const MatrixType1& A,
              const MatrixType2& B,
              MatrixType3& C,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::array2d_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename MatrixType3::value_type ValueType;

    C.resize(A.num_rows, B.num_cols);

    for(size_t i = 0; i < C.num_rows; i++)
    {
        for(size_t j = 0; j < C.num_cols; j++)
        {
            ValueType v = initialize(C(i,j));

            for(size_t k = 0; k < A.num_cols; k++)
                v = reduce(v, combine(A(i,k), B(k,j)));

            C(i,j) = v;
        }
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

