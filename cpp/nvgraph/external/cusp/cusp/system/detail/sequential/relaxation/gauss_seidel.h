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

template<typename DerivedPolicy,
         typename MatrixType,
         typename ArrayType1,
         typename ArrayType2>
void gauss_seidel_indexed(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                          const MatrixType& A,
                                ArrayType1&  x,
                          const ArrayType1&  b,
                          const ArrayType2& indices,
                          const int row_start,
                          const int row_stop,
                          const int row_step)
{
    typedef typename ArrayType1::value_type V;
    typedef typename ArrayType2::value_type I;

    for(int i = row_start; i != row_stop; i += row_step)
    {
        I inew  = indices[i];
        I start = A.row_offsets[inew];
        I end   = A.row_offsets[inew + 1];
        V rsum  = 0;
        V diag  = 0;

        for(I jj = start; jj < end; ++jj)
        {
            I j = A.column_indices[jj];
            if (inew == j)
            {
                diag = A.values[jj];
            }
            else
            {
                rsum += A.values[jj]*x[j];
            }
        }

        if (diag != 0)
        {
            x[inew] = (b[inew] - rsum)/diag;
        }
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

