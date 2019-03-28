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
#include <cusp/detail/temporary_array.h>

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
          typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(thrust::cpp::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array2d_format,
              cusp::array2d_format)
{
    typedef typename MatrixType::index_type	                    IndexType;
    typedef typename VectorType2::values_array_type::value_type ValueType;

    for(size_t i = 0; i < A.num_rows; i++)
    {
        const IndexType row_start = A.row_offsets[i];
        const IndexType row_end   = A.row_offsets[i + 1];

		    cusp::detail::temporary_array<ValueType, DerivedPolicy> accumulator(exec, x.num_cols);

        for(size_t k = 0; k < x.num_cols; k++)
            accumulator[k] = initialize(y(i,k));

        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType j   = A.column_indices[jj];
            const ValueType Aij = A.values[jj];

            for(size_t k = 0; k < x.num_cols; k++)
            {
              const ValueType xj  = x(j,k);
              accumulator[k] = reduce(accumulator[k], combine(Aij, xj));
            }
        }

        for(size_t k = 0; k < x.num_cols; k++)
            y(i,k) = accumulator[k];
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

