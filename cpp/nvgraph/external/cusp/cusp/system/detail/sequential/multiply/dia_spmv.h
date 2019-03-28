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

#include <cusp/functional.h>
#include <cusp/system/detail/sequential/execution_policy.h>

#include <algorithm>

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
              cusp::dia_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type  IndexType;
    typedef typename VectorType2::value_type ValueType;

    const size_t num_diagonals = A.values.num_cols;

    for(size_t i = 0; i < A.num_rows; i++)
        y[i] = initialize(y[i]);

    for(size_t i = 0; i < num_diagonals; i++)
    {
        const IndexType k = A.diagonal_offsets[i];

        const size_t i_start = std::max<IndexType>(0, -k);
        const size_t j_start = std::max<IndexType>(0,  k);

        // number of elements to process in this diagonal
        const size_t N = std::min(A.num_rows - i_start, A.num_cols - j_start);

        for(size_t n = 0; n < N; n++)
        {
            const ValueType Aij = A.values(i_start + n, i);

            const ValueType  xj = x[j_start + n];
                  ValueType  yi = y[i_start + n];

            y[i_start + n] = reduce(yi, combine(Aij, xj));
        }
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp
