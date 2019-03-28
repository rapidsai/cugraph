/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#include <thrust/detail/config.h>
#include <thrust/reduce.h>

#include <thrust/system/detail/generic/tag.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cusp/detail/format.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>

#include <cusp/detail/utils.h>
#include <cusp/detail/array2d_format_utils.h>

namespace cusp
{
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
		      typename MatrixType,
          typename VectorType1,
          typename VectorType2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(omp::execution_policy<DerivedPolicy>& exec,
              const MatrixType& A,
              const VectorType1& x,
              VectorType2& y,
              UnaryFunction   initialize,
              BinaryFunction1 combine,
              BinaryFunction2 reduce,
              cusp::csr_format,
              cusp::array1d_format,
              cusp::array1d_format)
{
    typedef typename MatrixType::index_type  IndexType;
    typedef typename VectorType2::value_type ValueType;

	int N = A.num_rows;

    #pragma omp parallel for
    for(int i = 0; i < N; i++)
    {
        const IndexType row_start = A.row_offsets[i];
        const IndexType row_end   = A.row_offsets[i+1];

        ValueType accumulator = initialize(y[i]);

        for (IndexType jj = row_start; jj < row_end; jj++)
        {
            const IndexType j   = A.column_indices[jj];
            const ValueType Aij = A.values[jj];
            const ValueType xj  = x[j];

            accumulator = reduce(accumulator, combine(Aij, xj));
        }

        y[i] = accumulator;
    }
}

} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace cusp
