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
          typename Array1,
          typename Array2,
          typename Array3,
          typename Array4>
size_t spmm_csr_pass1(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                      const size_t num_rows, const size_t num_cols,
                      const Array1& A_row_offsets, const Array2& A_column_indices,
                      const Array3& B_row_offsets, const Array4& B_column_indices)
{
    typedef typename Array1::value_type IndexType1;
    typedef typename Array2::value_type IndexType2;

    cusp::detail::temporary_array<size_t, DerivedPolicy> mask(exec, num_cols, static_cast<size_t>(-1));

    // Compute nnz in C (including explicit zeros)
    size_t num_nonzeros = 0;

    for(size_t i = 0; i < num_rows; i++)
    {
        for(IndexType1 jj = A_row_offsets[i]; jj < A_row_offsets[i+1]; jj++)
        {
            IndexType1 j = A_column_indices[jj];

            for(IndexType2 kk = B_row_offsets[j]; kk < B_row_offsets[j+1]; kk++)
            {
                IndexType2 k = B_column_indices[kk];

                if(mask[k] != i)
                {
                    mask[k] = i;
                    num_nonzeros++;
                }
            }
        }
    }

    return num_nonzeros;
}

template <typename DerivedPolicy,
          typename Array1, typename Array2, typename Array3,
          typename Array4, typename Array5, typename Array6,
          typename Array7, typename Array8, typename Array9,
          typename UnaryFunction, typename BinaryFunction1, typename BinaryFunction2>
size_t spmm_csr_pass2(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                      const size_t num_rows, const size_t num_cols,
                      const Array1& A_row_offsets, const Array2& A_column_indices, const Array3& A_values,
                      const Array4& B_row_offsets, const Array5& B_column_indices, const Array6& B_values,
                      Array7& C_row_offsets,       Array8& C_column_indices,       Array9& C_values,
                      UnaryFunction initialize,    BinaryFunction1 combine,        BinaryFunction2 reduce)
{
    typedef typename Array7::value_type IndexType;
    typedef typename Array9::value_type ValueType;

    size_t num_nonzeros = 0;

    const IndexType unseen = static_cast<IndexType>(-1);
    const IndexType init   = static_cast<IndexType>(-2);

    // Compute entries of C
    cusp::detail::temporary_array<IndexType, DerivedPolicy> next(exec, num_cols, unseen);
    cusp::detail::temporary_array<ValueType, DerivedPolicy> sums(exec, num_cols, ValueType(0));

    num_nonzeros = 0;

    C_row_offsets[0] = 0;

    for(size_t i = 0; i < num_rows; i++)
    {
        IndexType head   = init;
        IndexType length =    0;

        IndexType jj_start = A_row_offsets[i];
        IndexType jj_end   = A_row_offsets[i+1];

        for(IndexType jj = jj_start; jj < jj_end; jj++)
        {
            IndexType j = A_column_indices[jj];
            ValueType v = A_values[jj];

            IndexType kk_start = B_row_offsets[j];
            IndexType kk_end   = B_row_offsets[j+1];

            for(IndexType kk = kk_start; kk < kk_end; kk++)
            {
                IndexType k = B_column_indices[kk];

                sums[k] = reduce(sums[k], combine(v, B_values[kk]));

                if(next[k] == unseen)
                {
                    next[k] = head;
                    head  = k;
                    length++;
                }
            }
        }

        for(IndexType jj = 0; jj < length; jj++)
        {
            if(ValueType(sums[head]) != ValueType(0))
            {
                C_column_indices[num_nonzeros] = head;
                C_values[num_nonzeros]         = sums[head];
                num_nonzeros++;
            }

            IndexType temp = head;
            head = next[head];

            // clear arrays
            next[temp] = unseen;
            sums[temp] = ValueType(0);
        }

        C_row_offsets[i+1] = num_nonzeros;
    }

    // XXX note: entries of C are unsorted within each row

    return num_nonzeros;
}

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
              cusp::csr_format,
              cusp::csr_format,
              cusp::csr_format)
{
    typedef typename MatrixType3::index_type IndexType;

    IndexType num_nonzeros =
        spmm_csr_pass1(exec,
                       A.num_rows, B.num_cols,
                       A.row_offsets, A.column_indices,
                       B.row_offsets, B.column_indices);

    // Resize output
    C.resize(A.num_rows, B.num_cols, num_nonzeros);

    num_nonzeros =
        spmm_csr_pass2(exec,
                       A.num_rows, B.num_cols,
                       A.row_offsets, A.column_indices, A.values,
                       B.row_offsets, B.column_indices, B.values,
                       C.row_offsets, C.column_indices, C.values,
                       initialize, combine, reduce);

    // Resize output again since pass2 omits explict zeros
    C.resize(A.num_rows, B.num_cols, num_nonzeros);
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

