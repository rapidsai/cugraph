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

/*! \file elementwise.h
 *  \brief Sequential implementations of elementwise algorithms.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>
#include <cusp/copy.h>
#include <cusp/csr_matrix.h>

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
          typename BinaryFunction>
void elementwise(thrust::cpp::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                 MatrixType3& C,
                 BinaryFunction op,
                 cusp::csr_format,
                 cusp::csr_format,
                 cusp::csr_format)
{
    //Method that works for duplicate and/or unsorted indices
    typedef typename MatrixType3::index_type   IndexType;
    typedef typename MatrixType3::value_type   ValueType;
    typedef typename MatrixType3::memory_space MemorySpace;

    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("matrix dimensions do not match");

    cusp::detail::temporary_array<IndexType, DerivedPolicy>  next(exec, A.num_cols, IndexType(-1));
    cusp::detail::temporary_array<ValueType, DerivedPolicy> A_row(exec, A.num_cols, ValueType(0));
    cusp::detail::temporary_array<ValueType, DerivedPolicy> B_row(exec, A.num_cols, ValueType(0));

    cusp::csr_matrix<IndexType,ValueType,MemorySpace> temp(A.num_rows, A.num_cols, A.num_entries + B.num_entries);

    size_t nnz = 0;

    temp.row_offsets[0] = 0;

    for(size_t i = 0; i < A.num_rows; i++)
    {
        IndexType head   = -2;
        IndexType length =  0;

        //add a row of A to A_row
        IndexType i_start = A.row_offsets[i];
        IndexType i_end   = A.row_offsets[i + 1];
        for(IndexType jj = i_start; jj < i_end; jj++)
        {
            IndexType j = A.column_indices[jj];

            A_row[j] += A.values[jj];

            if(next[j] == -1) {
                next[j] = head;
                head = j;
                length++;
            }
        }

        //add a row of B to B_row
        i_start = B.row_offsets[i];
        i_end   = B.row_offsets[i + 1];
        for(IndexType jj = i_start; jj < i_end; jj++)
        {
            IndexType j = B.column_indices[jj];

            B_row[j] += B.values[jj];

            if(next[j] == -1) {
                next[j] = head;
                head = j;
                length++;
            }
        }

        // scan through columns where A or B has
        // contributed a non-zero entry
        for(IndexType jj = 0; jj < length; jj++)
        {
            ValueType result = op(A_row[head], B_row[head]);

            if(result != ValueType(0))
            {
                temp.column_indices[nnz] = head;
                temp.values[nnz]         = result;
                nnz++;
            }

            IndexType prev = head;
            head = next[head];
            next[prev]  = -1;

            A_row[prev] =  0;
            B_row[prev] =  0;
        }

        temp.row_offsets[i + 1] = nnz;
    }

    // TODO replace with destructive assignment?

    temp.resize(A.num_rows, A.num_cols, nnz);
    cusp::copy(exec, temp, C);
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

