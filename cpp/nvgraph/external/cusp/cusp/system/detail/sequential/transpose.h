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

/*! \file transpose.h
 *  \brief Sequential implementations of transpose algorithms.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace system
{
namespace detail
{
namespace sequential
{

// COO format
template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(thrust::cpp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               cusp::coo_format, cusp::coo_format)
{
    typedef typename MatrixType2::index_type IndexType;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    cusp::detail::temporary_array<IndexType, DerivedPolicy> starting_pos(exec, A.num_cols + 1, 0);

    if( A.num_entries > 0 )
    {
        for( size_t i = 0; i < A.num_entries; i++ )
        {
            IndexType col = A.column_indices[i];
            starting_pos[col + 1]++;
        }

        for( size_t i = 1; i < A.num_cols + 1; i++ )
        {
            starting_pos[i] += starting_pos[i - 1];
        }

        for( size_t i = 0; i < A.num_entries; i++ )
        {
            IndexType col = A.column_indices[i];
            IndexType j = starting_pos[col]++;

            At.row_indices[j] = A.column_indices[i];
            At.column_indices[j] = A.row_indices[i];
            At.values[j] = A.values[i];
        }
    }
}

// CSR format
template <typename DerivedPolicy, typename MatrixType1, typename MatrixType2>
void transpose(thrust::cpp::execution_policy<DerivedPolicy>& exec,
               const MatrixType1& A, MatrixType2& At,
               cusp::csr_format, cusp::csr_format)
{
    typedef typename MatrixType2::index_type IndexType;

    At.resize(A.num_cols, A.num_rows, A.num_entries);

    if( A.num_entries > 0 )
    {
        for( size_t i = 0; i < At.num_rows + 1; i++ )
        {
            At.row_offsets[i] = 0;
        }

        for( size_t i = 0; i < At.num_entries; i++ )
        {
            IndexType col = A.column_indices[i];
            At.row_offsets[col + 1]++;
        }

        for( size_t i = 1; i < At.num_rows + 1; i++ )
        {
            At.row_offsets[i] += At.row_offsets[i - 1];
        }

        cusp::detail::temporary_array<IndexType, DerivedPolicy> starting_pos(exec, At.row_offsets);

        for( size_t row = 0; row < A.num_rows; row++ )
        {
            IndexType row_start = A.row_offsets[row];
            IndexType row_end   = A.row_offsets[row + 1];

            for( IndexType i = row_start; i < row_end; i++ )
            {
                IndexType col = A.column_indices[i];
                IndexType j   = starting_pos[col]++;

                At.column_indices[j] = row;
                At.values[j] = A.values[i];
            }
        }
    }
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

