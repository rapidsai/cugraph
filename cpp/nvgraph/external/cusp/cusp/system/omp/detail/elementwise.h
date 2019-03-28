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

#include <cusp/detail/format.h>
#include <cusp/detail/temporary_array.h>

#include <thrust/scan.h>
#include <cusp/system/cpp/detail/elementwise.h>

namespace cusp
{
namespace system
{
namespace omp
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(omp::execution_policy<DerivedPolicy>& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                 MatrixType3& C,
                 BinaryFunction op,
                 cusp::csr_format,
                 cusp::csr_format,
                 cusp::csr_format)
{
    if(A.num_rows != B.num_rows || A.num_cols != B.num_cols)
        throw cusp::invalid_input_exception("matrix dimensions do not match");

    //Method that works for duplicate and/or unsorted indices
    typedef typename MatrixType3::index_type IndexType;
    typedef typename MatrixType3::value_type ValueType;

    //MW: compute number of nonzeros in each row of C
    cusp::detail::temporary_array<IndexType, DerivedPolicy> C_row_offsets(exec, A.num_rows + 1);

    C_row_offsets[0] = 0;

    #pragma omp parallel for
    for(int i = 0; i < int(A.num_rows); i++)
    {
        size_t num_nonzeros_in_row_i = B.row_offsets[i + 1]-B.row_offsets[i];

        for(IndexType jj = A.row_offsets[i]; jj < A.row_offsets[i + 1]; jj++)
        {
            IndexType j = A.column_indices[jj];
            bool different = true;

            for(IndexType kk = B.row_offsets[i]; kk < B.row_offsets[i + 1]; kk++)
            {
                IndexType k = B.column_indices[kk];
                if(j == k)
                {
                    different = false;
                    break;
                }
            }
            if(different) num_nonzeros_in_row_i++;
        }
        C_row_offsets[i + 1] = num_nonzeros_in_row_i;
    } //omp for

    //MW: now transform to offsets and resize column and values
    thrust::inclusive_scan(exec, C_row_offsets.begin(), C_row_offsets.end(), C_row_offsets.begin());

    size_t num_entries_in_C = C_row_offsets[A.num_rows];

    cusp::detail::temporary_array<IndexType, DerivedPolicy> C_column_indices(exec, num_entries_in_C); //MW: cheap
    cusp::detail::temporary_array<ValueType, DerivedPolicy> C_values(exec, num_entries_in_C); //MW: cheap

    #pragma omp parallel
    {
        cusp::detail::temporary_array<IndexType, DerivedPolicy>  next(exec, A.num_cols, IndexType(-1));
        cusp::detail::temporary_array<ValueType, DerivedPolicy> A_row(exec, A.num_cols, ValueType(0));
        cusp::detail::temporary_array<ValueType, DerivedPolicy> B_row(exec, A.num_cols, ValueType(0));

        #pragma omp for
        for(int i = 0; i < int(A.num_rows); i++)
        {
            IndexType head   = -2;
            IndexType length =  0;

            //add a row of A to A_row
            IndexType i_start = A.row_offsets[i];
            IndexType i_end   = A.row_offsets[i + 1];

            for(int jj = int(i_start); jj < int(i_end); jj++)
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

            for(int jj = int(i_start); jj < int(i_end); jj++)
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
            // MW iterate through list without destroying it
            IndexType j = C_row_offsets[i];

            for(int jj = 0; jj < int(length); jj++)
            {
                ValueType result = op( A_row[head], B_row[head]);
                C_column_indices[j + jj] = head;
                C_values[j+jj] = result;

                IndexType prev = head;
                head = next[head];
                next[prev]  = -1;

                A_row[prev] =  0;
                B_row[prev] =  0;
            }
        } //omp for
    } //omp parallel

    C.resize(A.num_rows, A.num_cols, num_entries_in_C);

    cusp::copy(exec, C_row_offsets, C.row_offsets);
    cusp::copy(exec, C_column_indices, C.column_indices);
    cusp::copy(exec, C_values, C.values);
} // csr_transform_elementwise

} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace cusp

