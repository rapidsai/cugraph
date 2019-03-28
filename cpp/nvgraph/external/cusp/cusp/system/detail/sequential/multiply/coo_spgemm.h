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
#include <cusp/format_utils.h>

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
              cusp::coo_format,
              cusp::coo_format,
              cusp::coo_format)
{
    typedef typename MatrixType1::index_type IndexType1;
    typedef typename MatrixType2::index_type IndexType2;
    typedef typename MatrixType3::index_type IndexType3;

    // allocate storage for row offsets for A, B, and C
    cusp::detail::temporary_array<IndexType1, DerivedPolicy> A_row_offsets(exec, A.num_rows + 1);
    cusp::detail::temporary_array<IndexType2, DerivedPolicy> B_row_offsets(exec, B.num_rows + 1);
    cusp::detail::temporary_array<IndexType3, DerivedPolicy> C_row_offsets(exec, A.num_rows + 1);

    // compute row offsets for A and B
    cusp::indices_to_offsets(exec, A.row_indices, A_row_offsets);
    cusp::indices_to_offsets(exec, B.row_indices, B_row_offsets);

    IndexType3 estimated_nonzeros =
        spmm_csr_pass1(exec, A.num_rows, B.num_cols,
                       A_row_offsets, A.column_indices,
                       B_row_offsets, B.column_indices);

    // Resize output
    C.resize(A.num_rows, B.num_cols, estimated_nonzeros);

    IndexType3 true_nonzeros =
        spmm_csr_pass2(exec, A.num_rows, B.num_cols,
                       A_row_offsets, A.column_indices, A.values,
                       B_row_offsets, B.column_indices, B.values,
                       C_row_offsets, C.column_indices, C.values,
                       initialize, combine, reduce);

    // true_nonzeros may be less than estimated_nonzeros
    C.resize(A.num_rows, B.num_cols, true_nonzeros);

    cusp::offsets_to_indices(exec, C_row_offsets, C.row_indices);
}

} // end namespace sequential
} // end namespace detail
} // end namespace system
} // end namespace cusp

