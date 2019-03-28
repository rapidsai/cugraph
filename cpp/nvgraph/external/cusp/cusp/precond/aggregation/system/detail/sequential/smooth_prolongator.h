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
#include <cusp/detail/execution_policy.h>
#include <cusp/detail/temporary_array.h>

#include <cusp/array1d.h>
#include <cusp/elementwise.h>
#include <cusp/format_utils.h>

#include <cusp/blas/blas.h>

#include <cusp/system/detail/sequential/execution_policy.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{
namespace prolongator_detail
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega,
                        cusp::csr_format)
{
    typedef typename MatrixType3::index_type   IndexType;
    typedef typename MatrixType3::value_type   ValueType;
    typedef typename MatrixType3::memory_space MemorySpace;

    cusp::detail::temporary_array<ValueType, DerivedPolicy> D(exec, S.num_rows);
    cusp::extract_diagonal(exec, S, D);

    // create D_inv_S by copying S then scaling
    cusp::csr_matrix<IndexType,ValueType,MemorySpace> D_inv_S(S);

    // scale the rows of D_inv_S by D^-1
    for (size_t row = 0; row < D_inv_S.num_rows; row++)
    {
        const IndexType row_start = D_inv_S.row_offsets[row];
        const IndexType row_end = D_inv_S.row_offsets[row + 1];
        const ValueType diagonal = D[row];

        for (IndexType index = row_start; index < row_end; index++)
            D_inv_S.values[index] /= diagonal;
    }

    const ValueType lambda = omega / rho_Dinv_S;
    cusp::blas::scal(exec, D_inv_S.values, lambda);

    cusp::csr_matrix<IndexType,ValueType,MemorySpace> temp;
    cusp::multiply(exec, D_inv_S, T, temp);
    cusp::subtract(T, temp, P);
}

template <typename DerivedPolicy,
         typename MatrixType1,
         typename MatrixType2,
         typename MatrixType3>
void smooth_prolongator(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega,
                        cusp::known_format)
{
    typedef typename MatrixType1::index_type                      IndexType;
    typedef typename MatrixType1::const_coo_view_type             CooViewType1;
    typedef typename MatrixType2::const_coo_view_type             CooViewType2;
    typedef typename cusp::detail::as_csr_type<MatrixType3>::type CsrType;

    CooViewType1 S_(S);
    CooViewType2 T_(T);
    CsrType P_;

    cusp::detail::temporary_array<IndexType, DerivedPolicy> S_row_offsets(exec, S.num_rows + 1);
    cusp::detail::temporary_array<IndexType, DerivedPolicy> T_row_offsets(exec, T.num_rows + 1);

    cusp::indices_to_offsets(exec, S_.row_indices, S_row_offsets);
    cusp::indices_to_offsets(exec, T_.row_indices, T_row_offsets);

    smooth_prolongator(exec,
                       cusp::make_csr_matrix_view(S.num_rows, S.num_cols, S.num_entries,
                                                  cusp::make_array1d_view(S_row_offsets),
                                                  cusp::make_array1d_view(S_.column_indices),
                                                  cusp::make_array1d_view(S_.values)),
                       cusp::make_csr_matrix_view(T.num_rows, T.num_cols, T.num_entries,
                                                  cusp::make_array1d_view(T_row_offsets),
                                                  cusp::make_array1d_view(T_.column_indices),
                                                  cusp::make_array1d_view(T_.values)),
                       P_,
                       rho_Dinv_S,
                       omega,
                       cusp::csr_format());

    cusp::convert(exec, P_, P);
}

} // end namespace prolongator_detail

template <typename DerivedPolicy,
         typename MatrixType1,
         typename MatrixType2,
         typename MatrixType3>
void smooth_prolongator(thrust::cpp::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                        MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega)
{
    typedef typename MatrixType1::format Format;

    Format format;

    prolongator_detail::smooth_prolongator(exec, S, T, P, rho_Dinv_S, omega, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

