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

#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/elementwise.h>
#include <cusp/sort.h>

#include <cusp/blas/blas.h>
#include <cusp/eigen/spectral_radius.h>

#include <cusp/detail/temporary_array.h>

#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cusp/precond/aggregation/system/detail/sequential/smooth_prolongator.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(thrust::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega,
                        cusp::coo_format)
{
    using namespace thrust::placeholders;

    typedef typename MatrixType3::index_type IndexType;
    typedef typename MatrixType3::value_type ValueType;

    // TODO handle case with unaggregated nodes more gracefully
    if (T.num_entries == T.num_rows)
    {
        const ValueType lambda = omega / rho_Dinv_S;

        // temp <- -lambda * S(i,j) * T(j,k)
        MatrixType3 temp(S.num_rows, T.num_cols, S.num_entries + T.num_entries);
        thrust::copy(exec, S.row_indices.begin(), S.row_indices.end(), temp.row_indices.begin());
        thrust::gather(exec, S.column_indices.begin(), S.column_indices.end(), T.column_indices.begin(), temp.column_indices.begin());
        thrust::transform(exec,
                          S.values.begin(), S.values.end(),
                          thrust::make_permutation_iterator(T.values.begin(), S.column_indices.begin()),
                          temp.values.begin(),
                          -lambda * _1 * _2);

        // temp <- D^-1
        {
            cusp::detail::temporary_array<ValueType, DerivedPolicy> D(exec, S.num_rows);
            cusp::extract_diagonal(exec, S, D);
            thrust::transform(exec,
                              temp.values.begin(), temp.values.begin() + S.num_entries,
                              thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                              temp.values.begin(),
                              thrust::divides<ValueType>());
        }

        // temp <- temp + T
        thrust::copy(exec, T.row_indices.begin(),    T.row_indices.end(),    temp.row_indices.begin()    + S.num_entries);
        thrust::copy(exec, T.column_indices.begin(), T.column_indices.end(), temp.column_indices.begin() + S.num_entries);
        thrust::copy(exec, T.values.begin(),         T.values.end(),         temp.values.begin()         + S.num_entries);

        // sort by (I,J)
        cusp::sort_by_row_and_column(exec, temp.row_indices, temp.column_indices, temp.values, 0, T.num_rows, 0, T.num_cols);

        // compute unique number of nonzeros in the output
        // throws a warning at compile (warning: expression has no effect)
        IndexType NNZ = thrust::inner_product(exec,
                                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end (),  temp.column_indices.end()))   - 1,
                                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())) + 1,
                                              IndexType(0),
                                              thrust::plus<IndexType>(),
                                              thrust::not_equal_to< thrust::tuple<IndexType,IndexType> >()) + 1;

        // allocate space for output
        P.resize(temp.num_rows, temp.num_cols, NNZ);

        // sum values with the same (i,j)
        thrust::reduce_by_key(exec,
                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.begin(), temp.column_indices.begin())),
                              thrust::make_zip_iterator(thrust::make_tuple(temp.row_indices.end(),   temp.column_indices.end())),
                              temp.values.begin(),
                              thrust::make_zip_iterator(thrust::make_tuple(P.row_indices.begin(), P.column_indices.begin())),
                              P.values.begin(),
                              thrust::equal_to< thrust::tuple<IndexType,IndexType> >(),
                              thrust::plus<ValueType>());
    }
    else
    {
        cusp::detail::temporary_array<ValueType, DerivedPolicy> D(exec, S.num_rows);
        cusp::extract_diagonal(exec, S, D);

        // create D_inv_S by copying S then scaling
        MatrixType3 D_inv_S(S);

        // scale the rows of D_inv_S by D^-1
        thrust::transform(exec,
                          D_inv_S.values.begin(),
                          D_inv_S.values.begin() + S.num_entries,
                          thrust::make_permutation_iterator(D.begin(), S.row_indices.begin()),
                          D_inv_S.values.begin(),
                          thrust::divides<ValueType>());

        const ValueType lambda = omega / rho_Dinv_S;
        cusp::blas::scal(exec, D_inv_S.values, lambda);

        MatrixType3 temp;
        cusp::multiply(exec, D_inv_S, T, temp);
        cusp::subtract(T, temp, P);
    }
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(thrust::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega,
                        cusp::known_format)
{
    typedef typename MatrixType1::const_coo_view_type CooViewType1;
    typedef typename MatrixType2::const_coo_view_type CooViewType2;
    typedef typename cusp::detail::as_coo_type<MatrixType3>::type CooType3;

    CooViewType1 S_(S);
    CooViewType2 T_(T);
    CooType3 P_;

    smooth_prolongator(exec, S_, T_, P_, rho_Dinv_S, omega, cusp::coo_format());

    cusp::convert(exec, P_, P);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void smooth_prolongator(thrust::execution_policy<DerivedPolicy> &exec,
                        const MatrixType1& S,
                        const MatrixType2& T,
                              MatrixType3& P,
                        const double rho_Dinv_S,
                        const double omega)
{
    typedef typename MatrixType1::format Format;

    Format format;

    smooth_prolongator(exec, S, T, P, rho_Dinv_S, omega, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

