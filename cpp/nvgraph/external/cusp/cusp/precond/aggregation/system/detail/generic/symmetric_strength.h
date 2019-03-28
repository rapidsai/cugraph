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


#include <cusp/array1d.h>
#include <cusp/complex.h>
#include <cusp/convert.h>
#include <cusp/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/format_utils.h>
#include <cusp/functional.h>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#include <cusp/precond/aggregation/system/detail/sequential/symmetric_strength.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{
namespace detail
{

template <typename ValueType>
struct is_strong_connection
{
    typedef typename cusp::norm_type<ValueType>::type NormType;

    NormType theta;

    is_strong_connection(const NormType theta) : theta(theta) {}

    template <typename Tuple>
    __host__ __device__
    bool operator()(const Tuple& t) const
    {
        NormType nAij = cusp::abs(thrust::get<0>(t));
        NormType nAii = cusp::abs(thrust::get<1>(t));
        NormType nAjj = cusp::abs(thrust::get<2>(t));

        // square everything to eliminate the sqrt()
        return (nAij*nAij) >= ((theta*theta) * (nAii * nAjj));
    }
};

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void symmetric_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                      const MatrixType1& A,
                                            MatrixType2& S,
                                      const double theta,
                                      cusp::coo_format)
{
    typedef typename MatrixType1::index_type   IndexType;
    typedef typename MatrixType1::value_type   ValueType;

    cusp::detail::temporary_array<ValueType, DerivedPolicy> diagonal(exec, A.num_rows);
    cusp::extract_diagonal(exec, A, diagonal);

    is_strong_connection<ValueType> pred(theta);

    cusp::detail::temporary_array<bool, DerivedPolicy> copyflags(exec, A.num_entries);

    // this is just zipping up (A[i,j],A[i,i],A[j,j]) and applying is_strong_connection to each tuple
    thrust::transform(exec,
                      thrust::make_zip_iterator(thrust::make_tuple(A.values.begin(),
                                                thrust::make_permutation_iterator(diagonal.begin(), A.row_indices.begin()),
                                                thrust::make_permutation_iterator(diagonal.begin(), A.column_indices.begin()))),
                      thrust::make_zip_iterator(thrust::make_tuple(A.values.begin(),
                                                thrust::make_permutation_iterator(diagonal.begin(), A.row_indices.begin()),
                                                thrust::make_permutation_iterator(diagonal.begin(), A.column_indices.begin()))) + A.num_entries,
                      copyflags.begin(),
                      pred);


    // compute number of entries in output
    IndexType num_entries = thrust::count(exec, copyflags.begin(), copyflags.end(), true);

    // resize output
    S.resize(A.num_rows, A.num_cols, num_entries);

    // copy strong connections to output
    // thrust::copy_if(exec,
    //                 thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), A.values.begin())),
    //                 thrust::make_zip_iterator(thrust::make_tuple(A.row_indices.begin(), A.column_indices.begin(), A.values.begin())) + A.num_entries,
    //                 copyflags.begin(),
    //                 thrust::make_zip_iterator(thrust::make_tuple(S.row_indices.begin(), S.column_indices.begin(), S.values.begin())),
    //                 thrust::identity<bool>());

    // WAR for runtime error "cudaFuncGetAttributes: invalid device function"
    // using zip_iterators
    thrust::copy_if(exec,
                    A.row_indices.begin(),
                    A.row_indices.begin() + A.num_entries,
                    copyflags.begin(),
                    S.row_indices.begin(),
                    thrust::identity<bool>());
    thrust::copy_if(exec,
                    A.column_indices.begin(),
                    A.column_indices.begin() + A.num_entries,
                    copyflags.begin(),
                    S.column_indices.begin(),
                    thrust::identity<bool>());
    thrust::copy_if(exec,
                    A.values.begin(),
                    A.values.begin() + A.num_entries,
                    copyflags.begin(),
                    S.values.begin(),
                    thrust::identity<bool>());
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void symmetric_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                      const MatrixType1& A,
                                            MatrixType2& S,
                                      const double theta,
                                      cusp::known_format)
{
    typedef typename MatrixType1::const_coo_view_type MatrixViewType;
    typedef typename cusp::detail::as_coo_type<MatrixType2>::type CooType;

    MatrixViewType A_(A);
    CooType S_;

    symmetric_strength_of_connection(exec, A_ , S_, theta, cusp::coo_format());

    cusp::convert(exec, S_, S);
}

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void symmetric_strength_of_connection(thrust::execution_policy<DerivedPolicy> &exec,
                                      const MatrixType1& A,
                                            MatrixType2& S,
                                      const double theta)
{
    typedef typename MatrixType1::format Format;

    Format format;

    symmetric_strength_of_connection(thrust::detail::derived_cast(exec), A, S, theta, format);
}

} // end namespace detail
} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

