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

#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/complex.h>
#include <cusp/format_utils.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>
#include <cusp/eigen/arnoldi.h>
#include <cusp/precond/diagonal.h>

#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

#include <thrust/detail/integer_traits.h>

#include <algorithm>

namespace cusp
{
namespace eigen
{
namespace detail
{

template <typename MatrixType>
struct Dinv_A : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
{
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;

    const MatrixType& A;
    const cusp::precond::diagonal<ValueType,MemorySpace> Dinv;

    Dinv_A(const MatrixType& A)
        : cusp::linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_entries + A.num_rows),
          A(A), Dinv(A)
    {}

    template <typename Array1, typename Array2>
    void operator()(const Array1& x, Array2& y) const
    {
        cusp::multiply(A,x,y);
        cusp::multiply(Dinv,y,y);
    }
};

template <typename MatrixType>
double disks_spectral_radius(const MatrixType& A, coo_format)
{
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    const IndexType N = A.num_rows;

    // compute sum of absolute values for each row of A
    cusp::array1d<NormType, MemorySpace> row_sums(N, NormType(0));

#if THRUST_VERSION >= 100800
    const typename MatrixType::row_indices_array_type& A_row_indices(A.row_indices);
#else
    cusp::array1d<IndexType,MemorySpace> A_row_indices(A.row_indices);
#endif

    thrust::reduce_by_key
    (A_row_indices.begin(), A_row_indices.end(),
     thrust::make_transform_iterator(A.values.begin(), cusp::abs_functor<ValueType>()),
     thrust::make_discard_iterator(),
     row_sums.begin(),
     thrust::equal_to<IndexType>(),
     thrust::plus<NormType>());

    return *thrust::max_element(row_sums.begin(), row_sums.end());
}

template <typename MatrixType>
double disks_spectral_radius(const MatrixType& A, sparse_format)
{
    typedef typename MatrixType::const_coo_view_type CooView;

    CooView A_coo(A);

    return disks_spectral_radius(A_coo, coo_format());
}

template <typename Matrix, typename Array2d>
void lanczos_estimate(const Matrix& A, Array2d& H, size_t k)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    size_t N = A.num_cols;
    size_t maxiter = std::min(N, k);

    // allocate workspace
    cusp::array1d<ValueType,MemorySpace> v0(N);
    cusp::array1d<ValueType,MemorySpace> v1(N);
    cusp::array1d<ValueType,MemorySpace> w(N);

    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), v1);

    cusp::blas::scal(v1, ValueType(1) / cusp::blas::nrm2(v1));

    Array2d H_(maxiter + 1, maxiter, 0);

    ValueType alpha = 0.0;
    NormType beta = 0.0;

    size_t j;

    for(j = 0; j < maxiter; j++)
    {
        cusp::multiply(A, v1, w);

        if(j >= 1)
        {
            H_(j - 1, j) = beta;
            cusp::blas::axpy(v0, w, -beta);
        }

        alpha = cusp::blas::dotc(w, v1);
        H_(j,j) = alpha;

        cusp::blas::axpy(v1, w, -alpha);

        beta = cusp::blas::nrm2(w);
        H_(j + 1, j) = beta;

        if(beta < 1e-10) break;

        cusp::blas::scal(w, ValueType(1) / beta);

        // [v0 v1  w] - > [v1  w v0]
        v0.swap(v1);
        v1.swap(w);
    }

    H.resize(j,j);
    for(size_t row = 0; row < j; row++)
        for(size_t col = 0; col < j; col++)
            H(row,col) = H_(row,col);
}

} // end detail namespace

template <typename MatrixType>
double disks_spectral_radius(const MatrixType& A)
{
    return detail::disks_spectral_radius(A, typename MatrixType::format());
}

template <typename MatrixType>
double estimate_rho_Dinv_A(const MatrixType& A)
{
    detail::Dinv_A<MatrixType> Dinv_A(A);

    return cusp::eigen::ritz_spectral_radius(Dinv_A, 8);
}

template <typename MatrixType>
double estimate_spectral_radius(const MatrixType& A, size_t k)
{
    typedef typename MatrixType::index_type   IndexType;
    typedef typename MatrixType::value_type   ValueType;
    typedef typename MatrixType::memory_space MemorySpace;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    const IndexType N = A.num_rows;

    cusp::array1d<ValueType, MemorySpace> x(N);
    cusp::array1d<ValueType, MemorySpace> y(N);

    // initialize x to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), x);

    for(size_t i = 0; i < k; i++)
    {
        cusp::blas::scal(x, NormType(1.0) / cusp::blas::nrmmax(x));
        cusp::multiply(A, x, y);
        x.swap(y);
    }

    return k == 0 ? 0 : cusp::blas::nrm2(x) / cusp::blas::nrm2(y);
}

template <typename MatrixType>
double ritz_spectral_radius(const MatrixType& A, size_t k, bool symmetric)
{
    typedef typename MatrixType::value_type ValueType;

    cusp::array2d<ValueType,cusp::host_memory> H;

    if(symmetric)
        detail::lanczos_estimate(A, H, k);
    else
        cusp::eigen::arnoldi(A, H, k);

    return estimate_spectral_radius(H);
}

} // end namespace eigen
} // end namespace cusp

