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

#include <cusp/array1d.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>

namespace cusp
{
namespace eigen
{

template <typename Matrix, typename Array2d>
void arnoldi(const Matrix& A, Array2d& H, size_t k)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef typename cusp::norm_type<ValueType>::type NormType;

    size_t N = A.num_rows;

    size_t maxiter = std::min(N, k);

    Array2d H_(maxiter + 1, maxiter, 0);

    // allocate workspace of k + 1 vectors
    std::vector< cusp::array1d<ValueType,MemorySpace> > V(maxiter + 1);
    for (size_t i = 0; i < maxiter + 1; i++)
        V[i].resize(N);

    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), V[0]);

    // normalize v0
    cusp::blas::scal(V[0], ValueType(1) / cusp::blas::nrm2(V[0]));

    NormType beta = 0.0;

    size_t j;

    for(j = 0; j < maxiter; j++)
    {
        cusp::multiply(A, V[j], V[j + 1]);

        for(size_t i = 0; i <= j; i++)
        {
            H_(i,j) = cusp::blas::dotc(V[i], V[j + 1]);

            cusp::blas::axpy(V[i], V[j + 1], -H_(i,j));
        }

        beta = cusp::blas::nrm2(V[j + 1]);
        H_(j + 1, j) = beta;

        if(beta < 1e-10) break;

        cusp::blas::scal(V[j + 1], ValueType(1) / H_(j+1,j));
    }

    H.resize(j,j);
    for( size_t row = 0; row < j; row++ )
        for( size_t col = 0; col < j; col++ )
            H(row,col) = H_(row,col);
}

} // end namespace eigen
} // end namespace cusp

