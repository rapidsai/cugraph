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
#include <cusp/gallery/stencil.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES 1  // make sure M_PI is defined
#endif
#include <math.h>

namespace cusp
{
namespace gallery
{

template <typename Method,
          typename MatrixType>
void diffusion(MatrixType& matrix,
               const size_t m,
               const size_t n,
               const double eps,
               const double theta)
{
    typedef typename MatrixType::index_type       IndexType;
    typedef typename MatrixType::value_type       ValueType;
    typedef typename MatrixType::memory_space     MemorySpace;
    typedef thrust::tuple<IndexType,IndexType>    StencilIndex;
    typedef thrust::tuple<StencilIndex,ValueType> StencilPoint;

    ValueType C = cos(theta);
    ValueType S = sin(theta);
    ValueType CC = C*C;
    ValueType SS = S*S;
    ValueType CS = C*S;

    ValueType a;
    ValueType b;
    ValueType c;
    ValueType d;
    ValueType e;

    if( thrust::detail::is_same<Method, FE>::value )
    {
        a = (-1.0*eps - 1.0)*CC + (-1.0*eps - 1.0)*SS + ( 3.0*eps - 3.0)*CS;
        b = ( 2.0*eps - 4.0)*CC + (-4.0*eps + 2.0)*SS;
        c = (-1.0*eps - 1.0)*CC + (-1.0*eps - 1.0)*SS + (-3.0*eps + 3.0)*CS;
        d = (-4.0*eps + 2.0)*CC + ( 2.0*eps - 4.0)*SS;
        e = ( 8.0*eps + 8.0)*CC + ( 8.0*eps + 8.0)*SS;

        a /= 6.0;
        b /= 6.0;
        c /= 6.0;
        d /= 6.0;
        e /= 6.0;
    }
    else if( thrust::detail::is_same<Method, FD>::value )
    {
        a = 0.5 * (eps-1.0) * CS;
        b = -(eps*SS + CC);
        c = -a;
        d = -(eps*CC + SS);
        e = 2.0 * (eps+1.0);
    }
    else
    {
        throw cusp::invalid_input_exception("unrecognized discretization method");
    }

    cusp::array1d<StencilPoint, MemorySpace> stencil;

    stencil.push_back(StencilPoint(StencilIndex( -1, -1), a));
    stencil.push_back(StencilPoint(StencilIndex(  0, -1), b));
    stencil.push_back(StencilPoint(StencilIndex(  1, -1), c));
    stencil.push_back(StencilPoint(StencilIndex( -1,  0), d));
    stencil.push_back(StencilPoint(StencilIndex(  0,  0), e));
    stencil.push_back(StencilPoint(StencilIndex(  1,  0), d));
    stencil.push_back(StencilPoint(StencilIndex( -1,  1), c));
    stencil.push_back(StencilPoint(StencilIndex(  0,  1), b));
    stencil.push_back(StencilPoint(StencilIndex(  1,  1), a));

    cusp::gallery::generate_matrix_from_stencil(matrix, stencil, StencilIndex(m,n));
}

} // end namespace gallery
} // end namespace cusp

