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

/*! \file polynomial.inl
 *  \brief Inline file for polynomial.h
 */

#include <cusp/multiply.h>

#include <cusp/format_utils.h>
#include <cusp/eigen/spectral_radius.h>

#ifdef _WIN32
	#define _USE_MATH_DEFINES
#endif

#include <math.h>

namespace cusp
{
namespace relaxation
{
namespace detail
{

template <typename ValueType>
void chebyshev_polynomial_coefficients( const ValueType rho,
                                        cusp::array1d<ValueType,cusp::host_memory>& coefficients,
                                        const ValueType lower_bound = 1.0/30.0,
                                        const ValueType upper_bound = 1.1)
{
    const size_t degree = 3;

    ValueType x0 = lower_bound * rho;
    ValueType x1 = upper_bound * rho;

    // Chebyshev roots for the interval [-1,1]
    cusp::array1d<ValueType,cusp::host_memory> std_roots(degree);

    for( size_t i=0; i<degree; i++ )
        std_roots[i] = std::cos( M_PI * (ValueType(i) + 0.5)/ degree );

    // Chebyshev roots for the interval [x0,x1]
    for( size_t i=0; i<degree; i++ )
        std_roots[i] = 0.5 * (x1-x0) * (1 + std_roots[i]) + x0;

    // Compute monic polynomial coefficients of polynomial with scaled roots
    // TODO: Implement convolution method for polynomial multiplication
    coefficients.resize(degree+1);
    ValueType a = std_roots[0];
    ValueType b = std_roots[1];
    ValueType c = std_roots[2];
    coefficients[0] = 1.0;
    coefficients[1] = -(a+b+c);
    coefficients[2] = (a*b) + (b*c) + (c*a);
    coefficients[3] = -(a*b*c);

    // Scale coefficients to enforce C(0) = 1.0
    ValueType scale_factor = 1.0/coefficients.back();
    cusp::blas::scal(coefficients, scale_factor);
}

} // end detail namespace

template <typename ValueType, typename MemorySpace>
template<typename MatrixType>
polynomial<ValueType,MemorySpace>
::polynomial(const MatrixType& A)
  : residual(A.num_rows), h(A.num_rows), y(A.num_rows)
{
    ValueType rho = cusp::eigen::ritz_spectral_radius(A, 8, true);
    detail::chebyshev_polynomial_coefficients(rho, default_coefficients);
    default_coefficients.resize( default_coefficients.size() - 1 );

    for( size_t index = 0; index < default_coefficients.size(); index++ )
        default_coefficients[index] *= -1.0;
}

template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType>
polynomial<ValueType,MemorySpace>
::polynomial(const MatrixType& A, const VectorType& coefficients)
  : residual(A.num_rows), h(A.num_rows), y(A.num_rows)
{
    size_t default_size = coefficients.size()-1;
    default_coefficients.resize( default_size );
    for( size_t index = 0; index < default_size; index++ )
        default_coefficients[index] = -ValueType(coefficients[index]);
}

// linear_operator
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2>
void polynomial<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x)
{
    polynomial<ValueType,MemorySpace>::operator()(A,b,x,default_coefficients);
}

// override default coefficients
template <typename ValueType, typename MemorySpace>
template<typename MatrixType, typename VectorType1, typename VectorType2, typename VectorType3>
void polynomial<ValueType,MemorySpace>
::operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, const VectorType3& coefficients)
{
    if( cusp::blas::nrm2(x) == 0.0 )
    {
        residual = b;
    }
    else
    {
        // compute residual <- b - A*x
        cusp::multiply(A, x, residual);
        cusp::blas::axpby(b, residual, residual, ValueType(1), ValueType(-1));
    }

    ValueType scale_factor = coefficients[0];
    cusp::blas::axpby(residual, h, h, scale_factor, ValueType(0));

    for( size_t i = 1; i<coefficients.size(); i++ )
    {
        scale_factor = coefficients[i];

        cusp::multiply(A, h, y);
        cusp::blas::axpby(y, residual, h, ValueType(1.0), scale_factor);
    }

    cusp::blas::axpy(h, x, ValueType(1.0));
}

} // end namespace relaxation
} // end namespace cusp

