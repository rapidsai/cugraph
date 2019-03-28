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

/*! \file smoothed_aggregation.h
 *  \brief Algebraic multigrid preconditoner based on smoothed aggregation.
 *
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/format_utils.h>
#include <cusp/multiply.h>
#include <cusp/relaxation/polynomial.h>

#include <thrust/transform.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup preconditioners Preconditioners
 *  \ingroup preconditioners
 *  \{
 */

template <typename ValueType, typename MemorySpace>
class polynomial_smoother
{
private:

    typedef cusp::relaxation::polynomial<ValueType,MemorySpace> BaseSmoother;

public:
    size_t num_iters;
    cusp::relaxation::polynomial<ValueType,MemorySpace> M;

    polynomial_smoother(void) {}

    template <typename ValueType2, typename MemorySpace2>
    polynomial_smoother(const polynomial_smoother<ValueType2,MemorySpace2>& A) : num_iters(A.num_iters), M(A.M) {}

    template <typename MatrixType, typename Level>
    polynomial_smoother(const MatrixType& A, const Level& L)
    {
        initialize(A, L);
    }

    template <typename MatrixType, typename Level>
    void initialize(const MatrixType& A, const Level& L)
    {
        num_iters = L.num_iters;
        M = BaseSmoother(A);
    }

    // ignores initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void presmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        // Ignore the initial x and use b as the residual
        ValueType scale_factor = M.default_coefficients[0];
        cusp::blas::axpby(b, x, x, scale_factor, ValueType(0));

        for( size_t i = 1; i < M.default_coefficients.size(); i++ )
        {
            scale_factor = M.default_coefficients[i];

            cusp::multiply(A, x, M.y);
            cusp::blas::axpby(M.y, b, x, ValueType(1.0), scale_factor);
        }
    }

    // smooths initial x
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void postsmooth(const MatrixType& A, const VectorType1& b, VectorType2& x)
    {
        // compute residual <- b - A*x
        cusp::multiply(A, x, M.residual);
        cusp::blas::axpby(b, M.residual, M.residual, ValueType(1), ValueType(-1));

        ValueType scale_factor = M.default_coefficients[0];
        cusp::blas::axpby(M.residual, M.h, M.h, scale_factor, ValueType(0));

        for( size_t i = 1; i < M.default_coefficients.size(); i++ )
        {
            scale_factor = M.default_coefficients[i];

            cusp::multiply(A, M.h, M.y);
            cusp::blas::axpby(M.y, M.residual, M.h, ValueType(1.0), scale_factor);
        }

        cusp::blas::axpy(M.h, x, ValueType(1.0));
    }
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp

