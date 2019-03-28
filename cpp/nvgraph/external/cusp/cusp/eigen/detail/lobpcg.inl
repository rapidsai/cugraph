/*
 *  Copyright 2008-2009 NVIDIA Corporation
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
#include <cusp/array2d.h>
#include <cusp/monitor.h>
#include <cusp/linear_operator.h>
#include <cusp/multiply.h>

#include <cusp/lapack/lapack.h>

namespace cusp
{
namespace eigen
{

template <typename LinearOperator,
          typename Array1d,
          typename Array2d,
          typename Monitor,
          typename Preconditioner>
void lobpcg(LinearOperator& A,
            Array1d& S,
            Array2d& X,
            Monitor& monitor,
            Preconditioner& M,
            bool largest)
{
    typedef typename LinearOperator::value_type   ValueType;

    typedef typename cusp::array1d<double,cusp::host_memory> VectorHost;
    typedef typename cusp::array2d<double,cusp::host_memory,cusp::column_major> Array2dHost;

    const size_t N = A.num_rows;

    // Normalize
    Array2d& blockVectorX(X);
    ValueType norm = cusp::blas::nrm2(blockVectorX);
    cusp::blas::scal(blockVectorX, ValueType(1.0/norm));

    Array1d blockVectorAX(N);

    Array1d blockVectorR(N);
    Array1d blockVectorP(N, ValueType(0));
    Array1d blockVectorAP(N, ValueType(0));

    Array1d activeBlockVectorR(N);
    Array1d activeBlockVectorAR(N);

    cusp::multiply(A, blockVectorX, blockVectorAX);

    ValueType _lambda = cusp::blas::dot(blockVectorX, blockVectorAX);

    while (monitor.iteration_count() < std::min(N,monitor.iteration_limit()))
    {
        cusp::blas::axpby(blockVectorX, blockVectorAX, blockVectorR, -_lambda, ValueType(1));

        monitor.residuals.push_back(cusp::blas::nrm2(blockVectorR));

        if(monitor.is_verbose())
        {
            std::cout << "Iteration      : " << monitor.iteration_count() << std::endl;
            std::cout << "Eigenvalue     : " << _lambda << std::endl;
            std::cout << "Residual norms : " << monitor.residuals.back() << std::endl << std::endl;
        }

        if( monitor.residuals.back() < monitor.relative_tolerance() ) break; // All eigenpairs converged

        // Apply preconditioner, M, to the active residuals
        cusp::multiply(M, blockVectorR, activeBlockVectorR);

        // Normalize
        ValueType norm = cusp::blas::nrm2(activeBlockVectorR);
        cusp::blas::scal(activeBlockVectorR, ValueType(1.0/norm));

        cusp::multiply(A, activeBlockVectorR, activeBlockVectorAR);

        if( monitor.iteration_count() > 0 )
        {
            ValueType norm = cusp::blas::nrm2(blockVectorP);
            cusp::blas::scal(blockVectorP, 1.0/norm);
            cusp::blas::scal(blockVectorAP, 1.0/norm);
        }

        // Perform the Rayleigh-Ritz procedure :
        // Compute symmetric Gram matrices
        size_t gram_size = monitor.iteration_count() > 0 ? 3 : 2;

        Array2dHost gramA(gram_size, gram_size, ValueType(0));
        Array2dHost gramB(gram_size, gram_size, ValueType(0));

        ValueType xaw = cusp::blas::dot( blockVectorX,        activeBlockVectorAR );
        ValueType waw = cusp::blas::dot( activeBlockVectorR, 	activeBlockVectorAR );
        ValueType xbw = cusp::blas::dot( blockVectorX, 	 	    activeBlockVectorR );

        if( monitor.iteration_count() > 0 )
        {
            ValueType xap = cusp::blas::dot( blockVectorX,       blockVectorAP );
            ValueType wap = cusp::blas::dot( activeBlockVectorR, blockVectorAP );
            ValueType pap = cusp::blas::dot( blockVectorP, 	     blockVectorAP );
            ValueType xbp = cusp::blas::dot( blockVectorX,       blockVectorP );
            ValueType wbp = cusp::blas::dot( activeBlockVectorR, blockVectorP );

            gramA(0,0) = _lambda;
            gramA(0,1) = xaw;
            gramA(0,2) = xap;
            gramA(1,0) = xaw;
            gramA(1,1) = waw;
            gramA(1,2) = wap;
            gramA(2,0) = xap;
            gramA(2,1) = wap;
            gramA(2,2) = pap;

            gramB(0,0) = 1.0;
            gramB(0,1) = xbw;
            gramB(0,2) = xbp;
            gramB(1,0) = xbw;
            gramB(1,1) = 1.0;
            gramB(1,2) = wbp;
            gramB(2,0) = xbp;
            gramB(2,1) = wbp;
            gramB(2,2) = 1.0;
        }
        else
        {
            gramA(0,0) = _lambda;
            gramA(0,1) = xaw;
            gramA(1,0) = xaw;
            gramA(1,1) = waw;

            gramB(0,0) = 1.0;
            gramB(0,1) = xbw;
            gramB(1,0) = xbw;
            gramB(1,1) = 1.0;
        }

        // Solve the generalized eigenvalue problem.
        VectorHost _lambda_h(gramA.num_rows, ValueType(0));
        Array2dHost eigBlockVector_h(gramA.num_rows, gramA.num_cols, ValueType(0));

        cusp::lapack::sygv(gramA, gramB, _lambda_h, eigBlockVector_h);

        int start_index = largest ? gramA.num_rows-1 : 0;

        _lambda = _lambda_h[start_index];
        ValueType eigBlockVectorX = eigBlockVector_h(0,start_index);
        ValueType eigBlockVectorR = eigBlockVector_h(1,start_index);

        // Compute Ritz vectors
        if( monitor.iteration_count() > 0 )
        {
            ValueType eigBlockVectorP = eigBlockVector_h(2,start_index);

            cusp::blas::axpby( activeBlockVectorR,  blockVectorP,  blockVectorP,  eigBlockVectorR, eigBlockVectorP );
            cusp::blas::axpby( activeBlockVectorAR, blockVectorAP, blockVectorAP, eigBlockVectorR, eigBlockVectorP );
        }
        else
        {
            cusp::blas::axpy( activeBlockVectorR,  blockVectorP,  eigBlockVectorR );
            cusp::blas::axpy( activeBlockVectorAR, blockVectorAP, eigBlockVectorR );
        }

        cusp::blas::axpby( blockVectorX,  blockVectorP,  blockVectorX,  eigBlockVectorX, ValueType(1) );
        cusp::blas::axpby( blockVectorAX, blockVectorAP, blockVectorAX, eigBlockVectorX, ValueType(1) );

        ++monitor;
    }

    S[0] = _lambda;
}

template <typename LinearOperator,
          typename Array1d,
          typename Array2d,
          typename Monitor>
void lobpcg(LinearOperator& A,
            Array1d& S,
            Array2d& X,
            Monitor& monitor,
            bool largest)
{
    typedef typename LinearOperator::value_type   ValueType;
    typedef typename LinearOperator::memory_space MemorySpace;

    cusp::identity_operator<ValueType,MemorySpace> M(A.num_rows, A.num_cols);

    cusp::eigen::lobpcg(A, S, X, monitor, M, largest);
}

template <typename LinearOperator,
          typename Array1d,
          typename Array2d>
void lobpcg(LinearOperator& A,
            Array1d& S,
            Array2d& X,
            bool largest)
{
    typedef typename LinearOperator::value_type   ValueType;

    cusp::constant_array<ValueType> b(A.num_rows, ValueType(1));
    cusp::monitor<ValueType> monitor(b);

    cusp::eigen::lobpcg(A, S, X, monitor, largest);
}

} // end namespace eigen
} // end namespace cusp

