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
#include <cusp/array2d.h>
#include <cusp/multiply.h>

#include <cusp/blas/blas.h>
#include <cusp/lapack/lapack.h>

// TODO : Abstract using generic QR interface
#include <cusp/eigen/detail/gram_schmidt.inl>

#include <algorithm>

#include <iostream>
#include <iomanip>

namespace cusp
{
namespace eigen
{

template <typename Matrix, typename Array1d, typename Array2d, typename LanczosOptions>
void lanczos(const Matrix& A, Array1d& eigVals, Array2d& eigVecs, LanczosOptions& options)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;
    typedef typename Array2d::view Array2dView;
    typedef typename Array2d::column_view ColumnView;

    size_t N = A.num_cols;
    size_t iter = 0;
    size_t neigWanted = eigVals.size();
    size_t neigLow = 0, neigLowFound = 0, neigLowConverged = 0;
    size_t neigHigh = 0, neigHighFound = 0, neigHighConverged = 0;
    size_t reorthIterCount = 0, reorthVectorCount = 0;
    size_t doubleReorthIterCount = 0, doubleReorthVectorCount = 0;

    const ValueType eps = std::numeric_limits<ValueType>::epsilon();
    const ValueType eps1 = std::sqrt(ValueType(N))*eps/2.0;
    const ValueType delta = std::sqrt(eps);
    const ValueType eta = std::min(eps1, ValueType(std::pow(eps, 3.0/4.0)));

    ValueType aa = 0.0, bb = 0.0, bb_old = 0.0;
    ValueType betaSum = 0.0;

    if(options.reorth)
        neigWanted = std::min(neigWanted, N);

    if(options.minIter == 0)
    {
        if(options.eigPart == cusp::eigen::BE)
            options.minIter = options.defaultMinIterFactor * ((neigWanted + 1)/2);
        else
            options.minIter = options.defaultMinIterFactor * neigWanted;

        if(options.maxIter)
            options.minIter = std::min(options.maxIter, options.minIter);
    }
    if(options.maxIter == 0)
    {
        if(options.eigPart == cusp::eigen::BE)
            options.maxIter = 500 + options.defaultMaxIterFactor*((neigWanted + 1)/2);
        else
            options.maxIter = 500 + options.defaultMaxIterFactor*neigWanted;
    }

    if(options.reorth)
    {
        options.minIter = std::min(options.minIter, N);
        options.maxIter = std::min(options.maxIter, N);
    }
    options.maxIter = std::max(options.maxIter, options.minIter);

    if(options.stride == 0)
        options.stride = 10;

    if(options.memoryExpansionFactor <= 1.0)
        options.memoryExpansionFactor = 1.2;
    if(options.tol < 0.0)
        options.tol = std::sqrt(eps);

    if(options.verbose)
        options.print();

    if(options.eigPart == cusp::eigen::LA)
    {
        neigHigh = neigWanted;
        neigLow = 0;
    }
    else if(options.eigPart == cusp::eigen::SA)
    {
        neigLow = neigWanted;
        neigHigh = 0;
    }
    else if(options.eigPart == cusp::eigen::BE)
    {
        neigLow = neigWanted / 2;
        neigHigh = neigWanted - neigLow;
    }
    else
    {
        throw cusp::runtime_exception("Invalid spectrum part specified!");
    }

    // allocate host workspace
    cusp::array1d<double,cusp::host_memory> ritzVal0(options.maxIter);
    cusp::array1d<double,cusp::host_memory> ritzVal(options.maxIter);

    cusp::array1d<double,cusp::host_memory> alphas(options.minIter);
    cusp::array1d<double,cusp::host_memory> betas(options.minIter);

    cusp::array1d<bool,cusp::host_memory> flag;

    // allocate device workspace
    cusp::array1d<ValueType,MemorySpace> v0(N, 0);
    cusp::array1d<ValueType,MemorySpace> v1(N);
    cusp::array1d<ValueType,MemorySpace> v2(N);

    // initialize starting vector to random values in [0,1)
    cusp::copy(cusp::random_array<ValueType>(N), v1);
    cusp::blas::scal(v1, 1.0/cusp::blas::nrm2(v1));

    cusp::array2d<ValueType,MemorySpace,cusp::column_major> V;

    if(!options.reorth==None)
        V.resize(N, options.minIter);

    bool allEigenvaluesCheckedConverged = false;

    size_t memorySize = options.minIter;
    size_t minIter = options.minIter;

    while(1)
    {
        if(iter == memorySize)
        {
            if(options.verbose)
                std::cout << "At iteration #" << iter+1 <<
                          ", memory expanded for storing Lanczos vectors" << std::endl;
            memorySize *= options.memoryExpansionFactor;
            if(memorySize <= iter)
                memorySize = iter + 1;

            alphas.resize(memorySize);
            betas.resize(memorySize);

            if(options.reorth==Full)
                V.resize(N, memorySize);
        }

        if(options.reorth==Full)
            cusp::blas::copy(v1, V.column(iter));

        cusp::multiply(A, v1, v2);

        cusp::blas::axpy(v0, v2, -bb);

        aa = cusp::blas::dot(v1, v2);
        alphas[iter] = aa;

        if(iter + 1 == options.maxIter)
        {
            if(options.computeEigVecs)
            {
                cusp::blas::axpy(v1, v2, -aa);
                betas[iter] = cusp::blas::nrm2(v2);
            }

            iter++;
            break;
        }

        if((options.reorth == cusp::eigen::Full || iter+1 < N) && (iter+1 >= minIter) && (iter+1-minIter)%options.stride == 0)
        {
            cusp::array2d<double, cusp::host_memory, cusp::column_major> tempV;

            if(options.stride > 1 || iter+1 == minIter)
                cusp::lapack::stev(alphas.subarray(0,iter-1), betas.subarray(0,iter-1), ritzVal0, tempV, 'N');

            cusp::lapack::stev(alphas.subarray(0,iter), betas.subarray(0,iter), ritzVal, tempV, 'N');

            int i = 0;
            int nev = neigLow;
            ValueType eigLowSum = 0.0;
            ValueType eigLowSum0 = 0.0;
            ValueType absEigLowSum0 = 0.0;

            while(nev && (ritzVal[i] <= options.eigLowCut))
            {
                absEigLowSum0 += std::abs(ritzVal0[i]);
                eigLowSum0 += ritzVal0[i];
                eigLowSum += ritzVal[i];
                nev--;
                i++;
            }

            neigLowFound = neigLow - nev;
            ValueType eigLowErr = std::abs((eigLowSum-eigLowSum0) / absEigLowSum0);

            if(options.verbose && neigLow)
            {
                std::cout << "At iteration #" << iter+1
                          << " estimated eigen error (low end) is "
                          << eigLowErr << " (" << neigLow - nev
                          << " eigenvalues)" << std::endl;
            }

            if(absEigLowSum0 == 0.0 && eigLowSum == 0.0)
                eigLowErr = 0.0;

            i = iter;
            int nev2 = neigHigh;
            ValueType eigHighSum = 0.0;
            ValueType eigHighSum0 = 0.0;
            ValueType absEigHighSum0 = 0.0;

            while(nev2 && (ritzVal[i] >= options.eigHighCut))
            {
                absEigHighSum0 += std::abs(ritzVal0[i-1]);
                eigHighSum0 += ritzVal0[i-1];
                eigHighSum += ritzVal[i];
                nev2--;
                i--;
            }

            neigHighFound = neigHigh - nev2;
            ValueType eigHighErr = std::abs((eigHighSum-eigHighSum0) / absEigHighSum0);

            if(options.verbose && neigHigh)
            {
                std::cout << "At iteration #" << iter+1
                          << " estimated eigen error (high end) is "
                          << eigHighErr << " (" << neigHigh - nev2
                          << " eigenvalues)" << std::endl;
            }

            if(absEigHighSum0 == 0.0 && eigHighSum == 0.0)
                eigHighErr = 0.0;

            if(eigLowErr <= options.tol && eigHighErr <= options.tol)
            {
                if(options.extraIter == 0)
                {
                    neigLowConverged = neigLowFound;
                    neigHighConverged = neigHighFound;
                }
                if((neigLowConverged < neigLowFound) || (neigHighConverged < neigHighFound))
                {
                    neigLowConverged = neigLowFound;
                    neigHighConverged = neigHighFound;
                    minIter = std::min(iter + options.extraIter + 1, options.maxIter);

                    if(options.verbose)
                        std::cout << "At iteration #" << iter+1 <<
                                  ", eigenvalues are deemed converged, running extra iterations." << std::endl;
                }
                else
                {
                    allEigenvaluesCheckedConverged = true;
                    cusp::blas::axpy(v1, v2, -aa);
                    bb = cusp::blas::nrm2(v2);
                    betas[iter] = bb;
                    iter++;
                    break;
                }
            }

            if(options.stride == 1)
                ritzVal0 = ritzVal;
        }

        cusp::blas::axpy(v1, v2, -aa);
        bb = cusp::blas::nrm2(v2);
        betas[iter] = bb;
        betaSum += bb;

        if(options.reorth == cusp::eigen::Full)
        {
            reorthIterCount++;
            reorthVectorCount += iter+1;

            detail::modifiedGramSchmidt(V, v2, flag, iter+1);

            bb_old = bb;
            bb = cusp::blas::nrm2(v2);

            if(options.doubleReorthGamma >= 1.0 || bb < options.doubleReorthGamma*bb_old)
            {
                if(options.verbose)
                    std::cout << "At iteration #" << iter+1 <<
                              ", reorthogonalization is doubled" << std::endl;

                doubleReorthIterCount++;
                doubleReorthVectorCount += iter+1;

                detail::modifiedGramSchmidt(V, v2, flag, iter+1);
                bb = cusp::blas::nrm2(v2);
            }

            betas[iter] = bb;
        }

        if(bb < betaSum*eps || bb == 0.0)
        {
            if(options.verbose)
                std::cout << "bb deemed zero!" << std::endl;

            cusp::copy(cusp::random_array<ValueType>(N), v2);

            detail::modifiedGramSchmidt(V, v2, flag, iter+1);
            bb = cusp::blas::nrm2(v2);
            betas[iter] = 0.0;
        }

        cusp::blas::scal(v2, 1.0/bb);

        // [v0 v1  w] - > [v1  w v0]
        v0.swap(v1);
        v1.swap(v2);

        iter++;
    } // end of Lanczos loop

    cusp::array2d<ValueType,MemorySpace,cusp::column_major> S;
    cusp::array2d<double,cusp::host_memory,cusp::column_major> S_h;

    if(options.computeEigVecs)
    {
        S_h.resize(iter,iter,ValueType(0));
        cusp::lapack::stev(alphas.subarray(0,iter-1), betas.subarray(0,iter-1), ritzVal, S_h);
        cusp::copy(S_h, S);
    }
    else
    {
        cusp::lapack::stev(alphas.subarray(0,iter-1), betas.subarray(0,iter-1), ritzVal, S_h, 'N');
    }

    if(allEigenvaluesCheckedConverged == false)
    {
        if((options.reorth == cusp::eigen::Full || options.maxIter < N) && (options.minIter != options.maxIter))
            std::cout << "Maximum number of Lanczos iterations " << iter
                      << " is met, but the desired eigenvalues may not be converged!" << std::endl;

        int i = 0;
        int nev = neigLow;
        while(nev && (ritzVal[i++] <= options.eigLowCut))
            nev--;
        neigLowFound = neigLow - nev;

        i = iter - 1;
        nev = neigHigh;
        while(nev && (ritzVal[i--] >= options.eigHighCut))
            nev--;
        neigHighFound = neigHigh - nev;
    }

    int a = 0;
    int b = neigLowFound - 1;
    int c = iter - neigHighFound - 1;
    int d = iter - 1;

    cusp::array1d<ValueType,cusp::host_memory> bottomElements;

    if(neigHighFound == 0)
    {
        if(neigLowFound == 0)
        {
            eigVals.resize(0);
            eigVecs.resize(0,0);
        }
        else
        {
            if(neigLowFound < eigVals.size())
            {
                eigVals.resize(neigLowFound);
                eigVecs.resize(N,neigLowFound);
            }

            cusp::blas::copy(ritzVal.subarray(a,b), eigVals);

            if(options.computeEigVecs)
            {
                Array2dView S_view(S.num_rows, neigLowFound, S.pitch, S.column(a));
                cusp::blas::gemm(V,S_view,eigVecs);

                bottomElements.resize(neigLowFound);
                cusp::blas::copy(S_view.row(iter-1), bottomElements);
            }
        }
    }
    else if(neigLowFound == 0)
    {
        if(neigHighFound < eigVals.size())
        {
            eigVals.resize(neigHighFound);
            eigVecs.resize(N, neigHighFound);
        }

        cusp::blas::copy(ritzVal.subarray(c,d-c), eigVals);

        if(options.computeEigVecs)
        {
            Array2dView S_view(S.num_rows, neigHighFound, S.pitch, S.column(c));
            cusp::blas::gemm(V,S_view,eigVecs);

            bottomElements.resize(neigHighFound);
            cusp::blas::copy(S_view.row(iter-1), bottomElements);
        }
    }
    else
    {
        cusp::blas::copy(ritzVal.subarray(a,b), eigVals.subarray(0,neigLowFound-1));
        cusp::blas::copy(ritzVal.subarray(c,d), eigVals.subarray(neigLowFound,neigLowFound+neigHighFound-1));

        if(options.computeEigVecs)
        {
            bottomElements.resize(neigLowFound + neigHighFound);

            Array2dView S_viewLow(S.num_rows, neigLowFound, S.pitch, S.column(a));
            Array2dView E_viewLow(S.num_rows, neigLowFound, S.pitch, eigVecs.column(0));
            cusp::blas::gemm(V, S_viewLow, E_viewLow);
            cusp::blas::copy(S_viewLow.row(iter-1), bottomElements.subarray(0,neigLowFound-1));

            Array2dView S_viewHigh(S.num_rows, neigHighFound, S.pitch, S.column(c));
            Array2dView E_viewHigh(S.num_rows, neigHighFound, S.pitch, eigVecs.column(neigLowFound));
            cusp::blas::gemm(V, S_viewHigh, E_viewHigh);
            cusp::blas::copy(S_viewHigh.row(iter-1), bottomElements.subarray(neigLowFound,neigLowFound+neigHighFound-1));
        }
    }

    double maxErr0 = 0.0, maxErr = 0.0;

    if(options.computeEigVecs)
    {
        double bottomBeta = betas[iter-1];
        for(size_t i = 0; i < neigLowFound + neigHighFound; i++)
        {
            double err0 = std::abs(bottomBeta*bottomElements[i]);
            maxErr0 = std::max(maxErr0, err0);
            maxErr = std::max(maxErr, std::abs(err0/ritzVal[i]));
        }
    }

    if(options.verbose)
    {
        std::cout << std::endl;
        std::cout << "Max absolute eigenvalue error   : " << maxErr0 << std::endl;
        std::cout << "Max relative eigenvalue error   : " << maxErr  << std::endl;

        std::cout << "Eigenvalues : " << std::endl;
        std::cout << std::scientific << std::setprecision(7);

        cusp::array2d<ValueType,MemorySpace,cusp::column_major> E(A.num_rows, eigVecs.num_cols, 0);

        std::cout << "eig id     eigenvalues      residual norm" << std::endl;
        if(options.computeEigVecs)
        {
            for(size_t i = 0; i < eigVals.size(); i++)
            {
                ColumnView eigV = eigVecs.column(i);
                ColumnView EV   = E.column(i);
                cusp::multiply(A, eigV, EV);
                cusp::blas::axpy(eigV, EV, -eigVals[i]);
            }

            for(size_t i = 0; i < eigVals.size(); i++)
                std::cout << std::setw(6) << i << std::setw(17) << eigVals[i] << std::setw(17)
                          << cusp::blas::nrm2(E.column(i)) << std::endl;

            std::cout << "||A*V-V*S||_F=" << cusp::blas::nrm2(E.values) << std::endl;
            std::cout.unsetf(std::ios::floatfield);
        }
        else
        {
            for(size_t i = 0; i < eigVals.size(); i++)
                std::cout << std::setw(6) << i << std::setw(17) << eigVals[i] << std::endl;
        }

        ValueType reorthVectorRate = 0.0;
        if(iter > 1)
            reorthVectorRate = 2.0 * (ValueType(reorthVectorCount)/ValueType(iter)/ValueType(iter-1));

        std::cout << std::endl;
        std::cout << "Iteration Count                     : " << iter << std::endl;
        std::cout << "Reorthogonalization Iteration Count : " << reorthIterCount << std::endl;
        std::cout << "Reorthogonalization Vector Count    : " << reorthVectorCount << " (" << 100.0*reorthVectorRate << "%%)" << std::endl;
        std::cout << std::endl;
    }
}

template <typename Matrix, typename Array1d, typename Array2d>
void lanczos(const Matrix& A, Array1d& eigVals, Array2d& eigVecs)
{
    typedef typename Matrix::value_type   ValueType;

    cusp::eigen::lanczos_options<ValueType> options;
    options.computeEigVecs = eigVecs.num_cols > 0;

    cusp::eigen::lanczos(A, eigVals, eigVecs, options);
}

template <typename Matrix, typename Array1d>
void lanczos(const Matrix& A, Array1d& eigVals)
{
    typedef typename Matrix::value_type   ValueType;
    typedef typename Matrix::memory_space MemorySpace;

    cusp::array2d<ValueType,MemorySpace,cusp::column_major> eigVecs;

    cusp::eigen::lanczos(A, eigVals, eigVecs);
}

} // end namespace eigen
} // end namespace cusp

