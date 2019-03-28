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

/*! \file lanczos.h
 *  \brief Lanczos method
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/eigen/lanczos_options.h>

namespace cusp
{
namespace eigen
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup eigensolvers EigenSolvers
 *  \ingroup iterative_solvers
 *  \{
 */

/* \cond */
template <typename MatrixType,
          typename Array1d>
void lanczos(const MatrixType& A,
                   Array1d& eigVals);

template <typename MatrixType,
          typename Array1d,
          typename Array2d>
void lanczos(const MatrixType& A,
                   Array1d& eigVals,
                   Array2d& eigVecs);
/* \endcond */

/**
 * \brief Lanczos method
 *
 * \tparam LinearOperator is a matrix or subclass of \p linear_operator
 * \tparam Array1d array of eigenvalues
 * \tparam Array2d matrix of eigenvectors
 *
 * \param A matrix of the linear system
 * \param eigvals eigenvalues
 * \param eigvecs eigenvectors
 *
 * \par Overview
 * Computes the extreme eigenpairs of hermitian linear systems A x = s x.
 *
 * \note \p A must be symmetric.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p lanczos to
 *  compute the eigenpairs of a 10x10 Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/eigen/lanczos.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 10, 10);
 *
 *      // allocate storage and initialize eigenpairs
 *      cusp::array1d<float, cusp::device_memory> eigvals(5,0);
 *      cusp::array2d<float, cusp::device_memory, cusp::column_major> eigvecs;
 *
 *      // initialize Lanzcos option
 *      cusp::eigen::lanczos_options<float> options;
 *      options.tol             = 1e-6;
 *      options.maxIter         = 100;
 *      options.verbose         = true;
 *      options.computeEigVecs  = false;
 *      options.reorth          = cusp::eigen::Full;
 *
 *      // compute the largest eigenpair of A
 *      cusp::eigen::lanczos(A, eigvals, eigvecs, options);
 *
 *      // print largest eigenvalue
 *      std::cout << "Largest eigenvalue : " << eigvals.back() << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename LinearOperator,
          typename Array1d,
          typename Array2d,
          typename LanczosOptions>
void lanczos(const LinearOperator& A,
             Array1d& eigVals,
             Array2d& eigVecs,
             LanczosOptions& options);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/lanczos.inl>
