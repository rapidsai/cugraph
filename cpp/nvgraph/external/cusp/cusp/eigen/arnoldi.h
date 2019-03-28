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

/*! \file arnoldi.h
 *  \brief Arnoldi method
 */

#pragma once

#include <cusp/detail/config.h>

#include <cstddef>

namespace cusp
{
namespace eigen
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup eigensolvers EigenSolvers
 *  \brief Iterative methods for computing eigenpairs of hermitian and
 *  non-hermitian linear systems
 *  \ingroup iterative_solvers
 *  \{
 */

/**
 * \brief Approximate spectral radius of A using Arnoldi
 *
 * \tparam Matrix type of a sparse or dense matrix
 * \tparam Array2d type of dense matrix of partials
 *
 * \param A matrix of the linear system
 * \param H dense matrix of ritz values
 * \param k maximum number of outer Arnoldi iterations
 *
 * \return spectral radius approximation
 *
 * \par Overview
 * Approximates the spectral radius A using a specified number of
 * Arnoldi iterations.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  arnoldi to compute the spectral radius of a 16x16
 *  Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/eigen/arnoldi.h>
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // create an empty sparse matrix structure (CSR format)
 *      cusp::csr_matrix<int, float, cusp::device_memory> A;
 *
 *      // initialize matrix
 *      cusp::gallery::poisson5pt(A, 4, 4);
 *
 *      // compute the largest eigenpair of A using 20 Arnoldi iterations
 *      float rho = cusp::eigen::arnoldi(A, 20);
 *      std::cout << "Spectral radius of A : " << rho << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename Matrix, typename Array2d>
void arnoldi(const Matrix& A, Array2d& H, size_t k = 10);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/arnoldi.inl>
