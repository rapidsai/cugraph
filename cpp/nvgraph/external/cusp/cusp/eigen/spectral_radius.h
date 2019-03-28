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

/*! \file spectral_radius.h
 *  \brief Various methods to compute spectral radius
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
 *  \ingroup iterative_solvers
 *  \{
 */

/**
 * \brief Uses Gershgorin disks to approximate spectral radius
 *
 * \tparam MatrixType type of a sparse or dense matrix
 *
 * \param A matrix of the linear system
 *
 * \return spectral radius approximation
 *
 * \par Overview
 * Approximates the spectral radius of a matrix using Gershgorin disks.
 *
 * \see https://en.wikipedia.org/wiki/Gershgorin_circle_theorem 
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p disks_spectral_radius to
 *  compute the spectral radius of a 16x16 Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/eigen/spectral_radius.h>
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
 *      // compute the largest eigenpair of A
 *      float rho = cusp::eigen::disks_spectral_radius(A);
 *      std::cout << "Spectral radius of A : " << rho << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType>
double disks_spectral_radius(const MatrixType& A);

/**
 * \brief Approximate spectral radius of (D^-1)A
 *
 * \tparam MatrixType type of a sparse or dense matrix
 *
 * \param A matrix of the linear system
 *
 * \return spectral radius approximation
 *
 * \par Overview
 * Approximates the spectral radius (D^-1)A, where D is a diagonal matrix
 * containing the diagonal entries of A. The spectral radius of (D^-1)A is
 * computed using either Lanczos or Arnoldi.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p estimate_rho_Dinv_A to
 *  compute the spectral radius of a 16x16 Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/eigen/spectral_radius.h>
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
 *      // compute the largest eigenpair of A
 *      float rho = cusp::eigen::estimate_rho_Dinv_A(A);
 *      std::cout << "Spectral radius of (D^-1)A : " << rho << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType>
double estimate_rho_Dinv_A(const MatrixType& A);

/**
 * \brief Approximate spectral radius of A using Lanczos
 *
 * \tparam MatrixType type of a sparse or dense matrix
 *
 * \param A matrix of the linear system
 *
 * \return spectral radius approximation
 *
 * \par Overview
 * Approximates the spectral radius A using a specified number of Lanczos
 * iterations.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  estimate_spectral_radius to compute the spectral radius of a 16x16
 *  Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/eigen/spectral_radius.h>
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
 *      // compute the largest eigenpair of A using 20 Lanczos iterations
 *      float rho = cusp::eigen::estimate_spectral_radius(A, 20);
 *      std::cout << "Spectral radius of A : " << rho << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType>
double estimate_spectral_radius(const MatrixType& A, size_t k = 20);

/**
 * \brief Approximate spectral radius of A using Lanczos or Arnoldi
 *
 * \tparam MatrixType type of a sparse or dense matrix
 *
 * \param A matrix of the linear system
 *
 * \return spectral radius approximation
 *
 * \par Overview
 * Approximates the spectral radius A using a specified number of Lanczos
 * or Arnoldi iterations.
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p
 *  ritz_spectral_radius to compute the spectral radius of a 16x16
 *  Laplacian matrix.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/eigen/spectral_radius.h>
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
 *      float rho = cusp::eigen::ritz_spectral_radius(A, 20, false);
 *      std::cout << "Spectral radius of A : " << rho << std::endl;
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType>
double ritz_spectral_radius(const MatrixType& A, size_t k = 10, bool symmetric=false);

/*! \}
 */

} // end namespace eigen
} // end namespace cusp

#include <cusp/eigen/detail/spectral_radius.inl>
