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


/*! \file cg_m.h
 *  \brief Multi-mass Conjugate Gradient (CG-M) method
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/format.h>

#include <cusp/detail/execution_policy.h>
#include <thrust/detail/type_traits.h>

namespace cusp
{
namespace krylov
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup krylov_methods Krylov Methods
 *  \ingroup iterative_solvers
 *  \{
 */

/* \cond */

template <typename DerivedPolicy,
          typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor);

template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3>
void cg_m(const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma);

/* \endcond */

/**
 * \brief Multi-mass Conjugate Gradient method
 *
 * \param A matrix of the linear system
 * \param x solutions of the system
 * \param b right-hand side of the linear system
 * \param sigma array of shifts
 * \param monitor monitors interation and determines stoppoing conditions
 *
 * \par Overview
 *
 * Solves the symmetric, positive-definited linear system (A+sigma) x = b
 * for some set of constant shifts \p sigma for the price of the smallest shift
 * iteratively, for sparse A, without additional matrix-vector multiplication.
 *
 * \see http://arxiv.org/abs/hep-lat/9612014
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p bicgstab to
 *  solve a 10x10 Poisson problem.
 *
 *  \code
 *  #include <cusp/csr_matrix.h>
 *  #include <cusp/monitor.h>
 *  #include <cusp/krylov/cg_m.h>
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
 *      // allocate storage for solution (x) and right hand side (b)
 *      size_t N_s = 4;
 *      cusp::array1d<float, cusp::device_memory> x(A.num_rows*N_s, 0);
 *      cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *      // set sigma values
 *      cusp::array1d<float, cusp::device_memory> sigma(N_s);
 *      sigma[0] = 0.1;
 *      sigma[1] = 0.5;
 *      sigma[2] = 1.0;
 *      sigma[3] = 5.0;
 *
 *      // set stopping criteria:
 *      //  iteration_limit    = 100
 *      //  relative_tolerance = 1e-6
 *      //  absolute_tolerance = 0
 *      //  verbose            = true
 *      cusp::monitor<float> monitor(b, 100, 1e-6, 0, true);
 *
 *      // solve the linear system A x = b
 *      cusp::krylov::cg_m(A, x, b, sigma, monitor);
 *
 *      return 0;
 *  }
 *  \endcode
 *
 *  \see \p monitor
 */
template <typename LinearOperator,
          typename VectorType1,
          typename VectorType2,
          typename VectorType3,
          typename Monitor>
void cg_m(const LinearOperator& A,
                VectorType1& x,
          const VectorType2& b,
          const VectorType3& sigma,
                Monitor& monitor);
/*! \}
 */

} // end namespace krylov
} // end namespace cusp

#include <cusp/krylov/detail/cg_m.inl>

