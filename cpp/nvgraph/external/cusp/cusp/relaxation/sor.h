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

/*! \file sor.h
 *  \brief Successive Over-Relaxation relaxation.
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/array1d.h>
#include <cusp/linear_operator.h>

#include <cusp/relaxation/gauss_seidel.h>

namespace cusp
{
namespace relaxation
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup relaxation Relaxation Methods
 *  \brief Several relaxation methods
 *  \ingroup iterative_solvers
 *  \{
 */

/**
 * \brief Represents a Successive Over-Relaxation relaxation scheme
 *
 * \tparam ValueType value_type of the array
 * \tparam MemorySpace memory space of the array (\c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 * Computes vertex coloring and performs indexed Successive Over-Relaxation relaxation
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/csr_matrix.h>
 * #include <cusp/monitor.h>
 *
 * #include <cusp/blas/blas.h>
 * #include <cusp/linear_operator.h>
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp sor header file
 * #include <cusp/relaxation/sor.h>
 *
 * int main()
 * {
 *    // Construct 5-pt Poisson example
 *    cusp::csr_matrix<int, float, cusp::device_memory> A;
 *    cusp::gallery::poisson5pt(A, 5, 5);
 *
 *    // Initialize data
 *    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *    // Allocate temporaries
 *    cusp::array1d<float, cusp::device_memory> r(A.num_rows);
 *
 *    // Construct sor relaxation class
 *    cusp::relaxation::sor<float, cusp::device_memory> M(A);
 *
 *    // Compute initial residual
 *    cusp::multiply(A, x, r);
 *    cusp::blas::axpy(b, r, float(-1));
 *
 *    // Construct monitor with stopping criteria of 100 iterations or 1e-4 residual error
 *    cusp::monitor<float> monitor(b, 100, 1e-4, 0, true);
 *
 *    // Iteratively solve system
 *    while (!monitor.finished(r))
 *    {
 *        M(A, b, x);
 *        cusp::multiply(A, x, r);
 *        cusp::blas::axpy(b, r, float(-1));
 *        ++monitor;
 *    }
 *  }
 * \endcode
 */
template <typename ValueType, typename MemorySpace>
class sor : public cusp::linear_operator<ValueType, MemorySpace>
{
public:

    /* \cond */
    ValueType default_omega;
    cusp::array1d<ValueType,MemorySpace> temp;
    gauss_seidel<ValueType,MemorySpace> gs;
    /* \endcond */

    /*! This constructor creates an empty \p sor smoother.
     */
    sor(void) {}

    /*! This constructor creates a \p sor smoother using a given
     *  matrix and sweeping strategy (FORWARD, BACKWARD, SYMMETRIC).
     *
     *  \tparam MatrixType Type of input matrix used to create this \p
     *  sor smoother.
     *
     *  \param A Input matrix used to create smoother.
     *  \param omega Damping factor used in SOR smoother.
     *  \param default_direction Sweep strategy used to perform Gauss-Seidel
     *  smoothing.
     */
    template <typename MatrixType>
    sor(const MatrixType& A, const ValueType omega, sweep default_direction=SYMMETRIC)
      : default_omega(omega), temp(A.num_cols), gs(A, default_direction) {}

    /*! Copy constructor for \p sor smoother.
     *
     *  \tparam MemorySpace2 Memory space of input \p sor smoother.
     *
     *  \param A Input \p sor smoother.
     */
    template<typename MemorySpace2>
    sor(const sor<ValueType,MemorySpace2>& A)
        : default_omega(A.default_omega), temp(A.temp), gs(A.gs) {}

    /*! Perform SOR relaxation using default omega damping factor specified during
     * construction of this \p sor smoother
     *
     * \tparam MatrixType  Type of input matrix.
     * \tparam VectorType1 Type of input right-hand side vector.
     * \tparam VectorType2 Type of input approximate solution vector.
     *
     * \param A matrix of the linear system
     * \param x approximate solution of the linear system
     * \param b right-hand side of the linear system
     */
    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x);

    /*! Perform SOR relaxation using specified omega damping factor.
     *
     * \tparam MatrixType  Type of input matrix.
     * \tparam VectorType1 Type of input right-hand side vector.
     * \tparam VectorType2 Type of input approximate solution vector.
     *
     * \param A matrix of the linear system
     * \param x approximate solution of the linear system
     * \param b right-hand side of the linear system
     * \param omega Damping factor used in SOR smoother.
     * \param direction sweeping strategy for this \p gauss_seidel smoother
     * (FORWARD, BACKWARD, SYMMETRIC).
     */
    template <typename MatrixType, typename VectorType1, typename VectorType2>
    void operator()(const MatrixType& A, const VectorType1& b, VectorType2& x, const ValueType omega, sweep direction);
};
/*! \}
 */

} // end namespace relaxation
} // end namespace cusp

#include <cusp/relaxation/detail/sor.inl>

