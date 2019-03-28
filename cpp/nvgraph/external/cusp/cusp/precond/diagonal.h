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

/*! \file diagonal.h
 *  \brief Diagonal preconditioner.
 *
 *  Contributed by Andrew Trachenko and Nikita Styopin
 *  at SALD Laboratory ( http://www.saldlab.com )
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/array1d.h>
#include <cusp/linear_operator.h>

namespace cusp
{
namespace precond
{

/*! \addtogroup iterative_solvers Iterative Solvers
 *  \addtogroup preconditioners Preconditioners
 *  \brief Several preconditioners for iterative solvers
 *  \ingroup iterative_solvers
 *  \{
 */

/** \brief Diagonal preconditoner (aka Jacobi preconditioner)
 *
 *  \tparam ValueType Type used for matrix values (e.g. \c float or \c double).
 *  \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 *  \par Overview
 *  Given a matrix \c A to precondition, the diagonal preconditioner
 *  simply extracts the main diagonal \c D of a \c A and implements
 *  <tt>y = D^-1 x</tt> when applied to a vector \p x.
 *
 *  Diagonal preconditioning is very inexpensive to use, but has
 *  limited effectiveness.  However, if the matrix \p A has poorly
 *  scaled rows then diagonal preconditioning can substantially
 *  reduce the number of solver iterations required to reach
 *  convergence.
 *
 *  \par Example
 *  The following code snippet demonstrates how to use a
 *  \p diagonal preconditioner to solve a linear system.
 *
 *  \code
 *  #include <cusp/precond/diagonal.h>
 *
 *  int main(void)
 *  {
 *    // allocate storage for solution (x) and right hand side (b)
 *    cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);
 *    cusp::array1d<float, cusp::device_memory> b(A.num_rows, 1);
 *
 *    cusp::default_monitor<float> monitor(b, 100, 1e-6);
 *
 *    // setup preconditioner
 *    cusp::precond::diagonal<float, cusp::device_memory> M(A);
 *
 *    // solve
 *    cusp::krylov::bicgstab(A, x, b, monitor, M);
 *
 *    return 0;
 *  }
 *  \endcode
 */
template <typename ValueType, typename MemorySpace>
class diagonal : public linear_operator<ValueType, MemorySpace>
{
    typedef linear_operator<ValueType, MemorySpace> Parent;
    cusp::array1d<ValueType, MemorySpace> diagonal_reciprocals;

public:
    /*! construct a \p diagonal preconditioner
     *
     * \param A matrix to precondition
     * \tparam MatrixType matrix
     */
    template<typename MatrixType>
    diagonal(const MatrixType& A);

    /*! apply the preconditioner to vector \p x and store the result in \p y
     *
     * \param x input vector
     * \param y ouput vector
     * \tparam VectorType1 vector
     * \tparam VectorType2 vector
     */
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const;
};
/*! \}
 */

} // end namespace precond
} // end namespace cusp

#include <cusp/precond/detail/diagonal.inl>

