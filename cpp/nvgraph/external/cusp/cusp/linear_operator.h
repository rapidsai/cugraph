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

/*! \file linear_operator.h
 *  \brief Abstract interface for iterative solvers
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/format.h>
#include <cusp/exception.h>
#include <cusp/blas/blas.h>
#include <cusp/detail/matrix_base.h>

namespace cusp
{

/**
 * \brief Abstract representation of a linear operator
 *
 * \tparam IndexType Type used for operator indices (e.g. \c int).
 * \tparam ValueType Type used for operator values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p linear operator is a abstract container that supports encapsulates
 *  abstract linear operators for use with other routines. All linear operators
 *  should provide a implementation of the operator()(x,y) for interoperability
 *  with the \p multiply routine.
 *
 * \par Example
 *  The following code snippet demonstrates how to create a custom
 *  linear operator.
 *
 *  \code
 * // include linear_operator header file
 * #include <cusp/linear_operator.h>
 *
 * #include <cusp/csr_matrix.h>
 * #include <cusp/multiply.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 * #include <cusp/precond/diagonal.h>
 *
 * template <typename MatrixType>
 * struct Dinv_A : public cusp::linear_operator<typename MatrixType::value_type, typename MatrixType::memory_space>
 * {
 *   typedef typename MatrixType::value_type ValueType;
 *   typedef typename MatrixType::memory_space MemorySpace;
 *
 *   const MatrixType& A;
 *   const cusp::precond::diagonal<ValueType,MemorySpace> Dinv;
 *
 *   Dinv_A(const MatrixType& A)
 *       : A(A), Dinv(A),
 *         cusp::linear_operator<ValueType,MemorySpace>(A.num_rows, A.num_cols, A.num_entries + A.num_rows)
 *   {}
 *
 *   template <typename Array1, typename Array2>
 *   void operator()(const Array1& x, Array2& y) const
 *   {
 *       cusp::multiply(A,x,y);
 *       cusp::multiply(Dinv,y,y);
 *   }
 * };
 *
 * int main(void)
 * {
 *   typedef cusp::csr_matrix<int, float, cusp::device_memory> CsrMatrix;
 *
 *   CsrMatrix A;
 *
 *   // number of entries
 *   const int N = 4;
 *
 *   // construct Poisson example matrix
 *   cusp::gallery::poisson5pt(A, N, N);
 *
 *   // construct instance of custom operator perform D^{-1}A
 *   Dinv_A<CsrMatrix> M(A);
 *
 *   // initialize x and y vectors
 *   cusp::array1d<float, cusp::device_memory> x(A.num_rows, 1);
 *   cusp::array1d<float, cusp::device_memory> y(A.num_rows, 0);
 *
 *   // call operator()(x,y) through multiply interface
 *   cusp::multiply(M,x,y);
 *
 *   // print the transformed vector
 *   cusp::print(y);
 * }
 *  \endcode
 */
template <typename ValueType, typename MemorySpace, typename IndexType=int>
class linear_operator : public cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::unknown_format>
{
private:

    typedef cusp::detail::matrix_base<IndexType,ValueType,MemorySpace,cusp::unknown_format> Parent;

public:

    /*! Construct an empty \p linear_operator.
     */
    linear_operator(void)
        : Parent() {}

    /*! Construct a \p linear_operator with a specific shape.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     */
    linear_operator(IndexType num_rows, IndexType num_cols)
        : Parent(num_rows, num_cols) {}

    /*! Construct a \p linear_operator with a specific shape and number of
     * nonzero entries.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     *  \param num_entries Number of nonzero entries.
     */
    linear_operator(IndexType num_rows, IndexType num_cols, IndexType num_entries)
        : Parent(num_rows, num_cols, num_entries) {}
}; // linear_operator

/**
 * \brief Simple identity operator
 *
 * \tparam IndexType Type used for operator indices (e.g. \c int).
 * \tparam ValueType Type used for operator values (e.g. \c float).
 * \tparam MemorySpace A memory space (e.g. \c cusp::host_memory or \c cusp::device_memory)
 *
 * \par Overview
 *  A \p linear operator that copies the input vector to the output vector
 *  unchanged. Corresponds to the identity matrix (I).
 *
 * \par Example
 *  The following code snippet demonstrates using the identity operator.
 *
 *  \code
 * // include linear_operator header file
 * #include <cusp/linear_operator.h>
 * #include <cusp/print.h>
 *
 * int main(void)
 * {
 *   // number of entries
 *   const int N = 4;
 *
 *   // construct instance of identity operator
 *   cusp::identity_operator A(N);
 *
 *   // initialize x and y vectors
 *   cusp::array1d<float, cusp::device_memory> x(A.num_rows, 1);
 *   cusp::array1d<float, cusp::device_memory> y(A.num_rows, 0);
 *
 *   // call operator()(x,y) through multiply interface
 *   A(x,y);
 *
 *   // print the transformed vector
 *   cusp::print(y);
 * }
 *  \endcode
 */
template <typename ValueType, typename MemorySpace, typename IndexType=int>
class identity_operator : public linear_operator<ValueType,MemorySpace,IndexType>
{
private:

    typedef linear_operator<ValueType,MemorySpace> Parent;

public:

    /*! Construct an empty \p identity_operator.
     */
    identity_operator(void)
        : Parent() {}

    /*! Construct a \p identity_operator with a specific shape.
     *
     *  \param num_rows Number of rows.
     *  \param num_cols Number of columns.
     */
    identity_operator(IndexType num_rows, IndexType num_cols)
        : Parent(num_rows, num_cols) {}

    template <typename DerivedPolicy, typename VectorType1, typename VectorType2>
    void operator()(thrust::execution_policy<DerivedPolicy>& exec, const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::copy(exec, x, y);
    }

    /*! Apply the \p identity_operator to vector x and produce vector y.
     *
     * \tparam VectorType1 Type of the input vector
     * \tparam VectorType2 Type of the output vector
     *
     *  \param x Input vector to copy.
     *  \param y Output vector to produce.
     */
    template <typename VectorType1, typename VectorType2>
    void operator()(const VectorType1& x, VectorType2& y) const
    {
        cusp::blas::copy(x, y);
    }
}; // identity_operator

} // end namespace cusp

