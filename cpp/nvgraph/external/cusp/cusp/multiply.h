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

/*! \file multiply.h
 *  \brief Matrix multiplication
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C);
/*! \endcond */

/**
 * \brief Implements matrix-matrix and matrix-vector multiplication
 *
 * \par Overview
 *
 * \p multiply can be used with dense matrices, sparse matrices, and user-defined
 * \p linear_operator objects.
 *
 * \tparam LinearOperator Type of first matrix
 * \tparam MatrixOrVector1 Type of second matrix or vector
 * \tparam MatrixOrVector2 Type of output matrix or vector
 *
 * \param A input matrix
 * \param B input matrix or vector
 * \param C output matrix or vector
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p multiply to
 *  compute a matrix-vector product.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // initialize matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,2);
 *      A(0,0) = 10;  A(0,1) = 20;
 *      A(1,0) = 40;  A(1,1) = 50;
 *
 *      // initialize input vector
 *      cusp::array1d<float, cusp::host_memory> x(2);
 *      x[0] = 1;
 *      x[1] = 2;
 *
 *      // allocate output vector
 *      cusp::array1d<float, cusp::host_memory> y(2);
 *
 *      // compute y = A * x
 *      cusp::multiply(A, x, y);
 *
 *      // print y
 *      cusp::print(y);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
void multiply(const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C);

/*! \cond */
template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C,
                    UnaryFunction  initialize,
                    BinaryFunction1 combine,
                    BinaryFunction2 reduce);
/*! \endcond */

/**
 * \brief Implements matrix-vector multiplication with custom combine and
 * reduce functionality
 *
 * \par Overview
 *
 * \p multiply can be used with dense matrices, sparse matrices, and user-defined
 * \p linear_operator objects.
 *
 * \tparam LinearOperator  Type of matrix
 * \tparam MatrixOrVector1 Type of second vector
 * \tparam MatrixOrVector2 Type of output vector
 * \tparam UnaryFunction   Type of unary function to initialize RHS
 * \tparam BinaryFunction1 Type of binary function to combine entries
 * \tparam BinaryFunction2 Type of binary function to reduce entries
 *
 * \param A input matrix
 * \param B input vector
 * \param C output vector
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p multiply to
 *  compute a matrix-vector product.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/functional.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // define multiply functors
 *      cusp::constant_functor<float> initialize;
 *      thrust::multiplies<float> combine;
 *      thrust::plus<float>       reduce;
 *
 *      // initialize matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,2);
 *      A(0,0) = 10;  A(0,1) = 20;
 *      A(1,0) = 40;  A(1,1) = 50;
 *
 *      // initialize input vector
 *      cusp::array1d<float, cusp::host_memory> x(2);
 *      x[0] = 1;
 *      x[1] = 2;
 *
 *      // allocate output vector
 *      cusp::array1d<float, cusp::host_memory> y(2);
 *
 *      // compute y = A * x
 *      cusp::multiply(A, x, y, initialize, combine, reduce);
 *
 *      // print y
 *      cusp::print(y);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void multiply(const LinearOperator&  A,
              const MatrixOrVector1& B,
                    MatrixOrVector2& C,
                    UnaryFunction  initialize,
                    BinaryFunction1 combine,
                    BinaryFunction2 reduce);

/*! \cond */
template <typename DerivedPolicy,
          typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spgemm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                        const LinearOperator&  A,
                        const MatrixOrVector1& B,
                              MatrixOrVector2& C,
                              UnaryFunction   initialize,
                              BinaryFunction1 combine,
                              BinaryFunction2 reduce);
/*! \endcond */

/**
 * \brief Implements generalized matrix-matrix multiplication
 *
 * \par Overview
 *
 * \p generalized multiply can be used with dense and sparse matrices, and user-defined
 * \p linear_operator objects. This function compute nonzeros only for the
 * entries present in the output matrix. Entries that are not specified in the
 * output matrix are disregarded (annihilated). This specification
 * significantly reduces the computational work corresponding to computing the
 * sparsity of the output and performing unnecessary (combine, reduce)
 * operations.
 *
 * \tparam LinearOperator  Type of matrix
 * \tparam MatrixOrVector1 Type of second matrix
 * \tparam MatrixOrVector2 Type of output matrix
 * \tparam UnaryFunction   Type of unary function to initialize RHS
 * \tparam BinaryFunction1 Type of binary function to combine entries
 * \tparam BinaryFunction2 Type of binary function to reduce entries
 *
 * \param A first input matrix
 * \param B second input matrix
 * \param C output matrix
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p generalized_spgemm to
 *  compute a matrix-matrix product.
 *
 *  \code
 *  #include <cusp/coo_matrix.h>
 *  #include <cusp/functional.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/print.h>
 *
 *  #include <cusp/gallery/poisson.h>
 *
 *  int main(void)
 *  {
 *      // define multiply functors
 *      thrust::identity<float>   identity;
 *      cusp::constant_functor<float> zero;
 *      thrust::multiplies<float> combine;
 *      thrust::plus<float>       reduce;
 *
 *      // initialize matrix
 *      cusp::coo_matrix<int,float,cusp::host_memory> A;
 *      cusp::gallery::poisson5pt(A, 3, 3);
 *
 *      // allocate output matrices and initialize output nonzeros
 *      cusp::coo_matrix<int, float, cusp::host_memory> B(A);
 *      cusp::coo_matrix<int, float, cusp::host_memory> C(A);
 *
 *      // compute B = A * A
 *      cusp::generalized_spgemm(A, A, B, zero, combine, reduce);
 *      // compute C += A * A
 *      cusp::generalized_spgemm(A, A, C, identity, combine, reduce);
 *
 *      // print output matrices
 *      cusp::print(B);
 *      cusp::print(C);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2,
          typename UnaryFunction,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spgemm(const LinearOperator&  A,
                        const MatrixOrVector1& B,
                              MatrixOrVector2& C,
                              UnaryFunction   initialize,
                              BinaryFunction1 combine,
                              BinaryFunction2 reduce);

/*! \cond */
template <typename DerivedPolicy,
          typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                      const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                            Vector3& z,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce);
/*! \endcond */

/**
 * \brief Implements generalized matrix-vector multiplication
 *
 * \par Overview
 *
 * \p generalized multiply can be used with dense and sparse matrices, and user-defined
 * \p linear_operator objects.
 *
 * \tparam LinearOperator Type of first matrix
 * \tparam Vector1 Type of second input vector
 * \tparam Vector2 Type of third  input vector
 * \tparam Vector3 Type of output vector
 *
 * \param A input matrix
 * \param x input vector
 * \param y input vector
 * \param z output vector
 *
 * \par Example
 *
 *  The following code snippet demonstrates how to use \p multiply to
 *  compute a matrix-vector product.
 *
 *  \code
 *  #include <cusp/array1d.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/multiply.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // define multiply functors
 *      thrust::multiplies<float>          combine;
 *      thrust::plus<float>                reduce;
 *
 *      // initialize matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,2);
 *      A(0,0) = 10;  A(0,1) = 20;
 *      A(1,0) = 40;  A(1,1) = 50;
 *
 *      // initialize input vector
 *      cusp::array1d<float, cusp::host_memory> x(2);
 *      x[0] = 1;
 *      x[1] = 2;
 *
 *      // initial RHS filled with 2's
 *      cusp::constant_array<float> y(2, 2);
 *
 *      // allocate output vector
 *      cusp::array1d<float, cusp::host_memory> z(2);
 *
 *      // compute z = y + (A * x)
 *      cusp::generalized_spmv(A, x, y, z, combine, reduce);
 *
 *      // print z
 *      cusp::print(z);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename LinearOperator,
          typename Vector1,
          typename Vector2,
          typename Vector3,
          typename BinaryFunction1,
          typename BinaryFunction2>
void generalized_spmv(const LinearOperator&  A,
                      const Vector1& x,
                      const Vector2& y,
                            Vector3& z,
                            BinaryFunction1 combine,
                            BinaryFunction2 reduce);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/multiply.inl>

