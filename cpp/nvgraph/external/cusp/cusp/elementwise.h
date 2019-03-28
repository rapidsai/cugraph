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

/*! \file elementwise.h
 *  \brief Elementwise operations on matrices.
 */

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/execution_policy.h>

namespace cusp
{

/*! \addtogroup algorithms Algorithms
 *  \addtogroup matrix_algorithms Matrix Algorithms
 *  \ingroup algorithms
 *  format
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
                 const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op);
/*! \endcond */

/**
 * \brief Perform transform operation on two matrices
 *
 * \tparam MatrixType1 Type of first matrix
 * \tparam MatrixType2 Type of second matrix
 * \tparam MatrixType3 Type of output matrix
 * \tparam BinaryFunction Type of binary transform to apply
 *
 * \param A first input matrix
 * \param B second input matrix
 * \param C output matrix
 * \param op binary transform to apply
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p elementwise.
 *
 *  \code
 *  #include <cusp/elementwise.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // initialize first 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *
 *      // print A
 *      cusp::print(A);
 *
 *      // initialize second 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> B(2,3);
 *      B(0,0) = 60;  B(0,1) = 50;  B(0,2) = 40;
 *      B(1,0) = 30;  B(1,1) = 20;  B(1,2) = 10;
 *
 *      // print B
 *      cusp::print(B);
 *
 *      // compute the sum
 *      cusp::array2d<float, cusp::host_memory> C;
 *      cusp::elementwise(A, B, C, thrust::plus<int>());
 *
 *      // print C
 *      cusp::print(C);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3,
          typename BinaryFunction>
void elementwise(const MatrixType1& A,
                 const MatrixType2& B,
                       MatrixType3& C,
                       BinaryFunction op);

/**
 * \brief Compute the sum of two matrices
 *
 * \tparam MatrixType1 Type of first matrix
 * \tparam MatrixType2 Type of second matrix
 * \tparam MatrixType3 Type of output matrix
 *
 * \param A first input matrix
 * \param B second input matrix
 * \param C output matrix
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p add.
 *
 *  \code
 *  #include <cusp/elementwise.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // initialize first 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *
 *      // print A
 *      cusp::print(A);
 *
 *      // initialize second 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> B(2,3);
 *      B(0,0) = 60;  B(0,1) = 50;  B(0,2) = 40;
 *      B(1,0) = 30;  B(1,1) = 20;  B(1,2) = 10;
 *
 *      // print B
 *      cusp::print(B);
 *
 *      // compute the sum
 *      cusp::array2d<float, cusp::host_memory> C;
 *      cusp::add(A, B, C);
 *
 *      // print C
 *      cusp::print(C);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void add(const MatrixType1& A,
         const MatrixType2& B,
               MatrixType3& C);

/**
 * \brief Compute the difference of two matrices
 *
 * \tparam MatrixType1 Type of first matrix
 * \tparam MatrixType2 Type of second matrix
 * \tparam MatrixType3 Type of output matrix
 *
 * \param A first input matrix
 * \param B second input matrix
 * \param C output matrix
 *
 * \par Example
 *  The following code snippet demonstrates how to use \p subtract.
 *
 *  \code
 *  #include <cusp/elementwise.h>
 *  #include <cusp/array2d.h>
 *  #include <cusp/print.h>
 *
 *  int main(void)
 *  {
 *      // initialize first 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> A(2,3);
 *      A(0,0) = 10;  A(0,1) = 20;  A(0,2) = 30;
 *      A(1,0) = 40;  A(1,1) = 50;  A(1,2) = 60;
 *
 *      // print A
 *      cusp::print(A);
 *
 *      // initialize second 2x3 matrix
 *      cusp::array2d<float, cusp::host_memory> B(2,3);
 *      B(0,0) = 60;  B(0,1) = 50;  B(0,2) = 40;
 *      B(1,0) = 30;  B(1,1) = 20;  B(1,2) = 10;
 *
 *      // print B
 *      cusp::print(B);
 *
 *      // compute the subtract
 *      cusp::array2d<float, cusp::host_memory> C;
 *      cusp::subtract(A, B, C);
 *
 *      // print C
 *      cusp::print(C);
 *
 *      return 0;
 *  }
 *  \endcode
 */
template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void subtract(const MatrixType1& A,
              const MatrixType2& B,
                    MatrixType3& C);
/*! \}
 */

} // end namespace cusp

#include <cusp/detail/elementwise.inl>

