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

/*! \file blas.h
 *  \brief BLAS-like functions
 */

#pragma once

#include <cusp/detail/config.h>
#include <cusp/detail/execution_policy.h>

#include <cusp/complex.h>

namespace cusp
{
namespace blas
{

/*! \addtogroup dense Dense Algorithms
 *  \addtogroup blas BLAS
 *  \ingroup dense
 *  \brief Interface to BLAS routines
 *  \{
 */

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
int amax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const ArrayType& x);
/*! \endcond */

/**
 * \brief index of the largest element in a array
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find max value
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *
 *   // find index of max absolute value in x
 *   int index = cusp::blas::amax(x);
 *
 *   std::cout << "Max value at pos: " << index << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
int amax(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x);
/*! \endcond */

/**
 * \brief sum of absolute value of all entries in array
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to compute sum of absolute values
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *
 *   // find index of max absolute value in x
 *   float sum = cusp::blas::asum(x);
 *
 *   std::cout << "asum(x) =" << sum << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
asum(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief scaled vector addition (y = alpha * x + y)
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 * \tparam ScalarType Type of the scale factor
 *
 * \param x The input array
 * \param y The output array to store the result
 * \param alpha The scale factor applied to array x
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty source array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // create a destination array filled with 1s
 *   cusp::array1d<float,cusp::host_memory> y(10, 1);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute y += 1.5*x
 *   cusp::blas::axpy(x, y, 1.5);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ScalarType>
void axpy(const ArrayType1& x,
                ArrayType2& y,
          const ScalarType alpha);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
           const ScalarType1 alpha,
           const ScalarType2 beta);
/*! \endcond */

/**
 * \brief compute linear combination of two vectors (z = alpha * x + beta * y)
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 * \tparam ArrayType3 Type of the third input array
 * \tparam ScalarType1 Type of the first scale factor
 * \tparam ScalarType2 Type of the second scale factor
 *
 * \param x The first input array
 * \param y The second input array
 * \param z The output array to store the result
 * \param alpha The scale factor applied to array x
 * \param beta The scale factor applied to array y
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create two empty source arrays
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::array1d<float,cusp::host_memory> y(10);
 *
 *   // create a destination array
 *   cusp::array1d<float,cusp::host_memory> z(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand1(10, 0);
 *   cusp::blas::copy(rand1, x);
 *
 *   // fill y array with random values
 *   cusp::random_array<float> rand2(10, 7);
 *   cusp::blas::copy(rand2, y);
 *
 *   // compute z = 1.5*x + 2.0*y
 *   cusp::blas::axpby(x, y, z, 1.5, 2.0);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ScalarType1,
          typename ScalarType2>
void axpby(const ArrayType1& x,
           const ArrayType2& y,
                 ArrayType3& z,
           const ScalarType1 alpha,
           const ScalarType2 beta);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
              const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& output,
              const ScalarType1 alpha,
              const ScalarType2 beta,
              const ScalarType3 gamma);
/*! \endcond */

/**
 * \brief compute linear combination of three vectors (output = alpha * x + beta * y + gamma * z)
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 * \tparam ArrayType3 Type of the third input array
 * \tparam ArrayType4 Type of the input/output array
 * \tparam ScalarType1 Type of the first scale factor
 * \tparam ScalarType2 Type of the second scale factor
 * \tparam ScalarType3 Type of the third scale factor
 *
 * \param x The first input array
 * \param y The second input array
 * \param z The third input array
 * \param w The output array to store the result
 * \param alpha The scale factor applied to array x
 * \param beta The scale factor applied to array y
 * \param gamma The scale factor applied to array z
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create two empty source arrays
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::array1d<float,cusp::host_memory> y(10);
 *   cusp::array1d<float,cusp::host_memory> z(10);
 *
 *   // create a destination array
 *   cusp::array1d<float,cusp::host_memory> w(10);
 *
 *   // fill x array with random values
 *   cusp::random_array<float> rand1(10, 0);
 *   cusp::blas::copy(rand1, x);
 *
 *   // fill y array with random values
 *   cusp::random_array<float> rand2(10, 7);
 *   cusp::blas::copy(rand2, y);
 *
 *   // fill z array with random values
 *   cusp::random_array<float> rand3(10, 4);
 *   cusp::blas::copy(rand3, z);
 *
 *   // compute w = 1.5*x + 2.0*y + 2.1*z
 *   cusp::blas::axpbypcz(x, y, z, w, 1.5, 2.0, 2.1);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3,
          typename ArrayType4,
          typename ScalarType1,
          typename ScalarType2,
          typename ScalarType3>
void axpbypcz(const ArrayType1& x,
              const ArrayType2& y,
              const ArrayType3& z,
                    ArrayType4& w,
              const ScalarType1 alpha,
              const ScalarType2 beta,
              const ScalarType3 gamma);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z);
/*! \endcond */

/**
 * \brief elementwise multiplication of two vectors (z[i] = x[i] * y[i])
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 * \tparam ArrayType3 Type of the output array
 *
 * \param x The first input array
 * \param y The second input array
 * \param z The output array
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // create an array filled with 3s
 *   cusp::array1d<float,cusp::host_memory> y(10, 3);
 *
 *   // create an empty array
 *   cusp::array1d<float,cusp::host_memory> z(10);
 *
 *   // multiply arrays x and y
 *   cusp::blas::xmy(x, y, z);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2,
          typename ArrayType3>
void xmy(const ArrayType1& x,
         const ArrayType2& y,
               ArrayType3& z);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
void copy(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const ArrayType1& x,
                ArrayType2& y);
/*! \endcond */

/**
 * \brief vector copy (y = x)
 *
 * \tparam ArrayType1 Type of the input array
 * \tparam ArrayType2 Type of the output array
 *
 * \param x The input array
 * \param y The output array
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // create an empty array
 *   cusp::array1d<float,cusp::host_memory> y(10);
 *
 *   // copy array x into y
 *   cusp::blas::copy(x, y);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2>
void copy(const ArrayType1& x,
                ArrayType2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
    const ArrayType1& x,
    const ArrayType2& y);
/*! \endcond */

/**
 * \brief dot product (x^T * y)
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 *
 * \param x The first input array
 * \param y The second input array
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // create an array filled with 3s
 *   cusp::array1d<float,cusp::host_memory> y(10, 3);
 *
 *   // compute dot product of array x into y
 *   float value = cusp::blas::dot(x, y);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dot(const ArrayType1& x,
    const ArrayType2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType1& x,
     const ArrayType2& y);
/*! \endcond */

/**
 * \brief conjugate dot product (conjugate(x)^T * y)
 *
 * \tparam ArrayType1 Type of the first input array
 * \tparam ArrayType2 Type of the second input array
 *
 * \param x The first input array
 * \param y The second input array
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // create an array filled with 3s
 *   cusp::array1d<float,cusp::host_memory> y(10, 3);
 *
 *   // compute dot product of array x into y
 *   float value = cusp::blas::dotc(x, y);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType1,
          typename ArrayType2>
typename ArrayType1::value_type
dotc(const ArrayType1& x,
     const ArrayType2& y);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void fill(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          ArrayType& array,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief vector fill (x[i] = alpha)
 *
 * \tparam ArrayType Type of the input array
 * \tparam ScalarType Type of the fill value
 *
 * \param x The input array to fill
 * \param alpha Value to fill array x
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *
 *   // fill x array with 1s
 *   cusp::blas::fill(x, 1);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType,
          typename ScalarType>
void fill(ArrayType& x,
          const ScalarType alpha);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& array);
/*! \endcond */

/**
 * \brief vector 1-norm (sum abs(x[i]))
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find 2-norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print 1-norm
 *   float nrm_x = cusp::blas::nrm1(x);
 *   std::cout << "nrm1(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm1(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
     const ArrayType& x);
/*! \endcond */

/**
 * \brief vector 2-norm (sqrt(sum x[i] * x[i] )
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find 2-norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print 2-norm
 *   float nrm_x = cusp::blas::nrm2(x);
 *   std::cout << "nrm2(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrm2(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
       const ArrayType& x);
/*! \endcond */

/**
 * \brief vector infinity norm
 *
 * \tparam ArrayType Type of the input array
 *
 * \param x The input array to find infinity norm
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with random values
 *   cusp::array1d<float,cusp::host_memory> x(10);
 *   cusp::random_array<float> rand(10);
 *   cusp::blas::copy(rand, x);
 *
 *   // compute and print infinity norm
 *   float nrm_x = cusp::blas::nrmmax(x);
 *   std::cout << "nrmmax(x) = " << nrm_x << std::endl;
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType>
typename cusp::norm_type<typename ArrayType::value_type>::type
nrmmax(const ArrayType& x);

/*! \cond */
template <typename DerivedPolicy,
          typename ArrayType,
          typename ScalarType>
void scal(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          ArrayType& x,
          const ScalarType alpha);
/*! \endcond */

/**
 * \brief scale vector (x[i] = alpha * x[i])
 *
 * \tparam ArrayType  Type of the input array
 * \tparam ScalarType Type of the scalar value
 *
 * \param x The input array to scale
 * \param alpha The scale factor
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/print.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an array initially filled with 2s
 *   cusp::array1d<float,cusp::host_memory> x(10, 2);
 *
 *   // scal x by 2
 *   cusp::blas::scal(x, 2);
 *
 *   // print the scaled vector
 *   cusp::print(x);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename ArrayType,
          typename ScalarType>
void scal(ArrayType& x,
          const ScalarType alpha);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void gemv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Computes a matrix-vector product using a general matrix
 *
 * \tparam Array2d1 Type of the input matrix
 * \tparam Array1d1 Type of the input vector
 * \tparam Array1d2 Type of the output vector
 *
 * \param A General matrix
 * \param x Input vector
 * \param y Output vector
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty output array
 *   cusp::array1d<float,cusp::host_memory> y(A.num_rows);
 *
 *   // multiply A and x to produce y
 *   cusp::blas::gemv(A, x, y);
 *
 *   // print the contents of y
 *   cusp::print(y);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array1d1,
         typename Array1d2>
void gemv(const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1d1,
          typename Array1d2,
          typename Array2d1>
void ger(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const typename Array2d1::value_type alpha = 1.0);
/*! \endcond */

/**
 * \brief Performs a rank-1 update of a general matrix.
 *
 * \tparam Array1d1 Type of the first input array
 * \tparam Array1d2 Type of the second input array
 * \tparam Array2d1 Type of the output matrix
 *
 * \param x First n-element array
 * \param y Second n-element array
 * \param A An n-by-n general matrix
 * \param alpha Scale input by alpha
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create 2 random dense arrays
 *   cusp::random_array<float> rand1(10, 0);
 *   cusp::array1d<float,cusp::host_memory> x(rand1);
 *
 *   cusp::random_array<float> rand2(10, 7);
 *   cusp::array1d<float,cusp::host_memory> y(rand2);
 *
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A(10,10);
 *
 *   // compute n-by-n general update
 *   cusp::blas::ger(x, y, A);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array1d1,
         typename Array1d2,
         typename Array2d1>
void ger(const Array1d1& x,
         const Array1d2& y,
               Array2d1& A,
         const typename Array2d1::value_type alpha = 1.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Computes a matrix-vector product using a symmetric matrix
 *
 * \tparam Array2d1 Type of the input matrix
 * \tparam Array1d1 Type of the input vector
 * \tparam Array1d2 Type of the output vector
 *
 * \param A Symmetric matrix
 * \param x Input vector
 * \param y Output vector
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty output array
 *   cusp::array1d<float,cusp::host_memory> y(A.num_rows);
 *
 *   // multiply A and x to produce y
 *   cusp::blas::symv(A, x, y);
 *
 *   // print the contents of y
 *   cusp::print(y);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename Array2d1,
          typename Array1d1,
          typename Array1d2>
void symv(const Array2d1& A,
          const Array1d1& x,
                Array1d2& y,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array1d,
          typename Array2d>
void syr(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
         const Array1d& x,
               Array2d& A,
         const typename Array1d::value_type alpha = 1.0);
/*! \endcond */

/**
 * \brief Performs a rank-1 update of a symmetric matrix.
 *
 * \tparam Array1d Type of the input array
 * \tparam Array2d Type of the output matrix
 *
 * \param x An n-element array
 * \param A An n-by-n symmetric matrix
 * \param alpha Scale input by alpha
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an random dense array
 *   cusp::random_array<float> rand(10);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A(10,10);
 *
 *   // compute rank-1 update
 *   cusp::blas::syr(x, A);
 *
 *   // print the contents of A
 *   cusp::print(A);
 *
 *   return 0;
 * }
 * \endcode
 */
template <typename Array1d,
          typename Array2d>
void syr(const Array1d& x,
               Array2d& A,
         const typename Array1d::value_type alpha = 1.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trmv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x);
/*! \endcond */

/**
 * \brief Computes a matrix-vector product using a triangular matrix
 *
 * \tparam Array2d1 Type of the input matrix
 * \tparam Array1d1 Type of the input vector
 * \tparam Array1d2 Type of the output vector
 *
 * \param A Triangular matrix
 * \param x Input vector
 * \param x Output vector
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // create an empty output array
 *   cusp::array1d<float,cusp::host_memory> y(A.num_rows);
 *
 *   // multiply A and x to produce y
 *   cusp::blas::trmv(A, x, y);
 *
 *   // print the contents of y
 *   cusp::print(y);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d,
         typename Array1d>
void trmv(const Array2d& A,
                Array1d& x);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d,
          typename Array1d>
void trsv(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d& A,
                Array1d& x);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d Type of the symmetric input matrix
 * \tparam Array1d Type of the input right-hand side
 *
 * \param A Upper or lower triangle of a symmetric matrix
 * \param x Right-hand side vector
 *
 * \par Example
 * \code
 * #include <cusp/array1d.h>
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create an random dense array
 *   cusp::random_array<float> rand(A.num_rows);
 *   cusp::array1d<float,cusp::host_memory> x(rand);
 *
 *   // solve for RHS vector
 *   cusp::blas::trsv(A, x);
 *
 *   // print the contents of x
 *   cusp::print(x);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d,
         typename Array1d>
void trsv(const Array2d& A,
                Array1d& x);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void gemm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Computes a matrix-matrix product with general matrices.
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A First input matrix
 * \param B Second input matrix
 * \param C Output matrix
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // allocate space for output matrix
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, A.num_cols);
 *
 *   // compute output matrix
 *   cusp::blas::gemm(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void gemm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void symm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Computes a matrix-matrix product where one input matrix is symmetric.
 *
 * \tparam Array2d1 Type of the first symmetric input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A First symmetric input matrix
 * \param B Second input matrix
 * \param C Output matrix
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // allocate space for output matrix
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, A.num_cols);
 *
 *   // compute output matrix
 *   cusp::blas::symm(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void symm(const Array2d1& A,
          const Array2d2& B,
                Array2d3& C,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void syrk(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Performs a symmetric rank-k update.
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A First input matrix
 * \param B Second input matrix
 * \param C Output matrix
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // allocate space for output matrix
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, A.num_cols);
 *
 *   // compute rank-k update
 *   cusp::blas::syrk(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void syrk(const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0,
          const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2,
          typename Array2d3>
void syr2k(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
           const typename Array2d1::value_type alpha = 1.0,
           const typename Array2d1::value_type beta  = 0.0);
/*! \endcond */

/**
 * \brief Performs a symmetric rank-2k update.
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A First input matrix
 * \param B Second input matrix
 * \param C Output matrix
 * \param alpha Scale input by alpha
 * \param beta  Scale output by beta
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // allocate space for output matrix
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, A.num_cols);
 *
 *   // compute rank-2 update
 *   cusp::blas::syr2k(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2,
         typename Array2d3>
void syr2k(const Array2d1& A,
           const Array2d2& B,
                 Array2d3& C,
           const typename Array2d1::value_type alpha = 1.0,
           const typename Array2d1::value_type beta  = 0.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trmm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0);
/*! \endcond */

/**
 * \brief Computes a matrix-matrix product where one input matrix is triangular.
 *
 * \tparam Array2d1 Type of the first triangular input matrix
 * \tparam Array2d2 Type of the second input matrix
 * \tparam Array2d3 Type of the output matrix
 *
 * \param A First triangular input matrix
 * \param B Second input matrix
 * \param C Output matrix
 * \param alpha Scale input by alpha
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // allocate space for output matrix
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, A.num_cols);
 *
 *   // compute output matrix
 *   cusp::blas::trmm(A, A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void trmm(const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0);

/*! \cond */
template <typename DerivedPolicy,
          typename Array2d1,
          typename Array2d2>
void trsm(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
          const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0);
/*! \endcond */

/**
 * \brief Solve a triangular matrix equation
 *
 * \tparam Array2d1 Type of the first input matrix
 * \tparam Array2d2 Type of the output matrix
 *
 * \param A Contains the upper or lower triangle of a symmetric matrix
 * \param B Contains block of right-hand side vectors
 * \param alpha Scale input by alpha
 *
 * \par Example
 * \code
 * #include <cusp/array2d.h>
 * #include <cusp/print.h>
 *
 * #include <cusp/gallery/poisson.h>
 *
 * // include cusp blas header file
 * #include <cusp/blas/blas.h>
 *
 * int main()
 * {
 *   // create an empty dense matrix structure
 *   cusp::array2d<float,cusp::host_memory> A;
 *
 *   // create 2D Poisson problem
 *   cusp::gallery::poisson5pt(A, 4, 4);
 *
 *   // create a set of random RHS vectors
 *   cusp::array2d<float,cusp::host_memory> B(A.num_rows, 5);
 *
 *   // fill B with random values
 *   cusp::random_array<float> rand(B.num_entries);
 *   cusp::blas::copy(rand, B.values);
 *
 *   // solve multiple RHS vectors
 *   cusp::blas::trsm(A, B);
 *
 *   // print the contents of B
 *   cusp::print(B);
 *
 *   return 0;
 * }
 * \endcode
 */
template<typename Array2d1,
         typename Array2d2>
void trsm(const Array2d1& A,
                Array2d2& B,
          const typename Array2d1::value_type alpha = 1.0);

/*! \}
 */

} // end namespace blas
} // end namespace cusp

#include <cusp/detail/blas.inl>

