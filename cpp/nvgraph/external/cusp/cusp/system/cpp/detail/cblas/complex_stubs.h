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

#include <cusp/complex.h>
#include <cblas.h>

#define CUSP_CBLAS_COMPLEX_DEFS_1(FUNC_MACRO)                                               \
  FUNC_MACRO(cusp::complex<float> , float , c)                                              \
  FUNC_MACRO(cusp::complex<double>, double, z)

#define CUSP_CBLAS_COMPLEX_DEFS_2(FUNC_MACRO)                                               \
  FUNC_MACRO(cusp::complex<float> , float , sc)                                             \
  FUNC_MACRO(cusp::complex<double>, double, dz)

#define CUSP_CBLAS_COMPLEX_DEFS_3(FUNC_MACRO)                                               \
  FUNC_MACRO(cusp::complex<float> , float , cs)                                             \
  FUNC_MACRO(cusp::complex<double>, double, zd)

#define CUSP_CBLAS_COMPLEX_AMAX(T,V,name)                                                   \
  template<int dummy>                                                                       \
  int amax( const int n, const T* X, const int incX )                                       \
  {                                                                                         \
    return cblas_i##name##amax(n, (const V*) X, incX);                                      \
  }

#define CUSP_CBLAS_COMPLEX_ASUM(T,V,name)                                                   \
  template<int dummy>                                                                       \
  V asum( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##asum(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_COMPLEX_AXPY(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void axpy( const int n, const T alpha, const T* X, const int incX, T* Y, const int incY ) \
  {                                                                                         \
    cblas_##name##axpy(n, (const V*) &alpha, (const V*) X, incX, (V*) Y, incY);             \
  }

#define CUSP_CBLAS_COMPLEX_COPY(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void copy( const int n, const T* X, const int incX, T* Y, const int incY )                \
  {                                                                                         \
    cblas_##name##copy(n, (const V*) X, incX, (V*) Y, incY);                                \
  }

// #define CUSP_CBLAS_COMPLEX_DOTC(T,V,name)
//   T dotc( const int n, const T* X, const int incX, const T* Y, const int incY )
//   {
//     typedef typename thrust::detail::eval_if<
//       thrust::detail::is_same<V,float>::value,
//           thrust::detail::identity_<openblas_complex_float>,
//           thrust::detail::identity_<openblas_complex_double>
//     >::type Cmplx;
//     Cmplx z;
//     cblas_##name##dotc_sub(n, (const V*) X, incX, (const V*) Y, incY, &z);
//     return T(z.real, z.imag);
//   }
//
// #define CUSP_CBLAS_COMPLEX_DOTU(T,V,name)
//   T dotu( const int n, const T* X, const int incX, const T* Y, const int incY )
//   {
//     typedef typename thrust::detail::eval_if<
//       thrust::detail::is_same<V,float>::value,
//           thrust::detail::identity_<openblas_complex_float>,
//           thrust::detail::identity_<openblas_complex_double>
//     >::type Cmplx;
//     Cmplx z;
//     cblas_##name##dotu_sub(n, (const V*) X, incX, (const V*) Y, incY, &z);
//     return T(z.real, z.imag);
//   }

#define CUSP_CBLAS_COMPLEX_NRM2(T,V,name)                                                   \
  template<int dummy>                                                                       \
  V nrm2( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##nrm2(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_COMPLEX_SCAL(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void scal( const int n, const V alpha, T* X, const int incX )                             \
  {                                                                                         \
    cblas_##name##scal(n, alpha, (V*) X, incX);                                             \
  }

#define CUSP_CBLAS_COMPLEX_SWAP(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void swap( const int n, T* X, const int incX, T* Y, const int incY )                      \
  {                                                                                         \
    cblas_##name##swap(n, (V*) X, incX, (V*) Y, incY);                                      \
  }

#define CUSP_CBLAS_COMPLEX_GEMV(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void gemv(  CBLAS_ORDER order,  CBLAS_TRANSPOSE trans,                                    \
           int m, int n, T alpha, const T* A, int lda,                                      \
           const T* x, int incx, T beta, T* y, int incy)                                    \
{                                                                                           \
    cblas_##name##gemv(order, trans, m, n, (const V*) &alpha, (const V*) A, lda,            \
                       (const V*) x, incx, (const V*) &beta, (V*) y, incy);                 \
}

#define CUSP_CBLAS_COMPLEX_GERC(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void ger(  CBLAS_ORDER order, int m, int n, T alpha, const T* x, int incx,                \
            const T* y, int incy, T* A, int lda)                                            \
{                                                                                           \
    cblas_##name##gerc(order, m, n, (const V*) &alpha,                                      \
                      (const V*) x, incx, (const V*) y, incy,                               \
                      (V*) A, lda);                                                         \
}

#define CUSP_CBLAS_COMPLEX_HEMV(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void hemv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
             int n, T alpha, const T* A, int lda,                                           \
             const T* x, int incx, T beta, T* y, int incy)                                  \
{                                                                                           \
    cblas_##name##hemv(order, uplo, n, (const V*) &alpha, (const V*) A, lda,                \
                       (const V*) x, incx, (const V*) &beta, (V*) y, incy);                 \
}

#define CUSP_CBLAS_COMPLEX_HER(T,V,name)                                                    \
  template<int dummy>                                                                       \
  void her(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                           \
            int n, const V alpha, const T* x, int incx, T* A, int lda)                      \
{                                                                                           \
    cblas_##name##her(order, uplo, n, alpha,                                                \
                      (const V*) x, incx, (V*) A, lda);                                     \
}

#define CUSP_CBLAS_COMPLEX_TRMV(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void trmv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int n, const T* A, int lda, T* x, int incx)                                    \
{                                                                                           \
    cblas_##name##trmv(order, uplo, trans, diag, n,                                         \
                       (const V*) A, lda, (V*) x, incx);                                    \
}

#define CUSP_CBLAS_COMPLEX_TRSV(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void trsv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int n, const T* A, int lda, T* x, int incx)                                    \
{                                                                                           \
    cblas_##name##trsv(order, uplo, trans, diag, n,                                         \
                       (const V*) A, lda, (V*) x, incx);                                    \
}

#define CUSP_CBLAS_COMPLEX_GEMM(T,V,name)                                                   \
  template<int dummy>                                                                       \
void gemm(  CBLAS_ORDER order,                                                              \
            CBLAS_TRANSPOSE transa,  CBLAS_TRANSPOSE transb,                                \
           int m, int n, int k, T alpha, const T* A, int lda,                               \
           const T* B, int ldb, T beta, T* C, int ldc)                                      \
{                                                                                           \
    cblas_##name##gemm(order, transa, transb,                                               \
                       m, n, k, (const V*) &alpha, (const V*) A, lda,                       \
                       (const V*) B, ldb, (const V*) &beta, (V*) C, ldc);                   \
}

#define CUSP_CBLAS_COMPLEX_SYMM(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void symm(  CBLAS_ORDER order,                                                            \
              CBLAS_SIDE side,  CBLAS_UPLO uplo,                                            \
             int m, int n, T alpha, const T* A, int lda,                                    \
             const T* B, int ldb, T beta, T* C, int ldc)                                    \
{                                                                                           \
    cblas_##name##symm(order, side, uplo, m, n,                                             \
                       (const V*) &alpha, (const V*) A, lda,                                \
                       (const V*) B, ldb, (const V*) &beta, (V*) C, ldc);                   \
}

#define CUSP_CBLAS_COMPLEX_SYRK(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void syrk(  CBLAS_ORDER order,                                                            \
              CBLAS_UPLO uplo,  CBLAS_TRANSPOSE trans,                                      \
             int n, int k, T alpha, const T* A, int lda,                                    \
             T beta, T* C, int ldc)                                                         \
{                                                                                           \
    cblas_##name##syrk(order, uplo, trans, n, k,                                            \
                       (const V*) &alpha, (const V*) A, lda,                                \
                       (const V*) &beta, (V*) C, ldc);                                      \
}

#define CUSP_CBLAS_COMPLEX_SYR2K(T,V,name)                                                  \
  template<int dummy>                                                                       \
  void syr2k(  CBLAS_ORDER order,                                                           \
               CBLAS_UPLO uplo,  CBLAS_TRANSPOSE trans,                                     \
              int n, int k, T& alpha, const T* A, int lda,                                  \
              const T* B, int ldb, T& beta, T* C, int ldc)                                  \
{                                                                                           \
    cblas_##name##syr2k(order, uplo, trans, n, k,                                           \
                        (const V*) &alpha, (const V*) A, lda,                               \
                        (const V*) B, ldb, (const V*) &beta, (V*) C, ldc);                  \
}

#define CUSP_CBLAS_COMPLEX_TRMM(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void trmm(  CBLAS_ORDER order,                                                            \
              CBLAS_SIDE side,  CBLAS_UPLO uplo,                                            \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int m, int n, T alpha, const T* A, int lda,                                    \
             T* B, int ldb)                                                                 \
{                                                                                           \
    cblas_##name##trmm(order, side, uplo, trans, diag, m, n,                                \
                      (const V*) &alpha, (const V*) A, lda, (V*) B, ldb);                   \
}

#define CUSP_CBLAS_COMPLEX_TRSM(T,V,name)                                                   \
  template<int dummy>                                                                       \
  void trsm(  CBLAS_ORDER order,                                                            \
              CBLAS_SIDE side,  CBLAS_UPLO uplo,                                            \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int m, int n, T alpha, const T* A, int lda,                                    \
             T* B, int ldb)                                                                 \
{                                                                                           \
    cblas_##name##trsm(order, side, uplo, trans, diag, m, n,                                \
                       (const V*) &alpha, (const V*) A, lda, (V*) B, ldb);                  \
}

namespace cusp
{
namespace system
{
namespace cpp
{
namespace detail
{
namespace cblas
{

// LEVEL 1
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_AMAX);
CUSP_CBLAS_COMPLEX_DEFS_2(CUSP_CBLAS_COMPLEX_ASUM);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_AXPY);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_COPY);
// CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_DOTC);
CUSP_CBLAS_COMPLEX_DEFS_2(CUSP_CBLAS_COMPLEX_NRM2);
CUSP_CBLAS_COMPLEX_DEFS_3(CUSP_CBLAS_COMPLEX_SCAL);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_SWAP);

// LEVEL 2
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_GEMV);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_GERC);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_HEMV);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_HER);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_TRMV);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_TRSV);

// LEVEL 3
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_GEMM);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_SYMM);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_SYRK);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_SYR2K);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_TRMM);
CUSP_CBLAS_COMPLEX_DEFS_1(CUSP_CBLAS_COMPLEX_TRSM);

} // end namespace cblas
} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace cusp

