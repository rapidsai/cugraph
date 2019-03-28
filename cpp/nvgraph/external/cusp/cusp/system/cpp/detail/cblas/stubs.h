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

#include <cblas.h>

#include <cusp/complex.h>

#define CUSP_CBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                                             \
  FUNC_MACRO(float , float , s)                                                             \
  FUNC_MACRO(double, double, d)

#define CUSP_CBLAS_AMAX(T,V,name)                                                           \
  template<int dummy>                                                                       \
  int amax( const int n, const T* X, const int incX )                                       \
  {                                                                                         \
    return cblas_i##name##amax(n, (const V*) X, incX);                                      \
  }

#define CUSP_CBLAS_ASUM(T,V,name)                                                           \
  template<int dummy>                                                                       \
  V asum( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##asum(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_AXPY(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void axpy( const int n, const T alpha, const T* X, const int incX, T* Y, const int incY ) \
  {                                                                                         \
    cblas_##name##axpy(n, alpha, (const V*) X, incX, (V*) Y, incY);                         \
  }

#define CUSP_CBLAS_COPY(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void copy( const int n, const T* X, const int incX, T* Y, const int incY )                \
  {                                                                                         \
    cblas_##name##copy(n, (const V*) X, incX, (V*) Y, incY);                                \
  }

#define CUSP_CBLAS_DOT(T,V,name)                                                            \
  template<int dummy>                                                                       \
  T dot( const int n, const T* X, const int incX, const T* Y, const int incY )              \
  {                                                                                         \
    return cblas_##name##dot(n, (const V*) X, incX, (const V*) Y, incY);                    \
  }

#define CUSP_CBLAS_NRM2(T,V,name)                                                           \
  template<int dummy>                                                                       \
  V nrm2( const int n, const T* X, const int incX )                                         \
  {                                                                                         \
    return cblas_##name##nrm2(n, (const V*) X, incX);                                       \
  }

#define CUSP_CBLAS_SCAL(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void scal( const int n, const T alpha, T* X, const int incX )                             \
  {                                                                                         \
    cblas_##name##scal(n, alpha, (V*) X, incX);                                             \
  }

#define CUSP_CBLAS_SWAP(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void swap( const int n, T* X, const int incX, T* Y, const int incY )                      \
  {                                                                                         \
    cblas_##name##swap(n, (V*) X, incX, (V*) Y, incY);                                      \
  }

#define CUSP_CBLAS_GEMV(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void gemv(  CBLAS_ORDER order,  CBLAS_TRANSPOSE trans,                                    \
              int m, int n, T alpha, const T* A, int lda,                                   \
              const T* x, int incx, T beta, T* y, int incy)                                 \
{                                                                                           \
    cblas_##name##gemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);          \
}

#define CUSP_CBLAS_GER(T,V,name)                                                            \
  template<int dummy>                                                                       \
  void ger(  CBLAS_ORDER order, int m, int n, T alpha, const T* x, int incx,                \
            const T* y, int incy, T* A, int lda)                                            \
{                                                                                           \
    cblas_##name##ger(order, m, n, alpha,                                                   \
                      (const V*) x, incx, (const V*) y, incy,                               \
                      (V*) A, lda);                                                         \
}

#define CUSP_CBLAS_SYMV(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void symv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
             int n, T alpha, const T* A, int lda,                                           \
             const T* x, int incx, T beta, T* y, int incy)                                  \
{                                                                                           \
    cblas_##name##symv(order, uplo, n, alpha, (const V*) A, lda,                            \
                       (const V*) x, incx, beta, (V*) y, incy);                             \
}

#define CUSP_CBLAS_SYR(T,V,name)                                                            \
  template<int dummy>                                                                       \
  void syr(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                           \
            int n, T alpha, const T* x, int incx, T* A, int lda)                            \
{                                                                                           \
    cblas_##name##syr(order, uplo, n, alpha,                                                \
                      (const V*) x, incx, (V*) A, lda);                                     \
}

#define CUSP_CBLAS_TRMV(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void trmv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int n, const T* A, int lda, T* x, int incx)                                    \
{                                                                                           \
    cblas_##name##trmv(order, uplo, trans, diag, n,                                         \
                       (const V*) A, lda, (V*) x, incx);                                    \
}

#define CUSP_CBLAS_TRSV(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void trsv(  CBLAS_ORDER order,  CBLAS_UPLO uplo,                                          \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int n, const T* A, int lda, T* x, int incx)                                    \
{                                                                                           \
    cblas_##name##trsv(order, uplo, trans, diag, n,                                         \
                       (const V*) A, lda, (V*) x, incx);                                    \
}

#define CUSP_CBLAS_GEMM(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void gemm(  CBLAS_ORDER order,                                                            \
              CBLAS_TRANSPOSE transa,  CBLAS_TRANSPOSE transb,                              \
              int m, int n, int k, T alpha, const T* A, int lda,                            \
              const T* B, int ldb, T beta, T* C, int ldc)                                   \
{                                                                                           \
    cblas_##name##gemm(order, transa, transb,                                               \
                       m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                       \
}

#define CUSP_CBLAS_SYMM(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void symm(  CBLAS_ORDER order,                                                            \
              CBLAS_SIDE side,  CBLAS_UPLO uplo,                                            \
             int m, int n, T alpha, const T* A, int lda,                                    \
             const T* B, int ldb, T beta, T* C, int ldc)                                    \
{                                                                                           \
    cblas_##name##symm(order, side, uplo, m, n,                                             \
                       (V) alpha, (const V*) A, lda, (const V*) B, ldb,                     \
                       (V) beta, (V*) C, ldc);                                              \
}

#define CUSP_CBLAS_SYRK(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void syrk(  CBLAS_ORDER order,                                                            \
              CBLAS_UPLO uplo,  CBLAS_TRANSPOSE trans,                                      \
             int n, int k, T alpha, const T* A, int lda,                                    \
             T beta, T* C, int ldc)                                                         \
{                                                                                           \
    cblas_##name##syrk(order, uplo, trans, n, k,                                            \
                       (V) alpha, (const V*) A, lda,                                        \
                       (V) beta, (V*) C, ldc);                                              \
}

#define CUSP_CBLAS_SYR2K(T,V,name)                                                          \
  template<int dummy>                                                                       \
  void syr2k(  CBLAS_ORDER order,                                                           \
               CBLAS_UPLO uplo,  CBLAS_TRANSPOSE trans,                                     \
              int n, int k, T& alpha, const T* A, int lda,                                  \
              const T* B, int ldb, T& beta, T* C, int ldc)                                  \
{                                                                                           \
    cblas_##name##syr2k(order, uplo, trans, n, k,                                           \
                        alpha, (const V*) A, lda,                                           \
                        (const V*) B, ldb, beta, (V*) C, ldc);                              \
}

#define CUSP_CBLAS_TRMM(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void trmm(  CBLAS_ORDER order,                                                            \
              CBLAS_SIDE side,  CBLAS_UPLO uplo,                                            \
              CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                      \
             int m, int n, T alpha, const T* A, int lda,                                    \
             T* B, int ldb)                                                                 \
{                                                                                           \
    cblas_##name##trmm(order, side, uplo, trans, diag, m, n,                                \
                       (V) alpha, (const V*) A, lda, (V*) B, ldb);                          \
}

#define CUSP_CBLAS_TRSM(T,V,name)                                                           \
  template<int dummy>                                                                       \
  void trsm( CBLAS_ORDER order,                                                             \
             CBLAS_SIDE side,  CBLAS_UPLO uplo,                                             \
             CBLAS_TRANSPOSE trans,  CBLAS_DIAG diag,                                       \
             int m, int n, T alpha, const T* A, int lda,                                    \
             T* B, int ldb)                                                                 \
{                                                                                           \
    cblas_##name##trsm(order, side, uplo, trans, diag, m, n,                                \
                       (V) alpha, (const V*) A, lda, (V*) B, ldb);                          \
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
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_AMAX);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_ASUM);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_AXPY);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_COPY);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_DOT);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_NRM2);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SCAL);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SWAP);

// LEVEL 2
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_GEMV);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_GER);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SYMV);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SYR);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_TRMV);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_TRSV);

// LEVEL 3
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_GEMM);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SYMM);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SYRK);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_SYR2K);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_TRMM);
CUSP_CBLAS_EXPAND_REAL_DEFS(CUSP_CBLAS_TRSM);

} // end namespace cblas
} // end namespace detail
} // end namespace cpp
} // end namespace system
} // end namespace cusp

