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

/*! \file lapack.inl
 *  \brief Two-dimensional array
 */

#pragma once


#include <cusp/complex.h>
#include <cublas_v2.h>

#define CUSP_CUBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                \
  FUNC_MACRO(float , float , S)                                 \
  FUNC_MACRO(double, double, D)

#define CUSP_CUBLAS_EXPAND_DEFS_LOWER(FUNC_MACRO)               \
  FUNC_MACRO(float , float , s)                                 \
  FUNC_MACRO(double, double, d)                                 \
  FUNC_MACRO(cusp::complex<float> , cuComplex,  c)              \
  FUNC_MACRO(cusp::complex<double>, cuDoubleComplex, z)

#define CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_1(FUNC_MACRO)           \
  FUNC_MACRO(cusp::complex<float> , cuComplex,  C)              \
  FUNC_MACRO(cusp::complex<double>, cuDoubleComplex, Z)

#define CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_2(FUNC_MACRO)           \
  FUNC_MACRO(cusp::complex<float> , cuComplex,  Sc)             \
  FUNC_MACRO(cusp::complex<double>, cuDoubleComplex, Dz)

#define CUSP_CUBLAS_EXPAND_DEFS_1(FUNC_MACRO)                   \
  CUSP_CUBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                      \
  CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_1(FUNC_MACRO)

#define CUSP_CUBLAS_EXPAND_DEFS_2(FUNC_MACRO)                   \
  CUSP_CUBLAS_EXPAND_REAL_DEFS(FUNC_MACRO)                      \
  CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_2(FUNC_MACRO)

#define CUSP_CUBLAS_AMAX(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t amax( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX, int& result )                 \
  {                                                                                           \
    return cublasI##name##amax(handle, n, (const V*) X, incX, &result);                       \
  }

#define CUSP_CUBLAS_ASUM(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t asum( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX,                               \
                       typename cusp::norm_type<T>::type& result )                            \
  {                                                                                           \
    typedef typename cusp::norm_type<T>::type Real;                                           \
    return cublas##name##asum(handle, n, (const V*) X, incX, (Real*) &result);                \
  }

#define CUSP_CUBLAS_AXPY(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t axpy( cublasHandle_t handle,                                                 \
                       const int n, const T& alpha, const T* X, const int incX,               \
                       T* Y, const int incY )                                                 \
  {                                                                                           \
    return cublas##name##axpy(handle, n, (const V*) &alpha, (const V*) X, incX,               \
                              (V*) Y, incY);                                                  \
  }

#define CUSP_CUBLAS_COPY(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t copy( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX, T* Y, const int incY )        \
  {                                                                                           \
    return cublas##name##copy(handle, n, (const V*) X, incX, (V*) Y, incY);                   \
  }

#define CUSP_CUBLAS_DOT(T,V,name)                                                             \
  template<int dummy>                                                                         \
  cublasStatus_t dot( cublasHandle_t handle,                                                  \
                      const int n, const T* X, const int incX, const T* Y, const int incY,    \
                      T& result )                                                             \
  {                                                                                           \
    return cublas##name##dot(handle, n, (const V*) X, incX, (const V*) Y, incY, &result);     \
  }

#define CUSP_CUBLAS_DOTC(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t dotc( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX, const T* Y,                   \
                       const int incY, T& ret )                                               \
  {                                                                                           \
    return cublas##name##dotc(handle, n, (const V*) X, incX, (const V*) Y, incY, (V*) &ret);  \
  }

#define CUSP_CUBLAS_DOTU(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t dotu( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX, const T* Y,                   \
                       const int incY, T& ret )                                               \
  {                                                                                           \
    return cublas##name##dotu(handle, n, (const V*) X, incX, (const V*) Y, incY, (V*) &ret);  \
  }

#define CUSP_CUBLAS_NRM2(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t nrm2( cublasHandle_t handle,                                                 \
                       const int n, const T* X, const int incX,                               \
                       typename cusp::norm_type<T>::type& result)                             \
  {                                                                                           \
    return cublas##name##nrm2(handle, n, (const V*) X, incX, &result);                        \
  }

#define CUSP_CUBLAS_SCAL(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t scal( cublasHandle_t handle,                                                 \
                       const int n, const T alpha, T* X, const int incX )                     \
  {                                                                                           \
    return cublas##name##scal(handle, n, (const V*) &alpha, (V*) X, incX);                    \
  }

#define CUSP_CUBLAS_SWAP(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t swap( cublasHandle_t handle,                                                 \
                       const int n, T* X, const int incX, T* Y, const int incY )              \
  {                                                                                           \
    return cublas##name##swap(handle, n, (V*) X, incX, (V*) Y, incY);                         \
  }

#define CUSP_CUBLAS_GEMV(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t gemv( cublasHandle_t handle,                                                 \
                       cublasOperation_t trans,                                               \
                       int m, int n, T alpha, const T* A, int lda,                            \
                       const T* x, int incx, T beta, T* y, int incy)                          \
{                                                                                             \
    return cublas##name##gemv(handle, trans, m, n, (const V*) &alpha, (const V*) A, lda,      \
                              (const V*) x, incx, (const V*) &beta, (V*) y, incy);            \
}

#define CUSP_CUBLAS_GER(T,V,name)                                                             \
  template<int dummy>                                                                         \
  cublasStatus_t ger( cublasHandle_t handle,                                                  \
                      int m, int n, T alpha, const T* x, int incx, const T* y, int incy,      \
                      T* A, int lda)                                                          \
{                                                                                             \
    return cublas##name##ger(handle, m, n, (const V*) &alpha,                                 \
                             (const V*) x, incx, (const V*) y, incy,                          \
                             (V*) A, lda);                                                    \
}

#define CUSP_CUBLAS_SYMV(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t symv( cublasHandle_t handle, cublasFillMode_t uplo,                          \
                       int n, T alpha, const T* A, int lda,                                   \
                       const T* x, int incx, T beta, T* y, int incy)                          \
{                                                                                             \
    return cublas##name##symv(handle, uplo, n, (const V*) &alpha, (const V*) A, lda,          \
                              (const V*) x, incx, (const V*) &beta, (V*) y, incy);            \
}

#define CUSP_CUBLAS_SYR(T,V,name)                                                             \
  template<int dummy>                                                                         \
  cublasStatus_t syr( cublasHandle_t handle, cublasFillMode_t uplo,                           \
                      int n, T alpha, const T* x, int incx, T* A, int lda)                    \
{                                                                                             \
    return cublas##name##syr(handle, uplo, n, (const V*) &alpha,                              \
                             (const V*) x, incx, (V*) A, lda);                                \
}

#define CUSP_CUBLAS_TRMV(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t trmv( cublasHandle_t handle, cublasFillMode_t uplo,                          \
                       cublasOperation_t trans, cublasDiagType_t diag,                        \
                       int n, const T* A, int lda, T* x, int incx)                            \
{                                                                                             \
    return cublas##name##trmv(handle, uplo, trans, diag, n,                                   \
                              (const V*) A, lda, (V*) x, incx);                               \
}

#define CUSP_CUBLAS_TRSV(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t trsv( cublasHandle_t handle, cublasFillMode_t uplo,                          \
                       cublasOperation_t trans, cublasDiagType_t diag,                        \
                       int n, const T* A, int lda, T* x, int incx)                            \
{                                                                                             \
    return cublas##name##trsv(handle, uplo, trans, diag, n,                                   \
                              (const V*) A, lda, (V*) x, incx);                               \
}

#define CUSP_CUBLAS_GEMM(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t gemm( cublasHandle_t handle,                                                 \
                       cublasOperation_t transa, cublasOperation_t transb,                    \
                       int m, int n, int k, T alpha, const T* A, int lda,                     \
                       const T* B, int ldb, T beta, T* C, int ldc)                            \
{                                                                                             \
    return cublas##name##gemm(handle, transa, transb, m, n, k,                                \
                              (const V*) &alpha, (const V*) A, lda, (const V*) B, ldb,        \
                              (const V*) &beta, (V*) C, ldc);                                 \
}

#define CUSP_CUBLAS_SYMM(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t symm( cublasHandle_t handle,                                                 \
                       cublasSideMode_t side, cublasFillMode_t uplo,                          \
                       int m, int n, T alpha, const T* A, int lda,                            \
                       const T* B, int ldb, T beta, T* C, int ldc)                            \
{                                                                                             \
    return cublas##name##symm(handle, side, uplo, m, n,                                       \
                              (const V*) &alpha, (const V*) A, lda, (const V*) B, ldb,        \
                              (const V*) &beta, (V*) C, ldc);                                 \
}

#define CUSP_CUBLAS_SYRK(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t syrk( cublasHandle_t handle,                                                 \
                       cublasFillMode_t uplo, cublasOperation_t trans,                        \
                       int n, int k, T alpha, const T* A, int lda,                            \
                       T beta, T* B, int ldb)                                                 \
{                                                                                             \
    return cublas##name##syrk(handle, uplo, trans, n, k,                                      \
                              (const V*) &alpha, (const V*) A, lda,                           \
                              (const V*) &beta, (V*) B, ldb);                                 \
}

#define CUSP_CUBLAS_SYR2K(T,V,name)                                                           \
  template<int dummy>                                                                         \
  cublasStatus_t syr2k( cublasHandle_t handle,                                                \
                        cublasFillMode_t uplo, cublasOperation_t trans,                       \
                        int n, int k, T alpha, const T* A, int lda,                           \
                        const T* B, int ldb, T beta, T* C, int ldc)                           \
{                                                                                             \
    return cublas##name##syr2k(handle, uplo, trans, n, k,                                     \
                               (const V*) &alpha, (const V*) A, lda,                          \
                               (const V*) B, ldb, (const V*) &beta, (V*) C, ldc);             \
}

#define CUSP_CUBLAS_TRMM(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t trmm( cublasHandle_t handle,                                                 \
                       cublasSideMode_t side, cublasFillMode_t uplo,                          \
                       cublasOperation_t trans, cublasDiagType_t diag,                        \
                       int m, int n, T alpha, const T* A, int lda,                            \
                       const T* B, int ldb, T* C, int ldc)                                    \
{                                                                                             \
    return cublas##name##trmm(handle, side, uplo, trans, diag, m, n,                          \
                              (const V*) &alpha, (const V*) A, lda,                           \
                              (const V*) B, ldb, (V*) C, ldc);                                \
}

#define CUSP_CUBLAS_TRSM(T,V,name)                                                            \
  template<int dummy>                                                                         \
  cublasStatus_t trsm( cublasHandle_t handle,                                                 \
                       cublasSideMode_t side, cublasFillMode_t uplo,                          \
                       cublasOperation_t trans, cublasDiagType_t diag,                        \
                       int m, int n, T alpha, const T* A, int lda,                            \
                       T* B, int ldb)                                                         \
{                                                                                             \
    return cublas##name##trsm(handle, side, uplo, trans, diag, m, n,                          \
                              (const V*) &alpha, (const V*) A, lda,                           \
                              (V*) B, ldb);                                                   \
}

namespace cusp
{
namespace system
{
namespace cuda
{
namespace detail
{
namespace cublas
{

// LEVEL 1
CUSP_CUBLAS_EXPAND_DEFS_LOWER(CUSP_CUBLAS_AMAX);
CUSP_CUBLAS_EXPAND_DEFS_2(CUSP_CUBLAS_ASUM);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_AXPY);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_COPY);
CUSP_CUBLAS_EXPAND_REAL_DEFS(CUSP_CUBLAS_DOT);
CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_1(CUSP_CUBLAS_DOTC);
CUSP_CUBLAS_EXPAND_COMPLEX_DEFS_1(CUSP_CUBLAS_DOTU);
CUSP_CUBLAS_EXPAND_DEFS_2(CUSP_CUBLAS_NRM2);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SCAL);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SWAP);

// LEVEL 2
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_GEMV);
CUSP_CUBLAS_EXPAND_REAL_DEFS(CUSP_CUBLAS_GER);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SYMV);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SYR);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_TRMV);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_TRSV);

// LEVEL 3
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_GEMM);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SYMM);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SYRK);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_SYR2K);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_TRMM);
CUSP_CUBLAS_EXPAND_DEFS_1(CUSP_CUBLAS_TRSM);

} // end namespace cublas
} // end namespace detail
} // end namespace cuda
} // end namespace system
} // end namespace cusp

