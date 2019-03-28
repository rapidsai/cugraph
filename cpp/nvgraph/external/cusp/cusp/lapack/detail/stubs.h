/*
 *  Copyright 2008-2009 NVIDIA Corporation
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

#define CUSP_LAPACK_EXPAND_REAL_DEFS(FUNC_MACRO)                                                           \
  FUNC_MACRO(float , float , s)                                                                            \
  FUNC_MACRO(double, double, d)

#define CUSP_LAPACK_EXPAND_COMPLEX_DEFS(FUNC_MACRO)                                                        \
  FUNC_MACRO(cusp::complex<float> , lapack_complex_float , c)                                              \
  FUNC_MACRO(cusp::complex<double>, lapack_complex_double, z)

#define CUSP_LAPACK_EXPAND_DEFS(FUNC_MACRO)                                                                \
  CUSP_LAPACK_EXPAND_REAL_DEFS(FUNC_MACRO)                                                                 \
  CUSP_LAPACK_EXPAND_COMPLEX_DEFS(FUNC_MACRO)

#define CUSP_LAPACK_GETRF(T,V,name)                                                                        \
  lapack_int getrf( lapack_int order, lapack_int m, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv ) \
  {                                                                                                        \
    return LAPACKE_##name##getrf(order, m, n, (V*) a, lda, ipiv);                                          \
  }

#define CUSP_LAPACK_POTRF(T,V,name)                                                                        \
  lapack_int potrf( lapack_int order, char uplo, lapack_int n, T* a, lapack_int lda )                      \
  {                                                                                                        \
    return LAPACKE_##name##potrf(order, uplo, n, (V*) a, lda);                                             \
  }

#define CUSP_LAPACK_SYTRF(T,V,name)                                                                        \
  lapack_int sytrf( lapack_int order, char uplo, lapack_int n, T* a, lapack_int lda, lapack_int* ipiv )    \
  {                                                                                                        \
    return LAPACKE_##name##sytrf(order, uplo, n, (V*) a, lda, ipiv);                                       \
  }

#define CUSP_LAPACK_GETRS(T,V,name)                                                                        \
  lapack_int getrs( lapack_int order, char trans, lapack_int n, lapack_int nrhs, const T* a,               \
  lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb )                                           \
  {                                                                                                        \
    return LAPACKE_##name##getrs(order, trans, n, nrhs, (V*) a, lda, ipiv, (V*) b, ldb);                   \
  }

#define CUSP_LAPACK_POTRS(T,V,name)                                                                        \
  lapack_int potrs( lapack_int order, char uplo, lapack_int n, lapack_int nrhs, const T* a,                \
  lapack_int lda, T* b, lapack_int ldb )                                                                   \
  {                                                                                                        \
    return LAPACKE_##name##potrs(order, uplo, n, nrhs, (V*) a, lda, (V*) b, ldb);                          \
  }

#define CUSP_LAPACK_SYTRS(T,V,name)                                                                        \
  lapack_int sytrs( lapack_int order, char uplo, lapack_int n, lapack_int nrhs, const T* a,                \
  lapack_int lda, const lapack_int* ipiv, T* b, lapack_int ldb )                                           \
  {                                                                                                        \
    return LAPACKE_##name##sytrs(order, uplo, n, nrhs, (V*) a, lda, ipiv, (V*) b, ldb);                    \
  }

#define CUSP_LAPACK_TRTRS(T,V,name)                                                                        \
  lapack_int trtrs( lapack_int order, char uplo, char trans, char diag, lapack_int n, lapack_int nrhs,     \
  const T* a, lapack_int lda, T* b, lapack_int ldb )                                                       \
  {                                                                                                        \
    return LAPACKE_##name##trtrs(order, uplo, trans, diag, n, nrhs, (V*) a, lda, (V*) b, ldb);             \
  }

#define CUSP_LAPACK_GETRI(T,V,name)                                                                        \
  lapack_int getri( lapack_int order, lapack_int n, T* a, lapack_int lda, const lapack_int* ipiv )         \
  {                                                                                                        \
    return LAPACKE_##name##getri(order, n, (V*) a, lda, ipiv);                                             \
  }

#define CUSP_LAPACK_POTRI(T,V,name)                                                                        \
  lapack_int potri( lapack_int order, char uplo, lapack_int n, T* a, lapack_int lda )                      \
  {                                                                                                        \
    return LAPACKE_##name##potri(order, uplo, n, (V*) a, lda);                                             \
  }

#define CUSP_LAPACK_SYTRI(T,V,name)                                                                        \
  lapack_int sytri( lapack_int order, char uplo, lapack_int n, T* a, lapack_int lda,                       \
  const lapack_int* ipiv )                                                                                 \
  {                                                                                                        \
    return LAPACKE_##name##sytri(order, uplo, n, (V*) a, lda, ipiv);                                       \
  }

#define CUSP_LAPACK_TRTRI(T,V,name)                                                                        \
  lapack_int trtri( lapack_int order, char uplo, char diag, lapack_int n, T* a, lapack_int lda )           \
  {                                                                                                        \
    return LAPACKE_##name##trtri(order, uplo, diag, n, (V*) a, lda);                                       \
  }

#define CUSP_LAPACK_SYEV(T,V,name)                                                                         \
  lapack_int syev( lapack_int order, char job, char uplo, lapack_int n, T* a, lapack_int lda, T* w )       \
  {                                                                                                        \
    return LAPACKE_##name##syev(order, job, uplo, n, (V*) a, lda, (V*) w);                                 \
  }

#define CUSP_LAPACK_STEV(T,V,name)                                                                         \
  lapack_int stev( lapack_int order, char job, lapack_int n, T* a, T* b, T* z, lapack_int ldz )            \
  {                                                                                                        \
    return LAPACKE_##name##stev(order, job, n, (V*) a, (V*) b, (V*) z, ldz);                               \
  }

#define CUSP_LAPACK_SYGV(T,V,name)                                                                         \
  lapack_int sygv( lapack_int order, lapack_int itype, char job, char uplo, lapack_int n,                  \
                   T* a, lapack_int lda, T* b, lapack_int ldb, T* w )                                      \
  {                                                                                                        \
    return LAPACKE_##name##sygv(order, itype, job, uplo, n, (V*) a, lda, (V*) b, ldb, (V*) w);             \
  }

#define CUSP_LAPACK_GESV(T,V,name)                                                                         \
  lapack_int gesv( lapack_int order, lapack_int n, lapack_int nrhs, T* a, lapack_int lda,                  \
                   lapack_int* ipiv, T* b, lapack_int ldb)                                                 \
  {                                                                                                        \
    return LAPACKE_##name##gesv(order, n, nrhs, (V*) a, lda, ipiv, (V*) b, ldb);                           \
  }

namespace cusp
{
namespace lapack
{
namespace detail
{

// FACTORIZE
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_GETRF);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_POTRF);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_SYTRF);

// FACTORIZED SOLVES
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_GETRS);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_POTRS);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_SYTRS);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_TRTRS);

// INVERT
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_GETRI);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_POTRI);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_SYTRI);
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_TRTRI);

// EIGENSOLVERS
CUSP_LAPACK_EXPAND_REAL_DEFS(CUSP_LAPACK_SYEV);
CUSP_LAPACK_EXPAND_REAL_DEFS(CUSP_LAPACK_STEV);
CUSP_LAPACK_EXPAND_REAL_DEFS(CUSP_LAPACK_SYGV);

// GENERIC SOLVERS
CUSP_LAPACK_EXPAND_DEFS(CUSP_LAPACK_GESV);

} // end namespace detail
} // end namespace lapack
} // end namespace cusp

