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

#pragma once

#include <cusp/detail/config.h>

#include <cusp/detail/execution_policy.h>

namespace cusp
{
namespace precond
{
namespace aggregation
{

// Forward definition
template<typename> struct sa_level;

/* \cond */
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void symmetric_strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                      const MatrixType1& A,
                                            MatrixType2& S,
                                      const double theta = 0.0);
/* \endcond */

/*  Compute a strength of connection matrix using the standard symmetric measure.
 *  An off-diagonal connection A[i,j] is strong iff::
 *
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 *  With the default threshold (theta = 0.0) all connections are strong.
 *
 *  Note: explicit diagonal entries are always considered strong.
 */
template <typename MatrixType1,
          typename MatrixType2>
void symmetric_strength_of_connection(const MatrixType1& A,
                                            MatrixType2& S,
                                      const double theta = 0.0);

/* \cond */
template<typename DerivedPolicy,
         typename MatrixType1,
         typename MatrixType2,
         typename ArrayType>
void evolution_strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                                      const MatrixType1& A,
                                            MatrixType2& S,
                                      const ArrayType& B,
                                      const double rho_DinvA = 0.0,
                                      const double epsilon = 4.0);
/* \endcond */

/*  Compute a strength of connection matrix using the standard symmetric measure.
 *  An off-diagonal connection A[i,j] is strong iff::
 *
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 *  With the default threshold (theta = 0.0) all connections are strong.
 *
 *  Note: explicit diagonal entries are always considered strong.
 */
template<typename MatrixType1,
         typename MatrixType2,
         typename ArrayType>
void evolution_strength_of_connection(const MatrixType1& A,
                                            MatrixType2& S,
                                      const ArrayType& B,
                                      const double rho_DinvA = 0.0,
                                      const double epsilon = 4.0);


/* \cond */
template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2>
void strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S);

template <typename DerivedPolicy,
          typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void strength_of_connection(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                            const MatrixType1& A,
                                  MatrixType2& S,
                                  sa_level<MatrixType3>& level);
/* \endcond */

/*  Compute a strength of connection matrix using the standard symmetric measure.
 *  An off-diagonal connection A[i,j] is strong iff::
 *
 *     abs(A[i,j]) >= theta * sqrt( abs(A[i,i]) * abs(A[j,j]) )
 *
 *  With the default threshold (theta = 0.0) all connections are strong.
 *
 *  Note: explicit diagonal entries are always considered strong.
 */
template <typename MatrixType1,
          typename MatrixType2>
void strength_of_connection(const MatrixType1& A,
                                  MatrixType2& S);

template <typename MatrixType1,
          typename MatrixType2,
          typename MatrixType3>
void strength_of_connection(const MatrixType1& A,
                                  MatrixType2& S,
                                  sa_level<MatrixType3>& level);

} // end namespace aggregation
} // end namespace precond
} // end namespace cusp

#include <cusp/precond/aggregation/detail/strength.inl>

