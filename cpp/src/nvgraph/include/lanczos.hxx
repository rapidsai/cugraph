/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 #pragma once

#include "nvgraph_error.hxx"
#include "matrix.hxx"

namespace nvgraph {

  /// Compute smallest eigenvectors of symmetric matrix
  /** Computes eigenvalues and eigenvectors that are least
   *  positive. If matrix is positive definite or positive
   *  semidefinite, the computed eigenvalues are smallest in
   *  magnitude.
   *
   *  The largest eigenvalue is estimated by performing several
   *  Lanczos iterations. An implicitly restarted Lanczos method is
   *  then applied to A+s*I, where s is negative the largest
   *  eigenvalue.
   *
   *  CNMEM must be initialized before calling this function.
   *
   *  @param A Pointer to matrix object.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter Maximum number of Lanczos steps. Does not include
   *    Lanczos steps used to estimate largest eigenvalue.
   *  @param restartIter Maximum size of Lanczos system before
   *    performing an implicit restart. Should be at least 4.
   *  @param tol Convergence tolerance. Lanczos iteration will
   *    terminate when the residual norm is less than tol*theta, where
   *    theta is an estimate for the smallest unwanted eigenvalue
   *    (i.e. the (nEigVecs+1)th smallest eigenvalue).
   *  @param reorthogonalize Whether to reorthogonalize Lanczos
   *    vectors.
   *  @param iter On exit, pointer to total number of Lanczos
   *    iterations performed. Does not include Lanczos steps used to
   *    estimate largest eigenvalue.
   *  @param eigVals_dev (Output, device memory, nEigVecs entries)
   *    Smallest eigenvalues of matrix.
   *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
   *    Eigenvectors corresponding to smallest eigenvalues of
   *    matrix. Vectors are stored as columns of a column-major matrix
   *    with dimensions n x nEigVecs.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR computeSmallestEigenvectors(const Matrix<IndexType_,ValueType_> & A,
					 IndexType_ nEigVecs,
					 IndexType_ maxIter,
					 IndexType_ restartIter,
					 ValueType_ tol,
					 bool reorthogonalize,
					 IndexType_ & iter,
					 ValueType_ * __restrict__ eigVals_dev,
					 ValueType_ * __restrict__ eigVecs_dev);

  /// Compute smallest eigenvectors of symmetric matrix
  /** Computes eigenvalues and eigenvectors that are least
   *  positive. If matrix is positive definite or positive
   *  semidefinite, the computed eigenvalues are smallest in
   *  magnitude.
   *
   *  The largest eigenvalue is estimated by performing several
   *  Lanczos iterations. An implicitly restarted Lanczos method is
   *  then applied to A+s*I, where s is negative the largest
   *  eigenvalue.
   *
   *  @param A Pointer to matrix object.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter Maximum number of Lanczos steps. Does not include
   *    Lanczos steps used to estimate largest eigenvalue.
   *  @param restartIter Maximum size of Lanczos system before
   *    performing an implicit restart. Should be at least 4.
   *  @param tol Convergence tolerance. Lanczos iteration will
   *    terminate when the residual norm is less than tol*theta, where
   *    theta is an estimate for the smallest unwanted eigenvalue
   *    (i.e. the (nEigVecs+1)th smallest eigenvalue).
   *  @param reorthogonalize Whether to reorthogonalize Lanczos
   *    vectors.
   *  @param iter On exit, pointer to final size of Lanczos system.
   *  @param totalIter On exit, pointer to total number of Lanczos
   *    iterations performed. Does not include Lanczos steps used to
   *    estimate largest eigenvalue.
   *  @param shift On exit, pointer to matrix shift.
   *  @param alpha_host (Output, host memory, restartIter entries)
   *    Diagonal entries of Lanczos system.
   *  @param beta_host (Output, host memory, restartIter entries)
   *    Off-diagonal entries of Lanczos system.
   *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
   *    entries) Lanczos vectors. Vectors are stored as columns of a
   *    column-major matrix with dimensions n x (restartIter+1).
   *  @param work_dev (Output, device memory,
   *    (n+restartIter)*restartIter entries) Workspace.
   *  @param eigVals_dev (Output, device memory, nEigVecs entries)
   *    Smallest eigenvalues of matrix.
   *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
   *    Eigenvectors corresponding to smallest eigenvalues of
   *    matrix. Vectors are stored as columns of a column-major matrix
   *    with dimensions n x nEigVecs.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR computeSmallestEigenvectors(const Matrix<IndexType_,ValueType_> * A,
					 IndexType_ nEigVecs,
					 IndexType_ maxIter,
					 IndexType_ restartIter,
					 ValueType_ tol,
					 bool reorthogonalize,
					 IndexType_ * iter,
					 IndexType_ * totalIter,
					 ValueType_ * shift,
					 ValueType_ * __restrict__ alpha_host,
					 ValueType_ * __restrict__ beta_host,
					 ValueType_ * __restrict__ lanczosVecs_dev,
					 ValueType_ * __restrict__ work_dev,
					 ValueType_ * __restrict__ eigVals_dev,
					 ValueType_ * __restrict__ eigVecs_dev);

    /// Compute largest eigenvectors of symmetric matrix
  /** Computes eigenvalues and eigenvectors that are least
   *  positive. If matrix is positive definite or positive
   *  semidefinite, the computed eigenvalues are largest in
   *  magnitude.
   *
   *  The largest eigenvalue is estimated by performing several
   *  Lanczos iterations. An implicitly restarted Lanczos method is
   *  then applied.
   *
   *  @param A Matrix.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter Maximum number of Lanczos steps. 
   *  @param restartIter Maximum size of Lanczos system before
   *    performing an implicit restart. Should be at least 4.
   *  @param tol Convergence tolerance. Lanczos iteration will
   *    terminate when the residual norm is less than tol*theta, where
   *    theta is an estimate for the largest unwanted eigenvalue
   *    (i.e. the (nEigVecs+1)th largest eigenvalue).
   *  @param reorthogonalize Whether to reorthogonalize Lanczos
   *    vectors.
   *  @param effIter On exit, pointer to final size of Lanczos system.
   *  @param totalIter On exit, pointer to total number of Lanczos
   *    iterations performed.
   *  @param alpha_host (Output, host memory, restartIter entries)
   *    Diagonal entries of Lanczos system.
   *  @param beta_host (Output, host memory, restartIter entries)
   *    Off-diagonal entries of Lanczos system.
   *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
   *    entries) Lanczos vectors. Vectors are stored as columns of a
   *    column-major matrix with dimensions n x (restartIter+1).
   *  @param work_dev (Output, device memory,
   *    (n+restartIter)*restartIter entries) Workspace.
   *  @param eigVals_dev (Output, device memory, nEigVecs entries)
   *    Largest eigenvalues of matrix.
   *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
   *    Eigenvectors corresponding to largest eigenvalues of
   *    matrix. Vectors are stored as columns of a column-major matrix
   *    with dimensions n x nEigVecs.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR computeLargestEigenvectors(const Matrix<IndexType_,ValueType_> * A,
           IndexType_ nEigVecs,
           IndexType_ maxIter,
           IndexType_ restartIter,
           ValueType_ tol,
           bool reorthogonalize,
           IndexType_ * effIter,
           IndexType_ * totalIter,
           ValueType_ * __restrict__ alpha_host,
           ValueType_ * __restrict__ beta_host,
           ValueType_ * __restrict__ lanczosVecs_dev,
           ValueType_ * __restrict__ work_dev,
           ValueType_ * __restrict__ eigVals_dev,
           ValueType_ * __restrict__ eigVecs_dev);

    /// Compute largest eigenvectors of symmetric matrix
  /** Computes eigenvalues and eigenvectors that are least
   *  positive. If matrix is positive definite or positive
   *  semidefinite, the computed eigenvalues are largest in
   *  magnitude.
   *
   *  The largest eigenvalue is estimated by performing several
   *  Lanczos iterations. An implicitly restarted Lanczos method is
   *  then applied to A+s*I, where s is negative the largest
   *  eigenvalue.
   *
   *  CNMEM must be initialized before calling this function.
   *
   *  @param A Matrix.
   *  @param nEigVecs Number of eigenvectors to compute.
   *  @param maxIter Maximum number of Lanczos steps. Does not include
   *    Lanczos steps used to estimate largest eigenvalue.
   *  @param restartIter Maximum size of Lanczos system before
   *    performing an implicit restart. Should be at least 4.
   *  @param tol Convergence tolerance. Lanczos iteration will
   *    terminate when the residual norm is less than tol*theta, where
   *    theta is an estimate for the largest unwanted eigenvalue
   *    (i.e. the (nEigVecs+1)th largest eigenvalue).
   *  @param reorthogonalize Whether to reorthogonalize Lanczos
   *    vectors.
   *  @param iter On exit, pointer to total number of Lanczos
   *    iterations performed. Does not include Lanczos steps used to
   *    estimate largest eigenvalue.
   *  @param eigVals_dev (Output, device memory, nEigVecs entries)
   *    Largest eigenvalues of matrix.
   *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
   *    Eigenvectors corresponding to largest eigenvalues of
   *    matrix. Vectors are stored as columns of a column-major matrix
   *    with dimensions n x nEigVecs.
   *  @return NVGRAPH error flag.
   */
  template <typename IndexType_, typename ValueType_>
  NVGRAPH_ERROR computeLargestEigenvectors(const Matrix<IndexType_,ValueType_> & A,
           IndexType_ nEigVecs,
           IndexType_ maxIter,
           IndexType_ restartIter,
           ValueType_ tol,
           bool reorthogonalize,
           IndexType_ & iter,
           ValueType_ * __restrict__ eigVals_dev,
           ValueType_ * __restrict__ eigVecs_dev);

}

