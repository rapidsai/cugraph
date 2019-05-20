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

//#ifdef NVGRAPH_PARTITION

#define _USE_MATH_DEFINES
#include <math.h>
#include "include/lanczos.hxx"

#include <stdio.h>
#include <time.h>

#include <cuda.h>

#define USE_CURAND 1

#ifdef USE_CURAND
  #include <curand.h>
#endif

#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_vector_kernels.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_lapack.hxx"
#include "include/debug_macros.h"
// =========================================================
// Useful macros
// =========================================================

// Get index of matrix entry
#define IDX(i,j,lda) ((i)+(j)*(lda))

// =========================================================
// Macros and functions for cuRAND
// =========================================================
//#ifdef USE_CURAND
//namespace {
//
//  /// Get message string from cuRAND status code
//  //static
//  //const char* curandGetErrorString(curandStatus_t e) {
//  //  switch(e) {
//  //  case CURAND_STATUS_SUCCESS:
//  //    return "CURAND_STATUS_SUCCESS";
//  //  case CURAND_STATUS_VERSION_MISMATCH:
//  //    return "CURAND_STATUS_VERSION_MISMATCH";
//  //  case CURAND_STATUS_NOT_INITIALIZED:
//  //    return "CURAND_STATUS_NOT_INITIALIZED";
//  //  case CURAND_STATUS_ALLOCATION_FAILED:
//  //    return "CURAND_STATUS_ALLOCATION_FAILED";
//  //  case CURAND_STATUS_TYPE_ERROR:
//  //    return "CURAND_STATUS_TYPE_ERROR";
//  //  case CURAND_STATUS_OUT_OF_RANGE:
//  //    return "CURAND_STATUS_OUT_OF_RANGE";
//  //  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
//  //    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
//  //  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
//  //    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
//  //  case CURAND_STATUS_LAUNCH_FAILURE:
//  //    return "CURAND_STATUS_LAUNCH_FAILURE";
//  //  case CURAND_STATUS_PREEXISTING_FAILURE:
//  //    return "CURAND_STATUS_PREEXISTING_FAILURE";
//  //  case CURAND_STATUS_INITIALIZATION_FAILED:
//  //    return "CURAND_STATUS_INITIALIZATION_FAILED";
//  //  case CURAND_STATUS_ARCH_MISMATCH:
//  //    return "CURAND_STATUS_ARCH_MISMATCH";
//  //  case CURAND_STATUS_INTERNAL_ERROR:
//  //    return "CURAND_STATUS_INTERNAL_ERROR";
//  //  default:
//  //    return "unknown cuRAND error";
//  //  }
//  //}
//
//  // curandGeneratorNormalX
//  inline static 
//  curandStatus_t
//  curandGenerateNormalX(curandGenerator_t generator,
//      float * outputPtr, size_t n,
//      float mean, float stddev) {
//    return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
//  }
//  inline static
//  curandStatus_t
//  curandGenerateNormalX(curandGenerator_t generator,
//      double * outputPtr, size_t n,
//      double mean, double stddev) {
//    return curandGenerateNormalDouble(generator, outputPtr,
//              n, mean, stddev);
//  }
//
//}
//#endif

namespace nvgraph {

  namespace {

    // =========================================================
    // Helper functions
    // =========================================================

    /// Perform Lanczos iteration
    /** Lanczos iteration is performed on a shifted matrix A+shift*I.
     *
     *  @param A Matrix.
     *  @param iter Pointer to current Lanczos iteration. On exit, the
     *    variable is set equal to the final Lanczos iteration.
     *  @param maxIter Maximum Lanczos iteration. This function will
     *    perform a maximum of maxIter-*iter iterations.
     *  @param shift Matrix shift.
     *  @param tol Convergence tolerance. Lanczos iteration will
     *    terminate when the residual norm (i.e. entry in beta_host) is
     *    less than tol.
     *  @param reorthogonalize Whether to reorthogonalize Lanczos
     *    vectors.
     *  @param alpha_host (Output, host memory, maxIter entries)
     *    Diagonal entries of Lanczos system.
     *  @param beta_host (Output, host memory, maxIter entries)
     *    Off-diagonal entries of Lanczos system.
     *  @param lanczosVecs_dev (Input/output, device memory,
     *    n*(maxIter+1) entries) Lanczos vectors. Vectors are stored as
     *    columns of a column-major matrix with dimensions
     *    n x (maxIter+1).
     *  @param work_dev (Output, device memory, maxIter entries)
     *    Workspace. Not needed if full reorthogonalization is disabled.
     *  @return Zero if successful. Otherwise non-zero.
     */
    template <typename IndexType_, typename ValueType_> static
    int performLanczosIteration(const Matrix<IndexType_, ValueType_> * A,
        IndexType_ * iter,
        IndexType_ maxIter,
        ValueType_ shift,
        ValueType_ tol,
        bool reorthogonalize,
        ValueType_ * __restrict__ alpha_host,
        ValueType_ * __restrict__ beta_host,
        ValueType_ * __restrict__ lanczosVecs_dev,
        ValueType_ * __restrict__ work_dev) {

      // -------------------------------------------------------
      // Variable declaration
      // -------------------------------------------------------

      // Useful variables
      const ValueType_ one    = 1;
      const ValueType_ negOne = -1;
      const ValueType_ zero   = 0;

      IndexType_ n = A->n;

      // -------------------------------------------------------
      // Compute second Lanczos vector
      // -------------------------------------------------------
      if(*iter<=0) {
  *iter = 1;

  // Apply matrix
  if(shift != 0)
    CHECK_CUDA(cudaMemcpyAsync(lanczosVecs_dev+n, lanczosVecs_dev,
             n*sizeof(ValueType_),
             cudaMemcpyDeviceToDevice));
  A->mv(1, lanczosVecs_dev, shift, lanczosVecs_dev+n);

  // Orthogonalize Lanczos vector
  Cublas::dot(n,
        lanczosVecs_dev, 1,
        lanczosVecs_dev+IDX(0,1,n), 1,
        alpha_host);
  Cublas::axpy(n, -alpha_host[0],
         lanczosVecs_dev, 1,
         lanczosVecs_dev+IDX(0,1,n), 1);
  beta_host[0] = Cublas::nrm2(n, lanczosVecs_dev+IDX(0,1,n), 1);

  // Check if Lanczos has converged
  if(beta_host[0] <= tol)
    return 0;

  // Normalize Lanczos vector
  Cublas::scal(n, 1/beta_host[0], lanczosVecs_dev+IDX(0,1,n), 1);

      }

      // -------------------------------------------------------
      // Compute remaining Lanczos vectors
      // -------------------------------------------------------

      while(*iter<maxIter) {
  ++(*iter);
    
  // Apply matrix
  if(shift != 0)
    CHECK_CUDA(cudaMemcpyAsync(lanczosVecs_dev+(*iter)*n,
             lanczosVecs_dev+(*iter-1)*n,
             n*sizeof(ValueType_),
             cudaMemcpyDeviceToDevice));
  A->mv(1, lanczosVecs_dev+IDX(0,*iter-1,n),
       shift, lanczosVecs_dev+IDX(0,*iter,n));

  // Full reorthogonalization
  //   "Twice is enough" algorithm per Kahan and Parlett
  if(reorthogonalize) {
    Cublas::gemv(true, n, *iter,
           &one, lanczosVecs_dev, n,
           lanczosVecs_dev+IDX(0,*iter,n), 1,
           &zero, work_dev, 1);
    Cublas::gemv(false, n, *iter,
           &negOne, lanczosVecs_dev, n, work_dev, 1,
           &one, lanczosVecs_dev+IDX(0,*iter,n), 1);
    CHECK_CUDA(cudaMemcpyAsync(alpha_host+(*iter-1), work_dev+(*iter-1), 
             sizeof(ValueType_), cudaMemcpyDeviceToHost));
    Cublas::gemv(true, n, *iter,
           &one, lanczosVecs_dev, n,
           lanczosVecs_dev+IDX(0,*iter,n), 1,
           &zero, work_dev, 1);
    Cublas::gemv(false, n, *iter,
           &negOne, lanczosVecs_dev, n, work_dev, 1,
           &one, lanczosVecs_dev+IDX(0,*iter,n), 1);
  }


  // Orthogonalization with 3-term recurrence relation
  else {
    Cublas::dot(n, lanczosVecs_dev+IDX(0,*iter-1,n), 1,
          lanczosVecs_dev+IDX(0,*iter,n), 1,
          alpha_host+(*iter-1));
    Cublas::axpy(n, -alpha_host[*iter-1],
           lanczosVecs_dev+IDX(0,*iter-1,n), 1,
           lanczosVecs_dev+IDX(0,*iter,n), 1);
    Cublas::axpy(n, -beta_host[*iter-2],
           lanczosVecs_dev+IDX(0,*iter-2,n), 1,
           lanczosVecs_dev+IDX(0,*iter,n), 1);
  }

  // Compute residual
  beta_host[*iter-1] = Cublas::nrm2(n, lanczosVecs_dev+IDX(0,*iter,n), 1);

  // Check if Lanczos has converged
  if(beta_host[*iter-1] <= tol)
    break;
  // Normalize Lanczos vector
  Cublas::scal(n, 1/beta_host[*iter-1],
         lanczosVecs_dev+IDX(0,*iter,n), 1);

      }

      CHECK_CUDA(cudaDeviceSynchronize());
      
      return 0;

    }

    /// Find Householder transform for 3-dimensional system
    /** Given an input vector v=[x,y,z]', this function finds a
     *  Householder transform P such that P*v is a multiple of
     *  e_1=[1,0,0]'. The input vector v is overwritten with the
     *  Householder vector such that P=I-2*v*v'.
     *
     *  @param v (Input/output, host memory, 3 entries) Input
     *    3-dimensional vector. On exit, the vector is set to the
     *    Householder vector.
     *  @param Pv (Output, host memory, 1 entry) First entry of P*v
     *    (here v is the input vector). Either equal to ||v||_2 or
     *    -||v||_2.
     *  @param P (Output, host memory, 9 entries) Householder transform
     *    matrix. Matrix dimensions are 3 x 3.
     */
    template <typename IndexType_, typename ValueType_> static
    void findHouseholder3(ValueType_ * v, ValueType_ * Pv,
        ValueType_ * P) {
  
      // Compute norm of vector
      *Pv = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);

      // Choose whether to reflect to e_1 or -e_1
      //   This choice avoids catastrophic cancellation
      if(v[0] >= 0)
  *Pv = -(*Pv);
      v[0] -= *Pv;

      // Normalize Householder vector
      ValueType_ normHouseholder = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      if(normHouseholder != 0) {
  v[0] /= normHouseholder;
  v[1] /= normHouseholder;
  v[2] /= normHouseholder;
      }
      else {
  v[0] = 0;
  v[1] = 0;
  v[2] = 0;
      }

      // Construct Householder matrix
      IndexType_ i, j;
      for(j=0; j<3; ++j)
  for(i=0; i<3; ++i)
    P[IDX(i,j,3)] = -2*v[i]*v[j];
      for(i=0; i<3; ++i)
  P[IDX(i,i,3)] += 1;

    }

    /// Apply 3-dimensional Householder transform to 4 x 4 matrix
    /** The Householder transform is pre-applied to the top three rows
     *  of the matrix and post-applied to the left three columns. The
     *  4 x 4 matrix is intended to contain the bulge that is produced
     *  in the Francis QR algorithm.
     *
     *  @param v (Input, host memory, 3 entries) Householder vector.
     *  @param A (Input/output, host memory, 16 entries) 4 x 4 matrix.
     */
    template <typename IndexType_, typename ValueType_> static
    void applyHouseholder3(const ValueType_ * v, ValueType_ * A) {

      // Loop indices
      IndexType_ i, j;
      // Dot product between Householder vector and matrix row/column
      ValueType_ vDotA;

      // Pre-apply Householder transform
      for(j=0; j<4; ++j) {
  vDotA = 0;
  for(i=0; i<3; ++i)
    vDotA += v[i]*A[IDX(i,j,4)];
  for(i=0; i<3; ++i)
    A[IDX(i,j,4)] -= 2*v[i]*vDotA;
      }

      // Post-apply Householder transform
      for(i=0; i<4; ++i) {
  vDotA = 0;
  for(j=0; j<3; ++j)
    vDotA += A[IDX(i,j,4)]*v[j];
  for(j=0; j<3; ++j)
    A[IDX(i,j,4)] -= 2*vDotA*v[j];
      }

    }

    /// Perform one step of Francis QR algorithm
    /** Equivalent to two steps of the classical QR algorithm on a
     *  tridiagonal matrix.
     *
     *  @param n Matrix dimension.
     *  @param shift1 QR algorithm shift.
     *  @param shift2 QR algorithm shift.
     *  @param alpha (Input/output, host memory, n entries) Diagonal
     *    entries of tridiagonal matrix.
     *  @param beta (Input/output, host memory, n-1 entries)
     *    Off-diagonal entries of tridiagonal matrix.
     *  @param V (Input/output, host memory, n*n entries) Orthonormal
     *    transforms from previous steps of QR algorithm. Matrix
     *    dimensions are n x n. On exit, the orthonormal transform from
     *    this Francis QR step is post-applied to the matrix.
     *  @param work (Output, host memory, 3*n entries) Workspace.
     *  @return Zero if successful. Otherwise non-zero.
     */
    template <typename IndexType_, typename ValueType_> static
    int francisQRIteration(IndexType_ n,
         ValueType_ shift1, ValueType_ shift2,
         ValueType_ * alpha, ValueType_ * beta,
         ValueType_ * V, ValueType_ * work) {

      // -------------------------------------------------------
      // Variable declaration
      // -------------------------------------------------------

      // Temporary storage of 4x4 bulge and Householder vector
      ValueType_ bulge[16];

      // Householder vector
      ValueType_ householder[3];
      // Householder matrix
      ValueType_ householderMatrix[3*3];

      // Shifts are roots of the polynomial p(x)=x^2+b*x+c
      ValueType_ b = -shift1 - shift2;
      ValueType_ c = shift1*shift2;

      // Loop indices
      IndexType_ i, j, pos;
      // Temporary variable
      ValueType_ temp;

      // -------------------------------------------------------
      // Implementation
      // -------------------------------------------------------

      // Compute initial Householder transform
      householder[0] = alpha[0]*alpha[0] + beta[0]*beta[0] + b*alpha[0] + c;
      householder[1] = beta[0]*(alpha[0]+alpha[1]+b);
      householder[2] = beta[0]*beta[1];
      findHouseholder3<IndexType_,ValueType_>(householder, &temp,
                householderMatrix);

      // Apply initial Householder transform to create bulge
      memset(bulge, 0, 16*sizeof(ValueType_));
      for(i=0; i<4; ++i)
  bulge[IDX(i,i,4)] = alpha[i];
      for(i=0; i<3; ++i) {
  bulge[IDX(i+1,i,4)] = beta[i];
  bulge[IDX(i,i+1,4)] = beta[i];
      }
      applyHouseholder3<IndexType_,ValueType_>(householder, bulge);
      Lapack<ValueType_>::gemm(false, false, n, 3, 3,
             1, V, n, householderMatrix, 3,
             0, work, n);
      memcpy(V, work, 3*n*sizeof(ValueType_));

      // Chase bulge to bottom-right of matrix with Householder transforms
      for(pos=0; pos<n-4; ++pos) {

  // Move to next position
  alpha[pos]     = bulge[IDX(0,0,4)];
  householder[0] = bulge[IDX(1,0,4)];
  householder[1] = bulge[IDX(2,0,4)];
  householder[2] = bulge[IDX(3,0,4)];
  for(j=0; j<3; ++j)
    for(i=0; i<3; ++i)
      bulge[IDX(i,j,4)] = bulge[IDX(i+1,j+1,4)];
  bulge[IDX(3,0,4)] = 0;
  bulge[IDX(3,1,4)] = 0;
  bulge[IDX(3,2,4)] = beta[pos+3];
  bulge[IDX(0,3,4)] = 0;
  bulge[IDX(1,3,4)] = 0;
  bulge[IDX(2,3,4)] = beta[pos+3];
  bulge[IDX(3,3,4)] = alpha[pos+4];

  // Apply Householder transform
  findHouseholder3<IndexType_,ValueType_>(householder, beta+pos,
            householderMatrix);
  applyHouseholder3<IndexType_,ValueType_>(householder, bulge);
  Lapack<ValueType_>::gemm(false, false, n, 3, 3,
         1, V+IDX(0,pos+1,n), n,
         householderMatrix, 3,
         0, work, n);
  memcpy(V+IDX(0,pos+1,n), work, 3*n*sizeof(ValueType_));

      }

      // Apply penultimate Householder transform
      //   Values in the last row and column are zero
      alpha[n-4]     = bulge[IDX(0,0,4)];
      householder[0] = bulge[IDX(1,0,4)];
      householder[1] = bulge[IDX(2,0,4)];
      householder[2] = bulge[IDX(3,0,4)];
      for(j=0; j<3; ++j)
  for(i=0; i<3; ++i)
    bulge[IDX(i,j,4)] = bulge[IDX(i+1,j+1,4)];
      bulge[IDX(3,0,4)] = 0;
      bulge[IDX(3,1,4)] = 0;
      bulge[IDX(3,2,4)] = 0;
      bulge[IDX(0,3,4)] = 0;
      bulge[IDX(1,3,4)] = 0;
      bulge[IDX(2,3,4)] = 0;
      bulge[IDX(3,3,4)] = 0;
      findHouseholder3<IndexType_,ValueType_>(householder, beta+n-4,
                householderMatrix);
      applyHouseholder3<IndexType_,ValueType_>(householder, bulge);
      Lapack<ValueType_>::gemm(false, false, n, 3, 3,
             1, V+IDX(0,n-3,n), n,
             householderMatrix, 3,
             0, work, n);
      memcpy(V+IDX(0,n-3,n), work, 3*n*sizeof(ValueType_));

      // Apply final Householder transform
      //   Values in the last two rows and columns are zero
      alpha[n-3]     = bulge[IDX(0,0,4)];
      householder[0] = bulge[IDX(1,0,4)];
      householder[1] = bulge[IDX(2,0,4)];
      householder[2] = 0;
      for(j=0; j<3; ++j)
  for(i=0; i<3; ++i)
    bulge[IDX(i,j,4)] = bulge[IDX(i+1,j+1,4)];
      findHouseholder3<IndexType_,ValueType_>(householder, beta+n-3,
                householderMatrix);
      applyHouseholder3<IndexType_,ValueType_>(householder, bulge);
      Lapack<ValueType_>::gemm(false, false, n, 2, 2,
             1, V+IDX(0,n-2,n), n,
             householderMatrix, 3,
             0, work, n);
      memcpy(V+IDX(0,n-2,n), work, 2*n*sizeof(ValueType_));

      // Bulge has been eliminated
      alpha[n-2] = bulge[IDX(0,0,4)];
      alpha[n-1] = bulge[IDX(1,1,4)];
      beta[n-2]  = bulge[IDX(1,0,4)]; 

      return 0;

    }

    /// Perform implicit restart of Lanczos algorithm
    /** Shifts are Chebyshev nodes of unwanted region of matrix spectrum.
     *
     *  @param n Matrix dimension.
     *  @param iter Current Lanczos iteration.
     *  @param iter_new Lanczos iteration after restart.
     *  @param shiftUpper Pointer to upper bound for unwanted
     *    region. Value is ignored if less than *shiftLower. If a
     *    stronger upper bound has been found, the value is updated on
     *    exit.
     *  @param shiftLower Pointer to lower bound for unwanted
     *    region. Value is ignored if greater than *shiftUpper. If a
     *    stronger lower bound has been found, the value is updated on
     *    exit.
     *  @param alpha_host (Input/output, host memory, iter entries)
     *    Diagonal entries of Lanczos system.
     *  @param beta_host (Input/output, host memory, iter entries)
     *    Off-diagonal entries of Lanczos system.
     *  @param V_host (Output, host memory, iter*iter entries)
     *    Orthonormal transform used to obtain restarted system. Matrix
     *    dimensions are iter x iter.
     *  @param work_host (Output, host memory, 4*iter entries)
     *    Workspace.
     *  @param lanczosVecs_dev (Input/output, device memory, n*(iter+1)
     *    entries) Lanczos vectors. Vectors are stored as columns of a
     *    column-major matrix with dimensions n x (iter+1).
     *  @param work_dev (Output, device memory, (n+iter)*iter entries)
     *    Workspace.
     */
    template <typename IndexType_, typename ValueType_> static
    int lanczosRestart(IndexType_ n,
           IndexType_ iter,
           IndexType_ iter_new,
           ValueType_ * shiftUpper,
           ValueType_ * shiftLower,
           ValueType_ * __restrict__ alpha_host,
           ValueType_ * __restrict__ beta_host,
           ValueType_ * __restrict__ V_host,
           ValueType_ * __restrict__ work_host,
           ValueType_ * __restrict__ lanczosVecs_dev,
           ValueType_ * __restrict__ work_dev,
           bool smallest_eig) {

      // -------------------------------------------------------
      // Variable declaration
      // -------------------------------------------------------

      // Useful constants
      const ValueType_ zero   = 0;
      const ValueType_ one    = 1;

      // Loop index
      IndexType_ i;

      // Number of implicit restart steps
      //   Assumed to be even since each call to Francis algorithm is
      //   equivalent to two calls of QR algorithm
      IndexType_ restartSteps = iter - iter_new;
 
      // Ritz values from Lanczos method
      ValueType_ * ritzVals_host = work_host + 3*iter;
      // Shifts for implicit restart
      ValueType_ * shifts_host;

      // Orthonormal matrix for similarity transform
      ValueType_ * V_dev = work_dev + n*iter;

      // -------------------------------------------------------
      // Implementation
      // -------------------------------------------------------

      // Compute Ritz values
      memcpy(ritzVals_host, alpha_host, iter*sizeof(ValueType_));
      memcpy(work_host, beta_host, (iter-1)*sizeof(ValueType_));
      Lapack<ValueType_>::sterf(iter, ritzVals_host, work_host);

      // Debug: Print largest eigenvalues
      //for (int i = iter-iter_new; i < iter; ++i)
      //  std::cout <<*(ritzVals_host+i)<< " ";
      //std::cout <<std::endl;

      // Initialize similarity transform with identity matrix
      memset(V_host, 0, iter*iter*sizeof(ValueType_));
      for(i=0; i<iter; ++i)
          V_host[IDX(i,i,iter)] = 1;

      // Determine interval to suppress eigenvalues
      if (smallest_eig) {
          if(*shiftLower > *shiftUpper) {
              *shiftUpper = ritzVals_host[iter-1];
              *shiftLower = ritzVals_host[iter_new];
          }
          else {
              *shiftUpper = max(*shiftUpper, ritzVals_host[iter-1]);
              *shiftLower = min(*shiftLower, ritzVals_host[iter_new]);
          }
      }
      else {
          if(*shiftLower > *shiftUpper) {
              *shiftUpper = ritzVals_host[iter-iter_new-1];
              *shiftLower = ritzVals_host[0];
          }
          else {
              *shiftUpper = max(*shiftUpper, ritzVals_host[iter-iter_new-1]);
              *shiftLower = min(*shiftLower, ritzVals_host[0]);
          }
      }

      // Calculate Chebyshev nodes as shifts
      shifts_host = ritzVals_host;
      for(i=0; i<restartSteps; ++i) {
          shifts_host[i] = cos((i+0.5)*static_cast<ValueType_>(M_PI)/restartSteps);
          shifts_host[i] *= 0.5*((*shiftUpper)-(*shiftLower));
          shifts_host[i] += 0.5*((*shiftUpper)+(*shiftLower));
      }
    
      // Apply Francis QR algorithm to implicitly restart Lanczos
      for(i=0; i<restartSteps; i+=2)
       if(francisQRIteration(iter,
          shifts_host[i], shifts_host[i+1],
          alpha_host, beta_host,
          V_host, work_host))
            WARNING("error in implicitly shifted QR algorithm");

      // Obtain new residual
      CHECK_CUDA(cudaMemcpyAsync(V_dev, V_host,
         iter*iter*sizeof(ValueType_),
         cudaMemcpyHostToDevice));

          beta_host[iter-1]
              = beta_host[iter-1]*V_host[IDX(iter-1,iter_new-1,iter)];
          Cublas::gemv(false, n, iter, beta_host+iter_new-1,
                       lanczosVecs_dev, n, V_dev+IDX(0,iter_new,iter), 1,
                       beta_host+iter-1, lanczosVecs_dev+IDX(0,iter,n), 1);
      
      // Obtain new Lanczos vectors
          Cublas::gemm(false, false, n, iter_new, iter,
                       &one, lanczosVecs_dev, n, V_dev, iter,
                       &zero, work_dev, n);
      
      CHECK_CUDA(cudaMemcpyAsync(lanczosVecs_dev, work_dev,
         n*iter_new*sizeof(ValueType_),
         cudaMemcpyDeviceToDevice));

      // Normalize residual to obtain new Lanczos vector
      CHECK_CUDA(cudaMemcpyAsync(lanczosVecs_dev+IDX(0,iter_new,n),
         lanczosVecs_dev+IDX(0,iter,n),
         n*sizeof(ValueType_),
         cudaMemcpyDeviceToDevice));
      beta_host[iter_new-1]
  = Cublas::nrm2(n, lanczosVecs_dev+IDX(0,iter_new,n), 1);
      Cublas::scal(n, 1/beta_host[iter_new-1],
       lanczosVecs_dev+IDX(0,iter_new,n), 1);

      return 0;

    }

  }

  // =========================================================
  // Eigensolver
  // =========================================================

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
   *  @param A Matrix.
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
   *  @param effIter On exit, pointer to final size of Lanczos system.
   *  @param totalIter On exit, pointer to total number of Lanczos
   *    iterations performed. Does not include Lanczos steps used to
   *    estimate largest eigenvalue.
   *  @param shift On exit, pointer to matrix shift (estimate for
   *    largest eigenvalue).
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
           IndexType_ * effIter,
           IndexType_ * totalIter,
           ValueType_ * shift,
           ValueType_ * __restrict__ alpha_host,
           ValueType_ * __restrict__ beta_host,
           ValueType_ * __restrict__ lanczosVecs_dev,
           ValueType_ * __restrict__ work_dev,
           ValueType_ * __restrict__ eigVals_dev,
           ValueType_ * __restrict__ eigVecs_dev) {

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Useful constants
    const ValueType_ one  = 1;
    const ValueType_ zero = 0;

    // Matrix dimension
    IndexType_ n = A->n;

    // Shift for implicit restart
    ValueType_ shiftUpper;
    ValueType_ shiftLower;

    // Lanczos iteration counters
    IndexType_ maxIter_curr = restartIter;  // Maximum size of Lanczos system

    // Status flags
    int status;

    // Loop index
    IndexType_ i;

    // Host memory
    ValueType_ * Z_host;           // Eigenvectors in Lanczos basis
    ValueType_ * work_host;        // Workspace


    // -------------------------------------------------------
    // Check that LAPACK is enabled
    // -------------------------------------------------------
    //Lapack<ValueType_>::check_lapack_enabled();

    // -------------------------------------------------------
    // Check that parameters are valid
    // -------------------------------------------------------
    if(A->m != A->n) {
      WARNING("invalid parameter (matrix is not square)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < 1) {
      WARNING("invalid parameter (restartIter<4)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs > n) {
      WARNING("invalid parameters (nEigVecs>n)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter < nEigVecs) {
      WARNING("invalid parameters (maxIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < nEigVecs) {
      WARNING("invalid parameters (restartIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // -------------------------------------------------------
    // Variable initialization
    // -------------------------------------------------------

    // Total number of Lanczos iterations
    *totalIter = 0;

    // Allocate host memory
    Z_host = (ValueType_*) malloc(restartIter*restartIter *sizeof(ValueType_));
    if(Z_host==NULL) WARNING("could not allocate host memory");
    work_host = (ValueType_*) malloc(4*restartIter*sizeof(ValueType_));
    if(work_host==NULL) WARNING("could not allocate host memory");

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();


    // -------------------------------------------------------
    // Compute largest eigenvalue to determine shift
    // -------------------------------------------------------

   
    #ifdef USE_CURAND
      // Random number generator
      curandGenerator_t randGen;
      // Initialize random number generator
      CHECK_CURAND(curandCreateGenerator(&randGen,
                 CURAND_RNG_PSEUDO_PHILOX4_32_10));
      CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen,
                  123456/*time(NULL)*/));
      // Initialize initial Lanczos vector
      CHECK_CURAND(curandGenerateNormalX(randGen, lanczosVecs_dev, n+n%2, zero, one));
      ValueType_ normQ1 = Cublas::nrm2(n, lanczosVecs_dev, 1);
      Cublas::scal(n, 1/normQ1, lanczosVecs_dev, 1);
    #else
        fill_raw_vec (lanczosVecs_dev, n, (ValueType_)1.0/n); // doesn't work
    #endif


    // Estimate number of Lanczos iterations 
    //   See bounds in Kuczynski and Wozniakowski (1992).
    //const ValueType_ relError = 0.25;  // Relative error
    //const ValueType_ failProb = 1e-4;  // Probability of failure
    //maxIter_curr = log(n/pow(failProb,2))/(4*std::sqrt(relError)) + 1;
    //maxIter_curr = min(maxIter_curr, restartIter);

    // Obtain tridiagonal matrix with Lanczos
    *effIter  = 0;
    *shift = 0;
    status =
      performLanczosIteration<IndexType_, ValueType_>
      (A, effIter, maxIter_curr, *shift, 0.0, reorthogonalize, 
       alpha_host, beta_host, lanczosVecs_dev, work_dev);
    if(status) WARNING("error in Lanczos iteration");

    // Determine largest eigenvalue

    Lapack<ValueType_>::sterf(*effIter, alpha_host, beta_host);
    *shift = -alpha_host[*effIter-1];
    //std::cout <<  *shift <<std::endl;
    // -------------------------------------------------------
    // Compute eigenvectors of shifted matrix
    // -------------------------------------------------------

    // Obtain tridiagonal matrix with Lanczos
    *effIter = 0;
    //maxIter_curr = min(maxIter, restartIter);
    status =
      performLanczosIteration<IndexType_, ValueType_>
      (A, effIter, maxIter_curr, *shift, 0, reorthogonalize,
       alpha_host, beta_host, lanczosVecs_dev, work_dev);
    if(status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter;

    // Apply Lanczos method until convergence
    shiftLower = 1;
    shiftUpper = -1;
    while(*totalIter<maxIter && beta_host[*effIter-1]>tol*shiftLower) {

      // Determine number of restart steps
      // Number of steps must be even due to Francis algorithm
      IndexType_ iter_new = nEigVecs+1;
      if(restartIter-(maxIter-*totalIter) > nEigVecs+1)
  iter_new = restartIter-(maxIter-*totalIter);
      if((restartIter-iter_new) % 2)
  iter_new -= 1;
      if(iter_new==*effIter)
  break;
      
      // Implicit restart of Lanczos method
      status = 
  lanczosRestart<IndexType_, ValueType_>
  (n, *effIter, iter_new,
   &shiftUpper, &shiftLower, 
   alpha_host, beta_host, Z_host, work_host,
   lanczosVecs_dev, work_dev, true);
      if(status) WARNING("error in Lanczos implicit restart");
      *effIter = iter_new;

      // Check for convergence
      if(beta_host[*effIter-1] <= tol*fabs(shiftLower))
  break;

      // Proceed with Lanczos method
      //maxIter_curr = min(restartIter, maxIter-*totalIter+*effIter);
      status = 
  performLanczosIteration<IndexType_, ValueType_>
  (A, effIter, maxIter_curr,
   *shift, tol*fabs(shiftLower), reorthogonalize,
   alpha_host, beta_host, lanczosVecs_dev, work_dev);
      if(status) WARNING("error in Lanczos iteration");
      *totalIter += *effIter-iter_new;

    }

    // Warning if Lanczos has failed to converge
    if(beta_host[*effIter-1] > tol*fabs(shiftLower))
    {
      WARNING("implicitly restarted Lanczos failed to converge");
    }

    // Solve tridiagonal system
    memcpy(work_host+2*(*effIter), alpha_host, (*effIter)*sizeof(ValueType_));
    memcpy(work_host+3*(*effIter), beta_host, (*effIter-1)*sizeof(ValueType_));
    Lapack<ValueType_>::steqr('I', *effIter,
            work_host+2*(*effIter), work_host+3*(*effIter),
            Z_host, *effIter, work_host);

    // Obtain desired eigenvalues by applying shift
    for(i=0; i<*effIter; ++i)
      work_host[i+2*(*effIter)] -= *shift;
    for(i=*effIter; i<nEigVecs; ++i)
      work_host[i+2*(*effIter)] = 0;

    // Copy results to device memory
    CHECK_CUDA(cudaMemcpy(eigVals_dev, work_host+2*(*effIter),
             nEigVecs*sizeof(ValueType_),
             cudaMemcpyHostToDevice));
    //for (int i = 0; i < nEigVecs; ++i)
    //{
    //  std::cout <<*(work_host+(2*(*effIter)+i))<< std::endl;
    //}
    CHECK_CUDA(cudaMemcpy(work_dev, Z_host,
             (*effIter)*nEigVecs*sizeof(ValueType_),
             cudaMemcpyHostToDevice));

    // Convert eigenvectors from Lanczos basis to standard basis
    Cublas::gemm(false, false, n, nEigVecs, *effIter,
     &one, lanczosVecs_dev, n, work_dev, *effIter,
     &zero, eigVecs_dev, n);

    // Clean up and exit
    free(Z_host);
    free(work_host);
    #ifdef USE_CURAND
      CHECK_CURAND(curandDestroyGenerator(randGen));
    #endif
    return NVGRAPH_OK;
  
  }

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
   *  @param A Matrix.
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
           ValueType_ * __restrict__ eigVecs_dev) {
    
    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;

    // Matrix dimension
    IndexType_ n = A.n;

    // Check that parameters are valid
    if(A.m != A.n) {
      WARNING("invalid parameter (matrix is not square)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < 1) {
      WARNING("invalid parameter (restartIter<4)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs > n) {
      WARNING("invalid parameters (nEigVecs>n)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter < nEigVecs) {
      WARNING("invalid parameters (maxIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < nEigVecs) {
      WARNING("invalid parameters (restartIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Allocate memory
    ValueType_ * alpha_host = (ValueType_*) malloc(restartIter*sizeof(ValueType_));
    ValueType_ * beta_host = (ValueType_*) malloc(restartIter*sizeof(ValueType_));
    Vector<ValueType_> lanczosVecs_dev(n*(restartIter+1), stream);
    Vector<ValueType_> work_dev((n+restartIter)*restartIter, stream);

    // Perform Lanczos method
    IndexType_ effIter;
    ValueType_ shift;
    NVGRAPH_ERROR status
      = computeSmallestEigenvectors(&A, nEigVecs, maxIter, restartIter,
            tol, reorthogonalize,
            &effIter, &iter, &shift,
            alpha_host, beta_host,
            lanczosVecs_dev.raw(), work_dev.raw(),
            eigVals_dev, eigVecs_dev);

    // Clean up and return
    free(alpha_host);
    free(beta_host);
    return status;

  }

    // =========================================================
  // Eigensolver
  // =========================================================

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
           ValueType_ * __restrict__ eigVecs_dev) {

    // -------------------------------------------------------
    // Variable declaration
    // -------------------------------------------------------

    // Useful constants
    const ValueType_ one  = 1;
    const ValueType_ zero = 0;

    // Matrix dimension
    IndexType_ n = A->n;

    // Lanczos iteration counters
    IndexType_ maxIter_curr = restartIter;  // Maximum size of Lanczos system

    // Status flags
    int status;

    // Loop index
    IndexType_ i;

    // Host memory
    ValueType_ * Z_host;           // Eigenvectors in Lanczos basis
    ValueType_ * work_host;        // Workspace


    // -------------------------------------------------------
    // Check that LAPACK is enabled
    // -------------------------------------------------------
    //Lapack<ValueType_>::check_lapack_enabled();

    // -------------------------------------------------------
    // Check that parameters are valid
    // -------------------------------------------------------
    if(A->m != A->n) {
      WARNING("invalid parameter (matrix is not square)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < 1) {
      WARNING("invalid parameter (restartIter<4)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs > n) {
      WARNING("invalid parameters (nEigVecs>n)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter < nEigVecs) {
      WARNING("invalid parameters (maxIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter <= nEigVecs) {
      WARNING("invalid parameters (restartIter<=nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // -------------------------------------------------------
    // Variable initialization
    // -------------------------------------------------------

    // Total number of Lanczos iterations
    *totalIter = 0;

    // Allocate host memory
    Z_host = (ValueType_*) malloc(restartIter*restartIter *sizeof(ValueType_));
    if(Z_host==NULL) WARNING("could not allocate host memory");
    work_host = (ValueType_*) malloc(4*restartIter*sizeof(ValueType_));
    if(work_host==NULL) WARNING("could not allocate host memory");

    // Initialize cuBLAS
    Cublas::set_pointer_mode_host();


    // -------------------------------------------------------
    // Compute largest eigenvalue 
    // -------------------------------------------------------

   
    #ifdef USE_CURAND
      // Random number generator
      curandGenerator_t randGen;
      // Initialize random number generator
      CHECK_CURAND(curandCreateGenerator(&randGen,
                 CURAND_RNG_PSEUDO_PHILOX4_32_10));
      CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(randGen,
                  123456));
       // Initialize initial Lanczos vector
      CHECK_CURAND(curandGenerateNormalX(randGen, lanczosVecs_dev, n+n%2, zero, one));
      ValueType_ normQ1 = Cublas::nrm2(n, lanczosVecs_dev, 1);
      Cublas::scal(n, 1/normQ1, lanczosVecs_dev, 1);
    #else
        fill_raw_vec (lanczosVecs_dev, n, (ValueType_)1.0/n); // doesn't work
    #endif


    // Estimate number of Lanczos iterations 
    //   See bounds in Kuczynski and Wozniakowski (1992).
    //const ValueType_ relError = 0.25;  // Relative error
    //const ValueType_ failProb = 1e-4;  // Probability of failure
    //maxIter_curr = log(n/pow(failProb,2))/(4*std::sqrt(relError)) + 1;
    //maxIter_curr = min(maxIter_curr, restartIter);

    // Obtain tridiagonal matrix with Lanczos
    *effIter  = 0;
    ValueType_ shift_val=0.0;
    ValueType_ *shift = &shift_val;
    //maxIter_curr = min(maxIter, restartIter);
    status =
      performLanczosIteration<IndexType_, ValueType_>
      (A, effIter, maxIter_curr, *shift, 0, reorthogonalize,
       alpha_host, beta_host, lanczosVecs_dev, work_dev);
    if(status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter;

    // Apply Lanczos method until convergence
    ValueType_ shiftLower = 1;
    ValueType_ shiftUpper = -1;
    while(*totalIter<maxIter && beta_host[*effIter-1]>tol*shiftLower) {

      // Determine number of restart steps
      //   Number of steps must be even due to Francis algorithm
      IndexType_ iter_new = nEigVecs+1;
      if(restartIter-(maxIter-*totalIter) > nEigVecs+1)
  iter_new = restartIter-(maxIter-*totalIter);
      if((restartIter-iter_new) % 2)
  iter_new -= 1;
      if(iter_new==*effIter)
  break;
      
      // Implicit restart of Lanczos method
      status = 
  lanczosRestart<IndexType_, ValueType_>
  (n, *effIter, iter_new,
   &shiftUpper, &shiftLower, 
   alpha_host, beta_host, Z_host, work_host,
   lanczosVecs_dev, work_dev, false);
      if(status) WARNING("error in Lanczos implicit restart");
      *effIter = iter_new;

      // Check for convergence
      if(beta_host[*effIter-1] <= tol*fabs(shiftLower))
  break;

      // Proceed with Lanczos method
      //maxIter_curr = min(restartIter, maxIter-*totalIter+*effIter);
      status = 
  performLanczosIteration<IndexType_, ValueType_>
  (A, effIter, maxIter_curr,
   *shift, tol*fabs(shiftLower), reorthogonalize,
   alpha_host, beta_host, lanczosVecs_dev, work_dev);
      if(status) WARNING("error in Lanczos iteration");
      *totalIter += *effIter-iter_new;

    }

    // Warning if Lanczos has failed to converge
    if(beta_host[*effIter-1] > tol*fabs(shiftLower))
    {
      WARNING("implicitly restarted Lanczos failed to converge");
    }
    for (int i = 0; i < restartIter; ++i)
    {
      for (int j = 0; j < restartIter; ++j)
        Z_host[i*restartIter+j] = 0;
      
    }
    // Solve tridiagonal system
    memcpy(work_host+2*(*effIter), alpha_host, (*effIter)*sizeof(ValueType_));
    memcpy(work_host+3*(*effIter), beta_host, (*effIter-1)*sizeof(ValueType_));
    Lapack<ValueType_>::steqr('I', *effIter,
            work_host+2*(*effIter), work_host+3*(*effIter),
            Z_host, *effIter, work_host);

    // note: We need to pick the top nEigVecs eigenvalues
    // but effItter can be larger than nEigVecs 
    // hence we add an offset for that case, because we want to access top nEigVecs eigenpairs in the matrix of size effIter. 
    // remember the array is sorted, so it is not needed for smallest eigenvalues case because the first ones are the smallest ones 

    IndexType_ top_eigenparis_idx_offset = *effIter - nEigVecs;

    //Debug : print nEigVecs largest eigenvalues
    //for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
    //  std::cout <<*(work_host+(2*(*effIter)+i))<< " ";
    //std::cout <<std::endl;

    //Debug : print nEigVecs largest eigenvectors
    //for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
    //{
    //  for (int j = 0; j < *effIter; ++j)
    //    std::cout <<Z_host[i*(*effIter)+j]<< " ";
    //  std::cout <<std::endl;
    //}

    // Obtain desired eigenvalues by applying shift
    for(i=0; i<*effIter; ++i)
      work_host[i+2*(*effIter)] -= *shift;
    
    for(i=0; i<top_eigenparis_idx_offset; ++i)
      work_host[i+2*(*effIter)] = 0;

    // Copy results to device memory
    // skip smallest eigenvalue if needed   
    CHECK_CUDA(cudaMemcpy(eigVals_dev, work_host+2*(*effIter)+top_eigenparis_idx_offset,
             nEigVecs*sizeof(ValueType_),
             cudaMemcpyHostToDevice));

    // skip smallest eigenvector if needed   
    CHECK_CUDA(cudaMemcpy(work_dev, Z_host+(top_eigenparis_idx_offset*(*effIter)),
             (*effIter)*nEigVecs*sizeof(ValueType_),
             cudaMemcpyHostToDevice));

    // Convert eigenvectors from Lanczos basis to standard basis
    Cublas::gemm(false, false, n, nEigVecs, *effIter,
     &one, lanczosVecs_dev, n, work_dev, *effIter,
     &zero, eigVecs_dev, n);

    // Clean up and exit
    free(Z_host);
    free(work_host);
    #ifdef USE_CURAND
      CHECK_CURAND(curandDestroyGenerator(randGen));
    #endif
    return NVGRAPH_OK;
  
  }

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
           ValueType_ * __restrict__ eigVecs_dev) {
    
    // CUDA stream
    //   TODO: handle non-zero streams
    cudaStream_t stream = 0;

    // Matrix dimension
    IndexType_ n = A.n;

    // Check that parameters are valid
    if(A.m != A.n) {
      WARNING("invalid parameter (matrix is not square)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs < 1) {
      WARNING("invalid parameter (nEigVecs<1)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < 1) {
      WARNING("invalid parameter (restartIter<4)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(tol < 0) {
      WARNING("invalid parameter (tol<0)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(nEigVecs > n) {
      WARNING("invalid parameters (nEigVecs>n)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(maxIter < nEigVecs) {
      WARNING("invalid parameters (maxIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }
    if(restartIter < nEigVecs) {
      WARNING("invalid parameters (restartIter<nEigVecs)");
      return NVGRAPH_ERR_BAD_PARAMETERS;
    }

    // Allocate memory
    ValueType_ * alpha_host = (ValueType_*) malloc(restartIter*sizeof(ValueType_));
    ValueType_ * beta_host = (ValueType_*) malloc(restartIter*sizeof(ValueType_));
    Vector<ValueType_> lanczosVecs_dev(n*(restartIter+1), stream);
    Vector<ValueType_> work_dev((n+restartIter)*restartIter, stream);

    // Perform Lanczos method
    IndexType_ effIter;
    NVGRAPH_ERROR status
      = computeLargestEigenvectors(&A, nEigVecs, maxIter, restartIter,
            tol, reorthogonalize,
            &effIter, &iter,
            alpha_host, beta_host,
            lanczosVecs_dev.raw(), work_dev.raw(),
            eigVals_dev, eigVecs_dev);

    // Clean up and return
    free(alpha_host);
    free(beta_host);
    return status;

  }

  // =========================================================
  // Explicit instantiation
  // =========================================================

  template NVGRAPH_ERROR computeSmallestEigenvectors<int,float>
  (const Matrix<int,float> * A,
   int nEigVecs, int maxIter, int restartIter, float tol,
   bool reorthogonalize,
   int * iter, int * totalIter, float * shift,
   float * __restrict__ alpha_host,
   float * __restrict__ beta_host,
   float * __restrict__ lanczosVecs_dev,
   float * __restrict__ work_dev,
   float * __restrict__ eigVals_dev,
   float * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeSmallestEigenvectors<int,double>
  (const Matrix<int,double> * A,
   int nEigVecs, int maxIter, int restartIter, double tol,
   bool reorthogonalize,
   int * iter, int * totalIter, double * shift,
   double * __restrict__ alpha_host,
   double * __restrict__ beta_host,
   double * __restrict__ lanczosVecs_dev,
   double * __restrict__ work_dev,
   double * __restrict__ eigVals_dev,
   double * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeSmallestEigenvectors<int, float>
  (const Matrix<int,float> & A,
   int nEigVecs,
   int maxIter,
   int restartIter,
   float tol,
   bool reorthogonalize,
   int & iter,
   float * __restrict__ eigVals_dev,
   float * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeSmallestEigenvectors<int, double>
  (const Matrix<int,double> & A,
   int nEigVecs,
   int maxIter,
   int restartIter,
   double tol,
   bool reorthogonalize,
   int & iter,
   double * __restrict__ eigVals_dev,
   double * __restrict__ eigVecs_dev);

  template NVGRAPH_ERROR computeLargestEigenvectors<int,float>
  (const Matrix<int,float> * A,
   int nEigVecs, int maxIter, int restartIter, float tol,
   bool reorthogonalize,
   int * iter, int * totalIter,
   float * __restrict__ alpha_host,
   float * __restrict__ beta_host,
   float * __restrict__ lanczosVecs_dev,
   float * __restrict__ work_dev,
   float * __restrict__ eigVals_dev,
   float * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeLargestEigenvectors<int,double>
  (const Matrix<int,double> * A,
   int nEigVecs, int maxIter, int restartIter, double tol,
   bool reorthogonalize,
   int * iter, int * totalIter,
   double * __restrict__ alpha_host,
   double * __restrict__ beta_host,
   double * __restrict__ lanczosVecs_dev,
   double * __restrict__ work_dev,
   double * __restrict__ eigVals_dev,
   double * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeLargestEigenvectors<int, float>
  (const Matrix<int,float> & A,
   int nEigVecs,
   int maxIter,
   int restartIter,
   float tol,
   bool reorthogonalize,
   int & iter,
   float * __restrict__ eigVals_dev,
   float * __restrict__ eigVecs_dev);
  template NVGRAPH_ERROR computeLargestEigenvectors<int, double>
  (const Matrix<int,double> & A,
   int nEigVecs,
   int maxIter,
   int restartIter,
   double tol,
   bool reorthogonalize,
   int & iter,
   double * __restrict__ eigVals_dev,
   double * __restrict__ eigVecs_dev);

}
//#endif //NVGRAPH_PARTITION

