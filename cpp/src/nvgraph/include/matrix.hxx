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

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include "nvgraph_vector.hxx"
#include "valued_csr_graph.hxx"

namespace nvgraph {

  /// Abstract matrix class
  /** Derived classes must implement matrix-vector products.
   */
  template <typename IndexType_, typename ValueType_>
  class Matrix {
  public:
    /// Number of rows
    const IndexType_ m;
    /// Number of columns
    const IndexType_ n;
    /// CUDA stream
    cudaStream_t s;  

    /// Constructor
    /** @param _m Number of rows.
     *  @param _n Number of columns.
     */
    Matrix(IndexType_ _m, IndexType_ _n) : m(_m), n(_n), s(0){}

    /// Destructor
    virtual ~Matrix() {}


    /// Get and Set CUDA stream  
    virtual void setCUDAStream(cudaStream_t _s) = 0;  
    virtual void getCUDAStream(cudaStream_t *_s) = 0;    

    /// Matrix-vector product
    /** y is overwritten with alpha*A*x+beta*y.
     *
     *  @param alpha Scalar.
     *  @param x (Input, device memory, n entries) Vector.
     *  @param beta Scalar.
     *  @param y (Input/output, device memory, m entries) Output
     *    vector.
     */
    virtual void mv(ValueType_ alpha,
		    const ValueType_ * __restrict__ x,
		    ValueType_ beta,
		    ValueType_ * __restrict__ y) const = 0;

    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const = 0;  
    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const = 0;  
    virtual void reorder(IndexType_ *p) const = 0;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M) = 0;
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const = 0; 
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const = 0;
  };

  /// Dense matrix class
  template <typename IndexType_, typename ValueType_>
  class DenseMatrix : public Matrix<IndexType_, ValueType_> {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Matrix entries, stored column-major in device memory
    const ValueType_ * A;
    /// Leading dimension of matrix entry array
    const IndexType_ lda;

  public:
    /// Constructor
    DenseMatrix(bool _trans,
		IndexType_ _m, IndexType_ _n,
		const ValueType_ * _A, IndexType_ _lda);

    /// Destructor
    virtual ~DenseMatrix();

    /// Get and Set CUDA stream  
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);     

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const; 
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Sparse matrix class in CSR format
  template <typename IndexType_, typename ValueType_>
  class CsrMatrix : public Matrix<IndexType_, ValueType_> {

  private:
    /// Whether to transpose matrix
    const bool trans;
    /// Whether matrix is stored in symmetric format
    const bool sym;
    /// Number of non-zero entries
    const IndexType_ nnz;
    /// Matrix properties
    const cusparseMatDescr_t descrA;
    /// Matrix entry values (device memory)
    /*const*/ ValueType_ * csrValA;
    /// Pointer to first entry in each row (device memory)
    const IndexType_ * csrRowPtrA;
    /// Column index of each matrix entry (device memory)
    const IndexType_ * csrColIndA;
    /// Analysis info (pointer to opaque CUSPARSE struct)  
    cusparseSolveAnalysisInfo_t info_l;
    cusparseSolveAnalysisInfo_t info_u;  
    /// factored flag (originally set to false, then reset to true after factorization), 
    /// notice we only want to factor once
    bool factored;  

  public:
    /// Constructor
    CsrMatrix(bool _trans, bool _sym,
	      IndexType_ _m, IndexType_ _n, IndexType_ _nnz,
        const cusparseMatDescr_t _descrA,
	      /*const*/ ValueType_ * _csrValA,
	      const IndexType_ * _csrRowPtrA,
	      const IndexType_ * _csrColIndA);

    /// Constructor
    CsrMatrix( ValuedCsrGraph<IndexType_,ValueType_> & G, const cusparseMatDescr_t _descrA =0);

    /// Destructor
    virtual ~CsrMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);  


    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
    /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;  

    /// Incomplete Cholesky (setup, factor and solve)
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;         

    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

  /// Graph Laplacian matrix
  template <typename IndexType_, typename ValueType_>
  class LaplacianMatrix 
    : public Matrix<IndexType_, ValueType_> {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> * A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> * M;  

  public:
    /// Constructor
    LaplacianMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A);

    /// Destructor
    virtual ~LaplacianMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);   

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
		    ValueType_ beta, ValueType_ * __restrict__ y) const;
     /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;    

    /// Solve preconditioned system M x = f for a set of k vectors 
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;    
    
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

    ///  Modularity matrix
  template <typename IndexType_, typename ValueType_>
  class ModularityMatrix 
    : public Matrix<IndexType_, ValueType_> {

  private:
    /// Adjacency matrix
    /*const*/ Matrix<IndexType_, ValueType_> * A;
    /// Degree of each vertex
    Vector<ValueType_> D;
    IndexType_ nnz;
    ValueType_ edge_sum;
    
    /// Preconditioning matrix
    Matrix<IndexType_, ValueType_> * M;  

  public:
    /// Constructor
    ModularityMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A, IndexType_ _nnz);

    /// Destructor
    virtual ~ModularityMatrix();

    /// Get and Set CUDA stream    
    virtual void setCUDAStream(cudaStream_t _s);  
    virtual void getCUDAStream(cudaStream_t *_s);   

    /// Matrix-vector product
    virtual void mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
        ValueType_ beta, ValueType_ * __restrict__ y) const;
     /// Matrix-set of k vectors product
    virtual void mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;

    /// Scale a set of k vectors by a diagonal
    virtual void dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const;  

    /// Color and Reorder
    virtual void color(IndexType_ *c, IndexType_ *p) const;  
    virtual void reorder(IndexType_ *p) const;    

    /// Solve preconditioned system M x = f for a set of k vectors 
    virtual void prec_setup(Matrix<IndexType_,ValueType_> * _M);
    virtual void prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const;    
   
    //Get the sum of all edges
    virtual ValueType_ getEdgeSum() const;
  };

// cublasIxamax
inline
cublasStatus_t cublasIxamax(cublasHandle_t handle, int n,
          const float *x, int incx, int *result) {
  return cublasIsamax(handle, n, x, incx, result);
}
inline
cublasStatus_t cublasIxamax(cublasHandle_t handle, int n,
          const double *x, int incx, int *result) {
  return cublasIdamax(handle, n, x, incx, result);
}

// cublasIxamin
inline
cublasStatus_t cublasIxamin(cublasHandle_t handle, int n,
          const float *x, int incx, int *result) {
  return cublasIsamin(handle, n, x, incx, result);
}
inline
cublasStatus_t cublasIxamin(cublasHandle_t handle, int n,
          const double *x, int incx, int *result) {
  return cublasIdamin(handle, n, x, incx, result);
}

// cublasXasum
inline
cublasStatus_t cublasXasum(cublasHandle_t handle, int n,
         const float *x, int incx,
         float  *result) {
  return cublasSasum(handle, n, x, incx, result);
}
inline
cublasStatus_t cublasXasum(cublasHandle_t handle, int n,
         const double *x, int incx,
         double  *result) {
  return cublasDasum(handle, n, x, incx, result);
}

// cublasXaxpy
inline
cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                           const float * alpha,
                           const float * x, int incx,
                           float * y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
inline
cublasStatus_t cublasXaxpy(cublasHandle_t handle, int n,
                           const double *alpha,
                           const double *x, int incx,
                           double *y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

// cublasXcopy
inline
cublasStatus_t cublasXcopy(cublasHandle_t handle, int n,
                           const float *x, int incx,
                           float *y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}
inline
cublasStatus_t cublasXcopy(cublasHandle_t handle, int n,
                           const double *x, int incx,
                           double *y, int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}

// cublasXdot
inline
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
        const float *x, int incx,
        const float *y, int incy,
        float *result) {
  return cublasSdot(handle, n, x, incx, y, incy, result);
}
inline
cublasStatus_t cublasXdot(cublasHandle_t handle, int n,
        const double *x, int incx,
        const double *y, int incy,
        double *result) {
  return cublasDdot(handle, n, x, incx, y, incy, result);
}

// cublasXnrm2
inline
cublasStatus_t cublasXnrm2(cublasHandle_t handle, int n,
         const float *x, int incx,
         float  *result) {
  return cublasSnrm2(handle, n, x, incx, result);
}
inline
cublasStatus_t cublasXnrm2(cublasHandle_t handle, int n,
         const double *x, int incx,
         double  *result) {
  return cublasDnrm2(handle, n, x, incx, result);
}

// cublasXscal
inline
cublasStatus_t cublasXscal(cublasHandle_t handle, int n,
         const float *alpha,
         float *x, int incx) {
  return cublasSscal(handle, n, alpha, x, incx);
}
inline
cublasStatus_t cublasXscal(cublasHandle_t handle, int n,
         const double *alpha,
         double *x, int incx) {
  return cublasDscal(handle, n, alpha, x, incx);
}

// cublasXgemv
inline
cublasStatus_t cublasXgemv(cublasHandle_t handle,
         cublasOperation_t trans,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           const float *x, int incx,
                           const float *beta,
                           float *y, int incy) {
  return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx,
         beta, y, incy);
}
inline
cublasStatus_t cublasXgemv(cublasHandle_t handle,
         cublasOperation_t trans,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                           const double *x, int incx,
                           const double *beta,
                           double *y, int incy) {
  return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx,
         beta, y, incy);
}

// cublasXger
inline
cublasStatus_t cublasXger(cublasHandle_t handle, int m, int n,
        const float *alpha,
        const float *x, int incx,
        const float *y, int incy,
        float *A, int lda) {
  return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
inline
cublasStatus_t cublasXger(cublasHandle_t handle, int m, int n,
        const double *alpha,
        const double *x, int incx,
        const double *y, int incy,
        double *A, int lda) {
  return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

// cublasXgemm
inline
cublasStatus_t cublasXgemm(cublasHandle_t handle,
         cublasOperation_t transa,
         cublasOperation_t transb,
         int m, int n, int k,
         const float *alpha,
         const float *A, int lda,
         const float *B, int ldb,
         const float *beta,
         float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k,
         alpha, A, lda, B, ldb, beta, C, ldc);
}
inline
cublasStatus_t cublasXgemm(cublasHandle_t handle,
         cublasOperation_t transa,
         cublasOperation_t transb,
         int m, int n, int k,
         const double *alpha,
         const double *A, int lda,
         const double *B, int ldb,
         const double *beta,
         double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k,
         alpha, A, lda, B, ldb, beta, C, ldc);
}

// cublasXgeam
inline
cublasStatus_t cublasXgeam(cublasHandle_t handle,
         cublasOperation_t transa,
         cublasOperation_t transb,
         int m, int n,
         const float *alpha,
         const float *A, int lda,
         const float *beta,
         const float *B, int ldb,
         float *C, int ldc) {
  return cublasSgeam(handle, transa, transb, m, n,
         alpha, A, lda, beta, B, ldb, C, ldc);
}
inline
cublasStatus_t cublasXgeam(cublasHandle_t handle,
         cublasOperation_t transa,
         cublasOperation_t transb,
         int m, int n,
         const double *alpha,
         const double *A, int lda,
         const double *beta,
         const double *B, int ldb,
         double *C, int ldc) {
  return cublasDgeam(handle, transa, transb, m, n,
         alpha, A, lda, beta, B, ldb, C, ldc);
}

// cublasXtrsm
inline cublasStatus_t cublasXtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb) {
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); 
}
inline cublasStatus_t cublasXtrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb) {
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); 
}

// curandGeneratorNormalX
inline 
curandStatus_t
curandGenerateNormalX(curandGenerator_t generator,
          float * outputPtr, size_t n,
          float mean, float stddev) {
  return curandGenerateNormal(generator, outputPtr, n, mean, stddev);
}
inline
curandStatus_t
curandGenerateNormalX(curandGenerator_t generator,
          double * outputPtr, size_t n,
          double mean, double stddev) {
  return curandGenerateNormalDouble(generator, outputPtr,
            n, mean, stddev);
}

// cusolverXpotrf_bufferSize
inline cusolverStatus_t cusolverXpotrf_bufferSize(cusolverDnHandle_t handle, int n, float *A, int lda, int *Lwork){
    return cusolverDnSpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,A,lda,Lwork);
}
inline cusolverStatus_t cusolverXpotrf_bufferSize(cusolverDnHandle_t handle, int n, double *A, int lda, int *Lwork){
    return cusolverDnDpotrf_bufferSize(handle,CUBLAS_FILL_MODE_LOWER,n,A,lda,Lwork);
}

// cusolverXpotrf
inline cusolverStatus_t cusolverXpotrf(cusolverDnHandle_t handle, int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo){
    return cusolverDnSpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,A,lda,Workspace,Lwork,devInfo);
}
inline cusolverStatus_t cusolverXpotrf(cusolverDnHandle_t handle, int n, double *A, int lda, double *Workspace, int Lwork, int *devInfo){
    return cusolverDnDpotrf(handle,CUBLAS_FILL_MODE_LOWER,n,A,lda,Workspace,Lwork,devInfo);
}

// cusolverXgesvd_bufferSize
inline cusolverStatus_t cusolverXgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *U, int ldu, float *VT, int ldvt, int *Lwork){
    //ideally
    //char jobu = 'O';
    //char jobvt= 'N';
    //only supported
    //char jobu = 'A';
    //char jobvt= 'A';
    return cusolverDnSgesvd_bufferSize(handle,m,n,Lwork);
}

inline cusolverStatus_t cusolverXgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *U, int ldu, double *VT, int ldvt, int *Lwork){
    //ideally
    //char jobu = 'O';
    //char jobvt= 'N';
    //only supported
    //char jobu = 'A';
    //char jobvt= 'A';
    return cusolverDnDgesvd_bufferSize(handle,m,n,Lwork);
}

// cusolverXgesvd
inline cusolverStatus_t cusolverXgesvd(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int  *devInfo){
    //ideally
    //char jobu = 'O';
    //char jobvt= 'N';
    //only supported
    char jobu = 'A';
    char jobvt= 'A';

    return cusolverDnSgesvd(handle,jobu,jobvt,m,n,A,lda,S,U,ldu,VT,ldvt,Work,Lwork,rwork,devInfo);
} 

inline cusolverStatus_t cusolverXgesvd(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int  *devInfo){
    //ideally
    //char jobu = 'O';
    //char jobvt= 'N';
    //only supported
    char jobu = 'A';
    char jobvt= 'A';
    return cusolverDnDgesvd(handle,jobu,jobvt,m,n,A,lda,S,U,ldu,VT,ldvt,Work,Lwork,rwork,devInfo);
} 

// cusolverXgesvd_cond
inline cusolverStatus_t cusolverXgesvd_cond(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *S, float *U, int ldu, float *VT, int ldvt, float *Work, int Lwork, float *rwork, int  *devInfo){
    //ideally
    //char jobu = 'N';
    //char jobvt= 'N';
    //only supported
    char jobu = 'A';
    char jobvt= 'A';
    return cusolverDnSgesvd(handle,jobu,jobvt,m,n,A,lda,S,U,ldu,VT,ldvt,Work,Lwork,rwork,devInfo);
} 

inline cusolverStatus_t cusolverXgesvd_cond(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *S, double *U, int ldu, double *VT, int ldvt, double *Work, int Lwork, double *rwork, int  *devInfo){
    //ideally
    //char jobu = 'N';
    //char jobvt= 'N';
    //only supported
    char jobu = 'A';
    char jobvt= 'A';
    return cusolverDnDgesvd(handle,jobu,jobvt,m,n,A,lda,S,U,ldu,VT,ldvt,Work,Lwork,rwork,devInfo);
} 

// cusparseXcsrmv
inline
cusparseStatus_t cusparseXcsrmv(cusparseHandle_t handle,
        cusparseOperation_t transA, 
        int m, int n, int nnz,
        const float * alpha, 
        const cusparseMatDescr_t descrA, 
        const float * csrValA, 
        const int * csrRowPtrA,
        const int * csrColIndA,
        const float * x,
        const float * beta, 
        float *y) {
  return cusparseScsrmv_mp(handle, transA, m, n, nnz, 
      alpha, descrA, csrValA, csrRowPtrA, csrColIndA, 
      x, beta, y);
}
inline
cusparseStatus_t cusparseXcsrmv(cusparseHandle_t handle,
        cusparseOperation_t transA, 
        int m, int n, int nnz,
        const double * alpha, 
        const cusparseMatDescr_t descrA, 
        const double * csrValA, 
        const int * csrRowPtrA,
        const int * csrColIndA,
        const double * x,
        const double * beta, 
        double *y) {
  return cusparseDcsrmv_mp(handle, transA, m, n, nnz,
        alpha, descrA, csrValA, csrRowPtrA, csrColIndA,
        x, beta, y);
}

// cusparseXcsrmm
inline
cusparseStatus_t cusparseXcsrmm(cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        int m, int n, int k, int nnz, 
        const float *alpha, 
        const cusparseMatDescr_t descrA, 
        const float *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const float *B, int ldb,
        const float *beta, 
        float *C, int ldc) {
  return cusparseScsrmm(handle, transA, m, n, k, nnz,
      alpha, descrA, csrValA,
      csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}
inline
cusparseStatus_t cusparseXcsrmm(cusparseHandle_t handle, 
        cusparseOperation_t transA, 
        int m, int n, int k, int nnz, 
        const double *alpha, 
        const cusparseMatDescr_t descrA, 
        const double *csrValA, 
        const int *csrRowPtrA, 
        const int *csrColIndA,
        const double *B, int ldb,
        const double *beta, 
        double *C, int ldc) {
  return cusparseDcsrmm(handle, transA, m, n, k, nnz,
      alpha, descrA, csrValA,
      csrRowPtrA, csrColIndA,
      B, ldb, beta, C, ldc);
}

// cusparseXcsrgeam
inline
cusparseStatus_t cusparseXcsrgeam(cusparseHandle_t handle, 
          int m, int n,
          const float *alpha,
          const cusparseMatDescr_t descrA, 
          int nnzA, const float *csrValA, 
          const int *csrRowPtrA, 
          const int *csrColIndA,
          const float *beta,
          const cusparseMatDescr_t descrB, 
          int nnzB, const float *csrValB, 
          const int *csrRowPtrB,
          const int *csrColIndB,
          const cusparseMatDescr_t descrC,
          float *csrValC, 
          int *csrRowPtrC, int *csrColIndC) {
  return cusparseScsrgeam(handle,m,n,
        alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,
        beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,
        descrC,csrValC,csrRowPtrC,csrColIndC);
}
inline
cusparseStatus_t cusparseXcsrgeam(cusparseHandle_t handle, 
          int m, int n,
          const double *alpha,
          const cusparseMatDescr_t descrA, 
          int nnzA, const double *csrValA, 
          const int *csrRowPtrA, 
          const int *csrColIndA,
          const double *beta,
          const cusparseMatDescr_t descrB, 
          int nnzB, const double *csrValB, 
          const int *csrRowPtrB,
          const int *csrColIndB,
          const cusparseMatDescr_t descrC,
          double *csrValC, 
          int *csrRowPtrC, int *csrColIndC) {
  return cusparseDcsrgeam(handle,m,n,
        alpha,descrA,nnzA,csrValA,csrRowPtrA,csrColIndA,
        beta,descrB,nnzB,csrValB,csrRowPtrB,csrColIndB,
        descrC,csrValC,csrRowPtrC,csrColIndC);
}

//ILU0, incomplete-LU with 0 threshhold (CUSPARSE)
inline cusparseStatus_t cusparseXcsrilu0(cusparseHandle_t handle, 
                                         cusparseOperation_t trans, 
                                         int m, 
                                         const cusparseMatDescr_t descrA, 
                                         float *csrValM,
                                         const int *csrRowPtrA, 
                                         const int *csrColIndA,
                                         cusparseSolveAnalysisInfo_t info){
    return cusparseScsrilu0(handle,trans,m,descrA,csrValM,csrRowPtrA,csrColIndA,info);
}

inline cusparseStatus_t cusparseXcsrilu0(cusparseHandle_t handle, 
                                         cusparseOperation_t trans, 
                                         int m, 
                                         const cusparseMatDescr_t descrA, 
                                         double *csrValM, 
                                         const int *csrRowPtrA, 
                                         const int *csrColIndA, 
                                         cusparseSolveAnalysisInfo_t info){
    return cusparseDcsrilu0(handle,trans,m,descrA,csrValM,csrRowPtrA,csrColIndA,info);
}

//IC0, incomplete-Cholesky with 0 threshhold (CUSPARSE)
inline cusparseStatus_t cusparseXcsric0(cusparseHandle_t handle, 
                                        cusparseOperation_t trans, 
                                        int m, 
                                        const cusparseMatDescr_t descrA, 
                                        float *csrValM,
                                        const int *csrRowPtrA, 
                                        const int *csrColIndA,
                                        cusparseSolveAnalysisInfo_t info){
    return cusparseScsric0(handle,trans,m,descrA,csrValM,csrRowPtrA,csrColIndA,info);
}
inline cusparseStatus_t cusparseXcsric0(cusparseHandle_t handle, 
                                        cusparseOperation_t trans, 
                                        int m, 
                                        const cusparseMatDescr_t descrA, 
                                        double *csrValM, 
                                        const int *csrRowPtrA, 
                                        const int *csrColIndA, 
                                        cusparseSolveAnalysisInfo_t info){
    return cusparseDcsric0(handle,trans,m,descrA,csrValM,csrRowPtrA,csrColIndA,info);
}

//sparse triangular solve (CUSPARSE)
//analysis phase
inline cusparseStatus_t cusparseXcsrsm_analysis (cusparseHandle_t handle, cusparseOperation_t transa, int m, int nnz, const cusparseMatDescr_t descra, 
                                                   const float *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info){
    return cusparseScsrsm_analysis(handle,transa,m,nnz,descra,a,ia,ja,info);
}   
inline cusparseStatus_t cusparseXcsrsm_analysis (cusparseHandle_t handle, cusparseOperation_t transa, int m, int nnz, const cusparseMatDescr_t descra, 
                                                   const double *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info){
    return cusparseDcsrsm_analysis(handle,transa,m,nnz,descra,a,ia,ja,info);
} 
//solve phase
inline cusparseStatus_t cusparseXcsrsm_solve (cusparseHandle_t handle, cusparseOperation_t transa, int m, int k, float alpha, const cusparseMatDescr_t descra, 
                                              const float *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info, const float *x, int ldx, float *y, int ldy){
    return cusparseScsrsm_solve(handle,transa,m,k,&alpha,descra,a,ia,ja,info,x,ldx,y,ldy);
}   
inline cusparseStatus_t cusparseXcsrsm_solve (cusparseHandle_t handle, cusparseOperation_t transa, int m, int k, double alpha, const cusparseMatDescr_t descra, 
                                              const double *a, const int *ia, const int *ja, cusparseSolveAnalysisInfo_t info, const double *x, int ldx, double *y, int ldy){
    return cusparseDcsrsm_solve(handle,transa,m,k,&alpha,descra,a,ia,ja,info,x,ldx,y,ldy);
} 


inline cusparseStatus_t cusparseXcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *fractionToColor, int *ncolors, int *coloring, int *reordering,cusparseColorInfo_t info) {
    return cusparseScsrcolor(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info);
}
inline cusparseStatus_t cusparseXcsrcolor(cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA, const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *fractionToColor, int *ncolors, int *coloring, int *reordering,cusparseColorInfo_t info) {
    return cusparseDcsrcolor(handle,m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,fractionToColor,ncolors,coloring,reordering,info);
}


}

