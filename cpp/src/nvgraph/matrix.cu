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
//#ifdef DEBUG

#include "include/matrix.hxx"

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "include/nvgraph_error.hxx"
#include "include/nvgraph_vector.hxx"
#include "include/nvgraph_cublas.hxx"
#include "include/nvgraph_cusparse.hxx"
#include "include/debug_macros.h"

// =========================================================
// Useful macros
// =========================================================

// CUDA block size
#define BLOCK_SIZE 1024

// Get index of matrix entry
#define IDX(i,j,lda) ((i)+(j)*(lda))

namespace nvgraph {

  // =============================================
  // CUDA kernels
  // =============================================

  namespace {

    /// Apply diagonal matrix to vector
    template <typename IndexType_, typename ValueType_> static __global__
    void diagmv(IndexType_ n, ValueType_ alpha,
    const ValueType_ * __restrict__ D,
    const ValueType_ * __restrict__ x,
    ValueType_ * __restrict__ y) {
      IndexType_ i = threadIdx.x + blockIdx.x*blockDim.x;
      while(i<n) {
  y[i] += alpha*D[i]*x[i];
  i += blockDim.x*gridDim.x;
      }
    }

    /// Apply diagonal matrix to a set of dense vectors (tall matrix)
    template <typename IndexType_, typename ValueType_, bool beta_is_zero> 
    static __global__  void diagmm(IndexType_ n, IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ D, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) {
        IndexType_ i,j,index;
       
        for(j=threadIdx.y+blockIdx.y*blockDim.y; j<k; j+=blockDim.y*gridDim.y) {
            for(i=threadIdx.x+blockIdx.x*blockDim.x; i<n; i+=blockDim.x*gridDim.x) {
                index = i+j*n;
                if (beta_is_zero) {
                    y[index] = alpha*D[i]*x[index];
                }
                else {
                    y[index] = alpha*D[i]*x[index] + beta*y[index];
                }
            }
        }
    }
  }

  // =============================================
  // Dense matrix class
  // =============================================

  /// Constructor for dense matrix class
  /** @param _trans Whether to transpose matrix.
   *  @param _m Number of rows.
   *  @param _n Number of columns.
   *  @param _A (Input, device memory, _m*_n entries) Matrix
   *    entries, stored column-major.
   *  @param _lda Leading dimension of _A.
   */
  template <typename IndexType_, typename ValueType_>
  DenseMatrix<IndexType_,ValueType_>
  ::DenseMatrix(bool _trans,
    IndexType_ _m, IndexType_ _n,
    const ValueType_ * _A, IndexType_ _lda) 
    : Matrix<IndexType_,ValueType_>(_m,_n),
      trans(_trans), A(_A), lda(_lda) {
    Cublas::set_pointer_mode_host();
    if(_lda<_m)
      FatalError("invalid dense matrix parameter (lda<m)",
     NVGRAPH_ERR_BAD_PARAMETERS);
  }

  /// Destructor for dense matrix class
  template <typename IndexType_, typename ValueType_>
  DenseMatrix<IndexType_,ValueType_>::~DenseMatrix() {}

   /// Get and Set CUDA stream    
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::setCUDAStream(cudaStream_t _s) {
      this->s = _s;
      //printf("DenseMatrix setCUDAStream stream=%p\n",this->s);
      Cublas::setStream(_s);
  }  
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::getCUDAStream(cudaStream_t *_s) {
      *_s = this->s;
      //CHECK_CUBLAS(cublasGetStream(cublasHandle, _s));
  }  


  /// Matrix-vector product for dense matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n entries) Vector.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m entries) Output vector.
   */
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {
    Cublas::gemv(this->trans, this->m, this->n,
     &alpha, this->A, this->lda, x, 1, &beta, y, 1);
  }

  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {
      Cublas::gemm(this->trans, false, this->m, k, this->n,
          &alpha, A, lda, x, this->m, &beta, y, this->n);
  }  

  /// Color and Reorder
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::color(IndexType_ *c, IndexType_ *p) const {
      
  } 

  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::reorder(IndexType_ *p) const {

  }  

  /// Incomplete Cholesky (setup, factor and solve)
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::prec_setup(Matrix<IndexType_,ValueType_> * _M) {
      printf("ERROR: DenseMatrix prec_setup dispacthed\n");
      //exit(1);
  }
  
  template <typename IndexType_, typename ValueType_>
  void DenseMatrix<IndexType_,ValueType_>
  ::prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const {
      printf("ERROR: DenseMatrix prec_solve dispacthed\n");
      //exit(1);
  }   

  template <typename IndexType_, typename ValueType_>
  ValueType_ DenseMatrix<IndexType_, ValueType_>
  ::getEdgeSum() const {
  return 0.0;  
  }  

  // =============================================
  // CSR matrix class
  // =============================================

  /// Constructor for CSR matrix class
  /** @param _transA Whether to transpose matrix.
   *  @param _m Number of rows.
   *  @param _n Number of columns.
   *  @param _nnz Number of non-zero entries.
   *  @param _descrA Matrix properties.
   *  @param _csrValA (Input, device memory, _nnz entries) Matrix
   *    entry values.
   *  @param _csrRowPtrA (Input, device memory, _m+1 entries) Pointer
   *    to first entry in each row.
   *  @param _csrColIndA (Input, device memory, _nnz entries) Column
   *    index of each matrix entry.
   */
  template <typename IndexType_, typename ValueType_>
  CsrMatrix<IndexType_,ValueType_>
  ::CsrMatrix(bool _trans, bool _sym,
        IndexType_ _m, IndexType_ _n, IndexType_ _nnz,
        const cusparseMatDescr_t _descrA,
        /*const*/ ValueType_ * _csrValA,
        const IndexType_ * _csrRowPtrA,
        const IndexType_ * _csrColIndA) 
    : Matrix<IndexType_,ValueType_>(_m,_n),
      trans(_trans), sym(_sym),
      nnz(_nnz),  descrA(_descrA), csrValA(_csrValA),
      csrRowPtrA(_csrRowPtrA), 
      csrColIndA(_csrColIndA) {
    if(nnz<0)
      FatalError("invalid CSR matrix parameter (nnz<0)",
     NVGRAPH_ERR_BAD_PARAMETERS);
    Cusparse::set_pointer_mode_host();
  }

  /// Constructor for CSR matrix class
  /** @param G Weighted graph in CSR format
   */
  template <typename IndexType_, typename ValueType_>
  CsrMatrix<IndexType_,ValueType_>
  ::CsrMatrix(  ValuedCsrGraph<IndexType_,ValueType_> & G, const cusparseMatDescr_t _descrA)
    : Matrix<IndexType_,ValueType_>(G.get_num_vertices(), G.get_num_vertices()),
      trans(false), sym(false),
      nnz(G.get_num_edges()),
      descrA(_descrA), 
      csrValA(G.get_raw_values()),
      csrRowPtrA(G.get_raw_row_offsets()),
      csrColIndA(G.get_raw_column_indices()) {
    Cusparse::set_pointer_mode_host();
  }

  /// Destructor for CSR matrix class
  template <typename IndexType_, typename ValueType_>
  CsrMatrix<IndexType_,ValueType_>::~CsrMatrix() {}

  /// Get and Set CUDA stream    
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::setCUDAStream(cudaStream_t _s) {
      this->s = _s;
      //printf("CsrMatrix setCUDAStream stream=%p\n",this->s);
      Cusparse::setStream(_s);
  }  
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::getCUDAStream(cudaStream_t *_s) {
      *_s = this->s;
      //CHECK_CUSPARSE(cusparseGetStream(Cusparse::get_handle(), _s));
  }     
   template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const {
      //CHECK_CUSPARSE(cusparseXcsrmm(Cusparse::get_handle(), transA, this->m, k, this->n, nnz, &alpha, descrA, csrValA, csrRowPtrA, csrColIndA, x, this->n, &beta, y, this->m));
      Cusparse::csrmm(this->trans, this->sym, this->m, k, this->n, this->nnz, &alpha, this->csrValA, this->csrRowPtrA, this->csrColIndA, x, this->n, &beta, y, this->m);
  }

  /// Color and Reorder
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::color(IndexType_ *c, IndexType_ *p) const {
      
  } 

  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::reorder(IndexType_ *p) const {

  }  

  /// Incomplete Cholesky (setup, factor and solve)
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::prec_setup(Matrix<IndexType_,ValueType_> * _M) {
      //printf("CsrMatrix prec_setup dispacthed\n");
      if (!factored) {
          //analyse lower triangular factor
          CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_l));
          CHECK_CUSPARSE(cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_LOWER));
          CHECK_CUSPARSE(cusparseSetMatDiagType(descrA,CUSPARSE_DIAG_TYPE_UNIT));
          CHECK_CUSPARSE(cusparseXcsrsm_analysis(Cusparse::get_handle(),CUSPARSE_OPERATION_NON_TRANSPOSE,this->m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,info_l));
          //analyse upper triangular factor
          CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&info_u));
          CHECK_CUSPARSE(cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_UPPER));
          CHECK_CUSPARSE(cusparseSetMatDiagType(descrA,CUSPARSE_DIAG_TYPE_NON_UNIT));
          CHECK_CUSPARSE(cusparseXcsrsm_analysis(Cusparse::get_handle(),CUSPARSE_OPERATION_NON_TRANSPOSE,this->m,nnz,descrA,csrValA,csrRowPtrA,csrColIndA,info_u));
          //perform csrilu0 (should be slightly faster than csric0)
          CHECK_CUSPARSE(cusparseXcsrilu0(Cusparse::get_handle(),CUSPARSE_OPERATION_NON_TRANSPOSE,this->m,descrA,csrValA,csrRowPtrA,csrColIndA,info_l));
          //set factored flag to true
          factored=true;
      }
  }
  
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const {
      //printf("CsrMatrix prec_solve dispacthed (stream %p)\n",this->s);
      
      //preconditioning Mx=f (where M = L*U, threfore x=U\(L\f))
      //solve lower triangular factor
      CHECK_CUSPARSE(cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_LOWER));
      CHECK_CUSPARSE(cusparseSetMatDiagType(descrA,CUSPARSE_DIAG_TYPE_UNIT));
      CHECK_CUSPARSE(cusparseXcsrsm_solve(Cusparse::get_handle(),CUSPARSE_OPERATION_NON_TRANSPOSE,this->m,k,alpha,descrA,csrValA,csrRowPtrA,csrColIndA,info_l,fx,this->m,t,this->m));
      //solve upper triangular factor
      CHECK_CUSPARSE(cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_UPPER));
      CHECK_CUSPARSE(cusparseSetMatDiagType(descrA,CUSPARSE_DIAG_TYPE_NON_UNIT));
      CHECK_CUSPARSE(cusparseXcsrsm_solve(Cusparse::get_handle(),CUSPARSE_OPERATION_NON_TRANSPOSE,this->m,k,alpha,descrA,csrValA,csrRowPtrA,csrColIndA,info_u,t,this->m,fx,this->m));
      
  } 

  /// Matrix-vector product for CSR matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n entries) Vector.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m entries) Output vector.
   */
  template <typename IndexType_, typename ValueType_>
  void CsrMatrix<IndexType_,ValueType_>
  ::mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {
    // TODO: consider using merge-path csrmv
    Cusparse::csrmv(this->trans, this->sym, this->m, this->n,
        this->nnz, &alpha, this->csrValA,
        this->csrRowPtrA, this->csrColIndA,
        x, &beta, y);

  }

  template <typename IndexType_, typename ValueType_>
  ValueType_ CsrMatrix<IndexType_, ValueType_>
  ::getEdgeSum() const {
  return 0.0;  
  }  

  // =============================================
  // Laplacian matrix class
  // =============================================

  /// Constructor for Laplacian matrix class
  /** @param A Adjacency matrix
   */
  template <typename IndexType_, typename ValueType_>
  LaplacianMatrix<IndexType_, ValueType_>
  ::LaplacianMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A)
    : Matrix<IndexType_,ValueType_>(_A.m,_A.n), A(&_A) {

    // Check that adjacency matrix is square
    if(_A.m != _A.n)
      FatalError("cannot construct Laplacian matrix from non-square adjacency matrix",
     NVGRAPH_ERR_BAD_PARAMETERS);
    //set CUDA stream
    this->s = NULL;
    // Construct degree matrix
    D.allocate(_A.m,this->s);
    Vector<ValueType_> ones(this->n,this->s);
    ones.fill(1.0);
    _A.mv(1, ones.raw(), 0, D.raw());

     // Set preconditioning matrix pointer to NULL
    M=NULL;
  }

  /// Destructor for Laplacian matrix class
  template <typename IndexType_, typename ValueType_>
  LaplacianMatrix<IndexType_, ValueType_>::~LaplacianMatrix() {}
  
  /// Get and Set CUDA stream     
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s) {
      this->s = _s;
      //printf("LaplacianMatrix setCUDAStream stream=%p\n",this->s);
      A->setCUDAStream(_s);
      if (M != NULL) {
          M->setCUDAStream(_s);
      }
  }  
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t * _s) {
      *_s = this->s;
      //A->getCUDAStream(_s);
  }  

  /// Matrix-vector product for Laplacian matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n entries) Vector.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m entries) Output vector.
   */
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>
  ::mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {

    // Scale result vector
    if(beta==0)
      CHECK_CUDA(cudaMemset(y, 0, (this->n)*sizeof(ValueType_)))
    else if(beta!=1)
      thrust::transform(thrust::device_pointer_cast(y),
      thrust::device_pointer_cast(y+this->n),
      thrust::make_constant_iterator(beta),
      thrust::device_pointer_cast(y),
      thrust::multiplies<ValueType_>());
    
    // Apply diagonal matrix
    dim3 gridDim, blockDim;
    gridDim.x  = min(((this->n)+BLOCK_SIZE-1)/BLOCK_SIZE, 65535);
    gridDim.y  = 1;
    gridDim.z  = 1;
    blockDim.x = BLOCK_SIZE;
    blockDim.y = 1;
    blockDim.z = 1;
    diagmv <<< gridDim, blockDim , 0, A->s>>> (this->n, alpha, D.raw(), x, y);
    cudaCheckError();

    // Apply adjacency matrix
    A->mv(-alpha, x, 1, y);
    
  }
    /// Matrix-vector product for Laplacian matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n*k entries) nxk dense matrix.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m*k entries) Output mxk dense matrix.
   */
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>
  ::mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {
      // Apply diagonal matrix
      ValueType_ one = (ValueType_)1.0;
      this->dm(k,alpha,x,beta,y);     

      // Apply adjacency matrix
      A->mm(k, -alpha, x, one, y);      
  }

  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>
  ::dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const {
      IndexType_ t = k*(this->n);
      dim3 gridDim, blockDim;

      //setup launch parameters
      gridDim.x  = min(((this->n)+BLOCK_SIZE-1)/BLOCK_SIZE, 65535);
      gridDim.y  = min(k,65535);
      gridDim.z  = 1;
      blockDim.x = BLOCK_SIZE;
      blockDim.y = 1;
      blockDim.z = 1;

      // Apply diagonal matrix
      if(beta == 0.0) {
          //set vectors to 0 (WARNING: notice that you need to set, not scale, because of NaNs corner case)
          CHECK_CUDA(cudaMemset(y, 0, t*sizeof(ValueType_)));
          diagmm<IndexType_,ValueType_,true> <<< gridDim, blockDim, 0, A->s >>> (this->n, k, alpha, D.raw(), x, beta, y);
      }
      else {
          diagmm<IndexType_,ValueType_,false><<< gridDim, blockDim, 0, A->s >>> (this->n, k, alpha, D.raw(), x, beta, y);
      }
      cudaCheckError();
  }


  /// Color and Reorder
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_,ValueType_>
  ::color(IndexType_ *c, IndexType_ *p) const {
      
  } 

  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_,ValueType_>
  ::reorder(IndexType_ *p) const {

  }    

  /// Solve preconditioned system M x = f for a set of k vectors 
  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>
  ::prec_setup(Matrix<IndexType_,ValueType_> * _M) {
      //save the pointer to preconditioner M
      M = _M;
      if (M != NULL) {
          //setup the preconditioning matrix M
          M->prec_setup(NULL);
      }
  }  

  template <typename IndexType_, typename ValueType_>
  void LaplacianMatrix<IndexType_, ValueType_>
  ::prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const {
      if (M != NULL) {
          //preconditioning
          M->prec_solve(k,alpha,fx,t);
      }
  }   

  template <typename IndexType_, typename ValueType_>
  ValueType_ LaplacianMatrix<IndexType_, ValueType_>
  ::getEdgeSum() const {
  return 0.0;  
  }  
// =============================================
  // Modularity matrix class
  // =============================================

  /// Constructor for Modularity matrix class
  /** @param A Adjacency matrix
   */
  template <typename IndexType_, typename ValueType_>
  ModularityMatrix<IndexType_, ValueType_>
  ::ModularityMatrix(/*const*/ Matrix<IndexType_,ValueType_> & _A, IndexType_ _nnz)
    : Matrix<IndexType_,ValueType_>(_A.m,_A.n), A(&_A), nnz(_nnz){

    // Check that adjacency matrix is square
    if(_A.m != _A.n)
      FatalError("cannot construct Modularity matrix from non-square adjacency matrix",
     NVGRAPH_ERR_BAD_PARAMETERS);

    //set CUDA stream
    this->s = NULL;
    // Construct degree matrix
    D.allocate(_A.m,this->s);
    Vector<ValueType_> ones(this->n,this->s);
    ones.fill(1.0);
    _A.mv(1, ones.raw(), 0, D.raw());
     // D.dump(0,this->n);
     edge_sum = D.nrm1();

     // Set preconditioning matrix pointer to NULL
    M=NULL;
  }

  /// Destructor for Modularity matrix class
  template <typename IndexType_, typename ValueType_>
  ModularityMatrix<IndexType_, ValueType_>::~ModularityMatrix() {}
  
  /// Get and Set CUDA stream     
  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>::setCUDAStream(cudaStream_t _s) {
      this->s = _s;
      //printf("ModularityMatrix setCUDAStream stream=%p\n",this->s);
      A->setCUDAStream(_s);
      if (M != NULL) {
          M->setCUDAStream(_s);
      }
  }  

  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>::getCUDAStream(cudaStream_t * _s) {
      *_s = this->s;
      //A->getCUDAStream(_s);
  }  

  /// Matrix-vector product for Modularity matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n entries) Vector.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m entries) Output vector.
   */
  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>
  ::mv(ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {

    // Scale result vector
    if(alpha!=1 || beta!=0)
      FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);

     //CHECK_CUBLAS(cublasXdot(handle, this->n, const double *x, int incx, const double *y, int incy, double *result));
    // y = A*x
    A->mv(alpha, x, 0, y);
     ValueType_  dot_res;
    //gamma = d'*x
    Cublas::dot(this->n, D.raw(), 1, x, 1, &dot_res);
    // y = y -(gamma/edge_sum)*d
    Cublas::axpy(this->n, -(dot_res/this->edge_sum), D.raw(), 1, y, 1);
  }
  /// Matrix-vector product for Modularity matrix class
  /** y is overwritten with alpha*A*x+beta*y.
   *
   *  @param alpha Scalar.
   *  @param x (Input, device memory, n*k entries) nxk dense matrix.
   *  @param beta Scalar.
   *  @param y (Input/output, device memory, m*k entries) Output mxk dense matrix.
   */
  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>
  ::mm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x,
       ValueType_ beta, ValueType_ * __restrict__ y) const {
       FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
  }

  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>
  ::dm(IndexType_ k, ValueType_ alpha, const ValueType_ * __restrict__ x, ValueType_ beta, ValueType_ * __restrict__ y) const {
       FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);

  }

  /// Color and Reorder
  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_,ValueType_>
  ::color(IndexType_ *c, IndexType_ *p) const {
    FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
 
  } 

  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_,ValueType_>
  ::reorder(IndexType_ *p) const {
    FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
  }    

  /// Solve preconditioned system M x = f for a set of k vectors 
  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>
  ::prec_setup(Matrix<IndexType_,ValueType_> * _M) {
      //save the pointer to preconditioner M
      M = _M;
      if (M != NULL) {
          //setup the preconditioning matrix M
          M->prec_setup(NULL);
      }
  }  

  template <typename IndexType_, typename ValueType_>
  void ModularityMatrix<IndexType_, ValueType_>
  ::prec_solve(IndexType_ k, ValueType_ alpha, ValueType_ * __restrict__ fx, ValueType_ * __restrict__ t) const {
      if (M != NULL) {
        FatalError("This isn't implemented for Modularity Matrix currently", NVGRAPH_ERR_NOT_IMPLEMENTED);
      }
  }   

  template <typename IndexType_, typename ValueType_>
  ValueType_ ModularityMatrix<IndexType_, ValueType_>
  ::getEdgeSum() const {
      return edge_sum;
  }  
  // Explicit instantiation
  template class Matrix<int,float>;
  template class Matrix<int, double>;
  template class DenseMatrix<int,float>;
  template class DenseMatrix<int,double>;
  template class CsrMatrix<int,float>;
  template class CsrMatrix<int,double>;
  template class LaplacianMatrix<int,float>;
  template class LaplacianMatrix<int,double>;
  template class ModularityMatrix<int,float>;
  template class ModularityMatrix<int,double>;

}
//#endif 
