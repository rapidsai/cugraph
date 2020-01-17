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

#if !defined(CUSPARSE_INTERNAL_H_)
#define CUSPARSE_INTERNAL_H_


#ifndef CUSPARSEAPI
#ifdef _WIN32
#define CUSPARSEAPI __stdcall
#else
#define CUSPARSEAPI 
#endif
#endif


#define CACHE_LINE_SIZE   128 

#define ALIGN_32(x)   ((((x)+31)/32)*32)



#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */


struct csrilu02BatchInfo;
typedef struct csrilu02BatchInfo *csrilu02BatchInfo_t;


struct csrxilu0Info;
typedef struct csrxilu0Info *csrxilu0Info_t;

struct csrxgemmSchurInfo;
typedef struct csrxgemmSchurInfo *csrxgemmSchurInfo_t;

struct csrxtrsmInfo;
typedef struct csrxtrsmInfo  *csrxtrsmInfo_t;

struct csrilu03Info;
typedef struct csrilu03Info *csrilu03Info_t;

struct csrmmInfo;
typedef struct csrmmInfo *csrmmInfo_t;


cudaStream_t cusparseGetStreamInternal(const struct cusparseContext *ctx);


cusparseStatus_t CUSPARSEAPI cusparseCheckBuffer(
    cusparseHandle_t handle,
    void *workspace);

//------- gather: dst = src(map) ---------------------

cusparseStatus_t CUSPARSEAPI cusparseIgather(
    cusparseHandle_t handle,
    int n,
    const int *src,
    const int *map,
    int *dst);

cusparseStatus_t CUSPARSEAPI cusparseSgather(
    cusparseHandle_t handle,
    int n,
    const float *src,
    const int *map,
    float *dst);

cusparseStatus_t CUSPARSEAPI cusparseDgather(
    cusparseHandle_t handle,
    int n,
    const double *src,
    const int *map,
    double *dst);

cusparseStatus_t CUSPARSEAPI cusparseCgather(
    cusparseHandle_t handle,
    int n,
    const cuComplex *src,
    const int *map,
    cuComplex *dst);

cusparseStatus_t CUSPARSEAPI cusparseZgather(
    cusparseHandle_t handle,
    int n,
    const cuDoubleComplex *src,
    const int *map,
    cuDoubleComplex *dst);


//------- scatter: dst(map) = src ---------------------

cusparseStatus_t CUSPARSEAPI cusparseIscatter(
    cusparseHandle_t handle,
    int n,
    const int *src,
    int *dst,
    const int *map);

cusparseStatus_t CUSPARSEAPI cusparseSscatter(
    cusparseHandle_t handle,
    int n,
    const float *src,
    float *dst,
    const int *map);

cusparseStatus_t CUSPARSEAPI cusparseDscatter(
    cusparseHandle_t handle,
    int n,
    const double *src,
    double *dst,
    const int *map);

cusparseStatus_t CUSPARSEAPI cusparseCscatter(
    cusparseHandle_t handle,
    int n,
    const cuComplex *src,
    cuComplex *dst,
    const int *map);

cusparseStatus_t CUSPARSEAPI cusparseZscatter(
    cusparseHandle_t handle,
    int n,
    const cuDoubleComplex *src,
    cuDoubleComplex *dst,
    const int *map);


// x[j] = j 
cusparseStatus_t CUSPARSEAPI cusparseIidentity(
    cusparseHandle_t handle,
    int n,
    int *x);

// x[j] = val
cusparseStatus_t CUSPARSEAPI cusparseImemset(
    cusparseHandle_t handle,
    int n,
    int val,
    int *x);

cusparseStatus_t CUSPARSEAPI cusparseI64memset(
    cusparseHandle_t handle,
    size_t n,
    int val,
    int *x);


// ----------- reduce -----------------

/*
 * cusparseStatus_t 
 *      cusparseIreduce_bufferSize( cusparseHandle_t handle,
 *                                   int n,
 *                                   int *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^31.
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseIreduce_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);

/*
 * cusparseStatus_t 
 *     cusparseIreduce(cusparseHandle_t handle,
 *                     int n,
 *                     int *src,
 *                     int *pBuffer,
 *                     int *total_sum)
 *  
 *    total_sum = reduction(src)
 *
 *  Input
 * -------
 *  handle            handle to the CUSPARSE library context.
 *    n               number of elements in src and dst.
 *  src               <int> array of n elements.
 *  pBuffer           working space, the size is reported by cusparseIinclusiveScan_bufferSizeExt.
 *                    Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * -------
 *  total_sum         total_sum = reduction(src) if total_sum is not a NULL pointer.
 *
 *
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 *                                  it is possible if pBuffer is NULL.
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *
 * 
 */
cusparseStatus_t CUSPARSEAPI cusparseIreduce(
    cusparseHandle_t handle,
    int n,
    int *src,
    void *pBuffer,
    int *total_sum);



// ----------- prefix sum -------------------

/*
 * cusparseStatus_t 
 *      cusparseIinclusiveScan_bufferSizeExt( cusparseHandle_t handle,
 *                                   int n,
 *                                   size_t *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^31.
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseIinclusiveScan_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);


/*
 * cusparseStatus_t 
 *     cusparseIinclusiveScan(cusparseHandle_t handle,
 *                             int base,
 *                             int n,
 *                             int *src,
 *                             void *pBuffer,
 *                             int *dst,
 *                             int *total_sum)
 *  
 *    dst = inclusiveScan(src) + base
 *    total_sum = reduction(src)
 *
 *  Input
 * -------
 *  handle            handle to the CUSPARSE library context.
 *    n               number of elements in src and dst.
 *  src               <int> array of n elements.
 *  pBuffer           working space, the size is reported by cusparseIinclusiveScan_bufferSizeExt.
 *                    Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * -------
 *  dst               <int> array of n elements.
 *                    dst = inclusiveScan(src) + base
 *  total_sum         total_sum = reduction(src) if total_sum is not a NULL pointer.
 *
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 *                                  it is possible if pBuffer is NULL.
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 * 
 */
cusparseStatus_t CUSPARSEAPI cusparseIinclusiveScan(
    cusparseHandle_t handle,
    int base,
    int n,
    int *src,
    void *pBuffer,
    int *dst,
    int *total_sum);

// ----------- stable sort -----------------

/*
 * cusparseStatus_t 
 *      cusparseIstableSortByKey_bufferSizeExt( cusparseHandle_t handle,
 *                                   int n,
 *                                   size_t *pBufferSizeInBytes)
 * Input
 * -----
 * handle        handle to CUSPARSE library context.
 * n             number of elements.
 *
 * Output
 * ------
 * pBufferSizeInBytes   size of working space in bytes.
 *  
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_INVALID_VALUE    n is too big or negative
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *                                  If n is normal, we should not have this internal error.
 *
 * ---------
 * Assumption:
 *    Only support n < 2^30 because of domino scheme. 
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseIstableSortByKey_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSizeInBytes);


/*
 * cusparseStatus_t 
 *      cusparseIstableSortByKey( cusparseHandle_t handle,
 *                                   int n,
 *                                   int *key,
 *                                   int *P)
 *
 *  in-place radix sort. 
 *  This is an inhouse design of thrust::stable_sort_by_key(key, P)
 *
 * Input
 * -----
 * handle    handle to CUSPARSE library context.
 * n         number of elements.
 * key       <int> array of n elements.  
 * P         <int> array of n elements.  
 * pBuffer   working space, the size is reported by cusparseIstableSortByKey_bufferSize.
 *           Or it can be a NULL pointer, then CUSPARSE library allocates working space implicitly.
 *
 * Output
 * ------
 * key       <int> array of n elements.  
 * P         <int> array of n elements.  
 *
 * Error Status
 * ------------
 * CUSPARSE_STATUS_SUCCESS          the operation completed successfully.
 * CUSPARSE_STATUS_NOT_INITIALIZED  the library was not initialized.   
 * CUSPARSE_STATUS_ALLOC_FAILED     the resources could not be allocated.
 * CUSPARSE_STATUS_INTERNAL_ERROR   an internal operation failed.
 *
 * -----
 * Assumption:
 *    Only support n < 2^30 because of domino scheme. 
 *
 * -----
 * Usage:
 *   int nBufferSize = 0;
 *   status = cusparseIstableSortByKey_bufferSize(handle, n, &nBufferSize);
 *   assert(CUSPARSE_STATUS_SUCCESS == status);
 *   
 *   int *pBuffer;
 *   cudaStat = cudaMalloc((void**)&pBuffer, (size_t)nBufferSize);
 *   assert(cudaSuccess == cudaStat);
 *
 *   d_P = 0:n-1 ;
 *   status = cusparseIstableSortByKey(handle, n, d_csrRowPtrA, d_P, pBuffer);
 *   assert(CUSPARSE_STATUS_SUCCESS == status);
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseIstableSortByKey(
    cusparseHandle_t handle,
    int n,
    int *key,
    int *P,
    void *pBuffer);



// ------------------- csr42csr ------------------

cusparseStatus_t CUSPARSEAPI cusparseXcsr42csr_bufferSize(
    cusparseHandle_t handle,
    int m,
    int n,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    size_t *pBufferSizeInByte );

cusparseStatus_t CUSPARSEAPI cusparseXcsr42csrRows(
    cusparseHandle_t handle,
    int m,
    int n,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,
    int *nnzTotalDevHostPtr,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseXcsr42csrCols(
    cusparseHandle_t handle,
    int m,
    int n,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    const int *csrRowPtrC,
    int *csrColIndC,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseScsr42csrVals(
    cusparseHandle_t handle,
    int m,
    int n,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseDcsr42csrVals(
    cusparseHandle_t handle,
    int m,
    int n,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseCcsr42csrVals(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuComplex *alpha,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    cuComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseZcsr42csrVals(
    cusparseHandle_t handle,
    int m,
    int n,
    const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,
    void *pBuffer );


// ----- csrmv_hyb ------------------------------

cusparseStatus_t CUSPARSEAPI cusparseScsrmv_hyb(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const float *alpha,
    const cusparseMatDescr_t descra,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *x,
    const float *beta,
    float *y);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmv_hyb(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const cusparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta, 
    double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv_hyb(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const cuComplex *alpha,
    const cusparseMatDescr_t descra,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *x,
    const cuComplex *beta,
    cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv_hyb(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descra,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *x,
    const cuDoubleComplex *beta,
    cuDoubleComplex *y);


// ------------- getrf_ilu ---------------------

cusparseStatus_t CUSPARSEAPI cusparseSgetrf_ilu(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    float *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    float *boost_ptr);

cusparseStatus_t CUSPARSEAPI cusparseDgetrf_ilu(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    double *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    double *boost_ptr);

cusparseStatus_t CUSPARSEAPI cusparseCgetrf_ilu(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    cuComplex *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    cuComplex *boost_ptr);

cusparseStatus_t CUSPARSEAPI cusparseZgetrf_ilu(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    cuDoubleComplex *A,
    const int *pattern,
    const int lda,
    int *d_status,
    int enable_boost,
    double *tol_ptr,
    cuDoubleComplex *boost_ptr);


// ------------- potrf_ic ---------------------

cusparseStatus_t CUSPARSEAPI cusparseSpotrf_ic(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    float *A,
    const int *pattern,
    const int lda,
    int *d_status);

cusparseStatus_t CUSPARSEAPI cusparseDpotrf_ic(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    double *A,
    const int *pattern,
    const int lda,
    int *d_status);

cusparseStatus_t CUSPARSEAPI cusparseCpotrf_ic(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    cuComplex *A,
    const int *pattern,
    const int lda,
    int *d_status);

cusparseStatus_t CUSPARSEAPI cusparseZpotrf_ic(
    cusparseHandle_t handle,
    const int submatrix_k,
    const int n,
    cuDoubleComplex *A,
    const int *pattern,
    const int lda,
    int *d_status);


cusparseStatus_t CUSPARSEAPI cusparseXcsric02_denseConfig(
    csric02Info_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

cusparseStatus_t CUSPARSEAPI cusparseXcsric02_workspaceConfig(
    csric02Info_t info,
    int disable_workspace_limit);


cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_denseConfig(
    csrilu02Info_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_workspaceConfig(
    csrilu02Info_t info,
    int disable_workspace_limit);


cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_denseConfig(
    csrilu02BatchInfo_t info,
    int enable_dense_block,
    int max_dim_dense_block,
    int threshold_dense_block,
    double ratio);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_workspaceConfig(
    csrilu02BatchInfo_t info,
    int disable_workspace_limit);



// ---------------- csric02 internal ----------------
cusparseStatus_t CUSPARSEAPI cusparseXcsric02_getLevel(
    csric02Info_t info,
    int **level_ref);

cusparseStatus_t CUSPARSEAPI cusparseScsric02_internal(
    cusparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsric02_internal(
    cusparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsric02_internal(
    cusparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsric02_internal(
    cusparseHandle_t handle,
    int enable_potrf,
    int dense_block_start,
    //int dense_block_dim, // = m - dense_block_start
    int dense_block_lda,
    int *level,  // level is a permutation vector of 0:(m-1)
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csric02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

// csrilu02 internal

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_getLevel(
    csrilu02Info_t info,
    int **level_ref);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02_getCsrEndPtrL(
    csrilu02Info_t info,
    int **csrEndPtrL_ref);


// ----------------- batch ilu0 -----------------

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu02BatchInfo(
    csrilu02BatchInfo_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu02BatchInfo(
    csrilu02BatchInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu02Batch_zeroPivot(
    cusparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_numericBoost(
    cusparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_numericBoost(
    cusparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_numericBoost(
    cusparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    cuComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_numericBoost(
    cusparseHandle_t handle,
    csrilu02BatchInfo_t info,
    int enable_boost,
    double *tol,
    cuDoubleComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    size_t *pBufferSizeInBytes);


cusparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseScsrilu02Batch(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu02Batch(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu02Batch(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu02Batch(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrilu02BatchInfo_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

// --------------- csrsv2 batch --------------

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_bufferSizeExt(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_bufferSizeExt(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_bufferSizeExt(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_bufferSizeExt(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_analysis(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_analysis(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_analysis(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_analysis(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    int batchSize,
    csrsv2Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseXcsrsv2Batch_zeroPivot(
    cusparseHandle_t handle,
    csrsv2Info_t info,
    int *position);


cusparseStatus_t CUSPARSEAPI cusparseScsrsv2Batch_solve(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const float *x,
    float *y,
    int batchSize,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrsv2Batch_solve(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const double *x,
    double *y,
    int batchSize,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrsv2Batch_solve(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    const cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const cuComplex *x,
    cuComplex *y,
    int batchSize,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrsv2Batch_solve(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int nnz,
    const cusparseMatDescr_t descra,
    const cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrsv2Info_t info,
    const cuDoubleComplex *x,
    cuDoubleComplex *y,
    int batchSize,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

//-------------- csrgemm2 -------------

cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2_spaceConfig(
    csrgemm2Info_t info,
    int disable_space_limit);

// internal-use only
cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Rows_bufferSize(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrColIndB,

    csrgemm2Info_t info,
    size_t *pBufferSize );

// internal-use only
cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Cols_bufferSize(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrColIndB,

    csrgemm2Info_t info,
    size_t *pBufferSize );



cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Rows(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const cusparseMatDescr_t descrC,
    int *csrRowPtrC,

    int *nnzTotalDevHostPtr,
    csrgemm2Info_t info,
    void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseXcsrgemm2Cols(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const cusparseMatDescr_t descrC,
    const int *csrRowPtrC,
    int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );

cusparseStatus_t CUSPARSEAPI cusparseScsrgemm2Vals(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const float *alpha,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const float *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const float *beta,

    const cusparseMatDescr_t descrC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseDcsrgemm2Vals(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const double *alpha,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const double *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const double *beta,

    const cusparseMatDescr_t descrC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseCcsrgemm2Vals(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cuComplex *alpha,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const cuComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const cuComplex *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const cuComplex *beta,

    const cusparseMatDescr_t descrC,
    cuComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


cusparseStatus_t CUSPARSEAPI cusparseZcsrgemm2Vals(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    const cuDoubleComplex *alpha,

    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    const cusparseMatDescr_t descrB,
    int nnzB,
    const cuDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    const cusparseMatDescr_t descrD,
    int nnzD,
    const cuDoubleComplex *csrValD,
    const int *csrRowPtrD,
    const int *csrEndPtrD,
    const int *csrColIndD,

    const cuDoubleComplex *beta,

    const cusparseMatDescr_t descrC,
    cuDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrgemm2Info_t info,
    void *pBuffer );


// ---------------- csr2csc2

cusparseStatus_t CUSPARSEAPI cusparseXcsr2csc2_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const int *csrRowPtr,
    const int *csrColInd,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseXcsr2csc2(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnz,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrColInd,
    int *cscColPtr,
    int *cscRowInd,
    int *cscValInd,
    void *pBuffer);

#if 0
// ------------- CSC ILU0

cusparseStatus_t CUSPARSEAPI cusparseXcscilu02_getLevel(
    cscilu02Info_t info,
    int **level_ref);

cusparseStatus_t CUSPARSEAPI cusparseXcscilu02_getCscColPtrL(
    cscilu02Info_t info,
    int **cscColPtrL_ref);

cusparseStatus_t CUSPARSEAPI cusparseCreateCscilu02Info(
    cscilu02Info_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCscilu02Info(
    cscilu02Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcscilu02_zeroPivot(
    cusparseHandle_t handle,
    cscilu02Info_t info,
    int *position);

cusparseStatus_t CUSPARSEAPI cusparseScscilu02_numericBoost(
    cusparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseDcscilu02_numericBoost(
    cusparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseCcscilu02_numericBoost(
    cusparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    cuComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseZcscilu02_numericBoost(
    cusparseHandle_t handle,
    cscilu02Info_t info,
    int enable_boost,
    double *tol,
    cuDoubleComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseScscilu02_bufferSize(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDcscilu02_bufferSize(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCcscilu02_bufferSize(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZcscilu02_bufferSize(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    int *pBufferSizeInBytes);


cusparseStatus_t CUSPARSEAPI cusparseScscilu02_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcscilu02_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcscilu02_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcscilu02_analysis(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseScscilu02(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcscilu02(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcscilu02(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcscilu02(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *cscVal,
    const int *cscColPtr,
    const int *cscEndPtr,
    const int *cscRowInd,
    cscilu02Info_t info,
    cusparseSolvePolicy_t policy,
    void *pBuffer);
#endif

// ------------- csrxjusqua

cusparseStatus_t CUSPARSEAPI cusparseXcsrxjusqua(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,
    int *csrjusqua );

// ------------ csrxilu0

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrxilu0Info(
    csrxilu0Info_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrxilu0Info(
    csrxilu0Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcsrxilu0_zeroPivot(
    cusparseHandle_t handle,
    csrxilu0Info_t info,
    int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsrxilu0_numericBoost(
    cusparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseDcsrxilu0_numericBoost(
    cusparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseCcsrxilu0_numericBoost(
    cusparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    cuComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseZcsrxilu0_numericBoost(
    cusparseHandle_t handle,
    csrxilu0Info_t info,
    int enable_boost,
    double *tol,
    cuDoubleComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseXcsrxilu0_bufferSizeExt(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseScsrxilu0(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const cusparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrxilu0(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const cusparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrxilu0(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const cusparseMatDescr_t descrA,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrxilu0(
    cusparseHandle_t handle,
    int iax,
    int iay,
    int m,
    int n,
    int k,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrEndPtr,
    const int *csrColInd,
    csrxilu0Info_t info,
    void *pBuffer);

// ----------- csrxgemmSchur

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrxgemmSchurInfo(
    csrxgemmSchurInfo_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrxgemmSchurInfo(
    csrxgemmSchurInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcsrxgemmSchur_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    int nnzB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const cusparseMatDescr_t descrC,
    int nnzC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    size_t *pBufferSizeInBytes);


cusparseStatus_t CUSPARSEAPI cusparseScsrxgemmSchur(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    int nnzB,
    const float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const cusparseMatDescr_t descrC,
    int nnzC,
    float *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseDcsrxgemmSchur(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    int nnzB,
    const double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const cusparseMatDescr_t descrC,
    int nnzC,
    double *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseCcsrxgemmSchur(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    int nnzB,
    const cuComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const cusparseMatDescr_t descrC,
    int nnzC,
    cuComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseZcsrxgemmSchur(
    cusparseHandle_t handle,
    int m,
    int n,
    int k,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    int nnzA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    int nnzB,
    const cuDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    int icx,
    int icy,
    const cusparseMatDescr_t descrC,
    int nnzC,
    cuDoubleComplex *csrValC,
    const int *csrRowPtrC,
    const int *csrEndPtrC,
    const int *csrColIndC,

    csrxgemmSchurInfo_t info,
    void *pBuffer);

// ---------- csrxtrsm

#if 0
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrxtrsmInfo(
    csrxtrsmInfo_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrxtrsmInfo(
    csrxtrsmInfo_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcsrxtrsm_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t  CUSPARSEAPI cusparseScsrxtrsm(
    cusparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    float *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);

cusparseStatus_t  CUSPARSEAPI cusparseDcsrxtrsm(
    cusparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    double *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);

cusparseStatus_t  CUSPARSEAPI cusparseCcsrxtrsm(
    cusparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    cuComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);


cusparseStatus_t  CUSPARSEAPI cusparseZcsrxtrsm(
    cusparseHandle_t handle,

    int m,
    int n,

    cusparseSideMode_t side,

    int iax,
    int iay,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrEndPtrA,
    const int *csrColIndA,

    int ibx,
    int iby,
    const cusparseMatDescr_t descrB,
    cuDoubleComplex *csrValB,
    const int *csrRowPtrB,
    const int *csrEndPtrB,
    const int *csrColIndB,

    csrxtrsmInfo_t info,
    void *pBuffer);
#endif

// ------ CSR ilu03
cusparseStatus_t CUSPARSEAPI cusparseCreateCsrilu03Info(
    csrilu03Info_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrilu03Info(
    csrilu03Info_t info);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu03_bufferSizeExt(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseXcsrilu03_zeroPivot(
    cusparseHandle_t handle,
    csrilu03Info_t info,
    int *position);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu03_numericBoost(
    cusparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    float *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu03_numericBoost(
    cusparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    double *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu03_numericBoost(
    cusparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    cuComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu03_numericBoost(
    cusparseHandle_t handle,
    csrilu03Info_t info,
    int enable_boost,
    double *tol,
    cuDoubleComplex *numeric_boost);

cusparseStatus_t CUSPARSEAPI cusparseScsrilu03(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrilu03(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrilu03(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrilu03(
    cusparseHandle_t handle,
    int m,
    int nnz,
    const cusparseMatDescr_t descrA,
    cuDoubleComplex *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    csrilu03Info_t info,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseXcsrValid(
    cusparseHandle_t handle,
    int m,
    int n,
    int nnzA,
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    int *valid);


cusparseStatus_t CUSPARSEAPI cusparseScsrmm3(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm3(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmm3(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const cuComplex *alpha,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *B,
    int ldb,
    const cuComplex *beta,
    cuComplex *C,
    int ldc,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmm3(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *B,
    int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C,
    int ldc,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseStranspose(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    int m,
    int n,
    const float *alpha,
    const float *A,
    int lda,
    float *C,
    int ldc);

cusparseStatus_t CUSPARSEAPI cusparseDtranspose(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    int m,
    int n,
    const double *alpha,
    const double *A,
    int lda,
    double *C,
    int ldc);

cusparseStatus_t CUSPARSEAPI cusparseCtranspose(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    int m,
    int n,
    const cuComplex *alpha,
    const cuComplex *A,
    int lda,
    cuComplex *C,
    int ldc);

cusparseStatus_t CUSPARSEAPI cusparseZtranspose(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    int m,
    int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *A,
    int lda,
    cuDoubleComplex *C,
    int ldc);


cusparseStatus_t CUSPARSEAPI cusparseScsrmv_binary(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const float *alpha,
    const cusparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *x,
    const float *beta,
    float *y);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmv_binary(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const cusparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta,
    double *y);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmv_binary(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const cuComplex *alpha,
    const cusparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuComplex *x,
    const cuComplex *beta,
    cuComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmv_binary(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descra,
    const int *csrRowPtr,
    const int *csrColInd,
    const cuDoubleComplex *x,
    const cuDoubleComplex *beta,
    cuDoubleComplex *y);

cusparseStatus_t CUSPARSEAPI cusparseCreateCsrmmInfo(
    csrmmInfo_t *info);

cusparseStatus_t CUSPARSEAPI cusparseDestroyCsrmmInfo(
    csrmmInfo_t info);

cusparseStatus_t CUSPARSEAPI csrmm4_analysis(
    cusparseHandle_t handle,
    int m, // number of rows of A
    int k, // number of columns of A
    int nnzA, // number of nonzeros of A
    const cusparseMatDescr_t descrA,
    const int *csrRowPtrA, // <int> m+1
    const int *csrColIndA, // <int> nnzA
    csrmmInfo_t info,
    double *ratio // nnzB / nnzA
    );


cusparseStatus_t CUSPARSEAPI cusparseScsrmm4(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm4(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseCcsrmm4(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const cuComplex *alpha,
    const cusparseMatDescr_t descrA,
    const cuComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuComplex *B,
    int ldb,
    const cuComplex *beta,
    cuComplex *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseZcsrmm4(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnz,
    const cuDoubleComplex *alpha,
    const cusparseMatDescr_t descrA,
    const cuDoubleComplex *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const cuDoubleComplex *B,
    int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex *C,
    int ldc,
    csrmmInfo_t info,
    void *buffer);

cusparseStatus_t CUSPARSEAPI cusparseScsrmm5(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    const float  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc
    );

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm5(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    const double  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc
    );


cusparseStatus_t CUSPARSEAPI cusparseScsrmm6(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    const float  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc
    );

cusparseStatus_t CUSPARSEAPI cusparseDcsrmm6(
    cusparseHandle_t handle,
    cusparseOperation_t transa,
    cusparseOperation_t transb,
    int m,
    int n,
    int k,
    int nnzA,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    const double  *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *B,
    int ldb,
    const double *beta,
    double *C,
    int ldc
    );



cusparseStatus_t CUSPARSEAPI cusparseSmax(
    cusparseHandle_t handle,
    int n,
    const float *x,
    float *valueHost,
    float *work  /* at least n+1 */
    );

cusparseStatus_t CUSPARSEAPI cusparseDmax(
    cusparseHandle_t handle,
    int n,
    const double *x,
    double *valueHost,
    double *work  /* at least n+1 */
    );

cusparseStatus_t CUSPARSEAPI cusparseSmin(
    cusparseHandle_t handle,
    int n,
    const float *x,
    float *valueHost,
    float *work  /* at least n+1 */
    );

cusparseStatus_t CUSPARSEAPI cusparseDmin(
    cusparseHandle_t handle,
    int n,
    const double *x,
    double *valueHost,
    double *work  /* at least n+1 */
    );

cusparseStatus_t CUSPARSEAPI cusparseI16sort_internal_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseI16sort_internal(
    cusparseHandle_t handle,
    int num_bits, /* <= 16 */
    int n,
    unsigned short *key,
    int *P,
    int ascend,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseI32sort_internal_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseI32sort_internal(
    cusparseHandle_t handle,
    int num_bits, /* <= 32 */
    int n,
    unsigned int *key,
    int *P,
    int ascend,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseI64sort_internal_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseI64sort_internal(
    cusparseHandle_t handle,
    int num_bits, /* <= 64 */
    int n,
    unsigned long long *key,
    int *P,
    int ascend,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseIsort_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const int *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseIsort(
    cusparseHandle_t handle,
    int n,
    int *key,
    int *P,
    int ascend,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseSsort_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const float *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseSsort(
    cusparseHandle_t handle,
    int n,
    float *key,
    int *P,
    int ascend,
    void *pBuffer);


cusparseStatus_t CUSPARSEAPI cusparseDsort_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const double *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDsort(
    cusparseHandle_t handle,
    int n,
    double *key,
    int *P,
    int ascend,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseHsort_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const __half *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseHsort(
    cusparseHandle_t handle,
    int n,
    __half *key_fp16,
    int *P,
    int ascend,
    void *pBuffer);





cusparseStatus_t CUSPARSEAPI cusparseHsortsign_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const __half *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseSsortsign_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const float *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseDsortsign_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const double *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

cusparseStatus_t CUSPARSEAPI cusparseIsortsign_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const int *key,
    const int *P,
    int ascend,
    size_t *pBufferSize);

//#if defined(__cplusplus)
cusparseStatus_t CUSPARSEAPI cusparseHsortsign(
    cusparseHandle_t handle,
    int n,
    __half *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);
//#endif

cusparseStatus_t CUSPARSEAPI cusparseSsortsign(
    cusparseHandle_t handle,
    int n,
    float *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDsortsign(
    cusparseHandle_t handle,
    int n,
    double *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseIsortsign(
    cusparseHandle_t handle,
    int n,
    int *key,
    int *P,
    int ascend,
    int *h_nnz_bucket0, /* host */
    void *pBuffer);

//----------------------------------------------


cusparseStatus_t CUSPARSEAPI cusparseDDcsrMv_hyb(
    cusparseHandle_t handle,
    cusparseOperation_t trans,
    int m,
    int n,
    int nnz,
    const double *alpha,
    const cusparseMatDescr_t descra,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const double *x,
    const double *beta,
    double *y);


/*
 * gtsv2Batch: cuThomas algorithm
 * gtsv3Batch: QR
 * gtsv4Batch: LU with partial pivoting
 */
cusparseStatus_t CUSPARSEAPI cusparseSgtsv2Batch(
    cusparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchCount);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv2Batch(
    cusparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchCount);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv2Batch(
    cusparseHandle_t handle,
    int n,
    cuComplex *dl,
    cuComplex  *d,
    cuComplex *du,
    cuComplex *x,
    int batchCount);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv2Batch(
    cusparseHandle_t handle,
    int n,
    cuDoubleComplex *dl,
    cuDoubleComplex  *d,
    cuDoubleComplex *du,
    cuDoubleComplex *x,
    int batchCount);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv3Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv3Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv3Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    const cuComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv3Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    const cuDoubleComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv3Batch(
    cusparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv3Batch(
    cusparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv3Batch(
    cusparseHandle_t handle,
    int n,
    cuComplex *dl,
    cuComplex  *d,
    cuComplex *du,
    cuComplex *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv3Batch(
    cusparseHandle_t handle,
    int n,
    cuDoubleComplex *dl,
    cuDoubleComplex  *d,
    cuDoubleComplex *du,
    cuDoubleComplex *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv4Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const float *dl,
    const float  *d,
    const float *du,
    const float *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv4Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const double *dl,
    const double  *d,
    const double *du,
    const double *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv4Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const cuComplex *dl,
    const cuComplex  *d,
    const cuComplex *du,
    const cuComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv4Batch_bufferSizeExt(
    cusparseHandle_t handle,
    int n,
    const cuDoubleComplex *dl,
    const cuDoubleComplex  *d,
    const cuDoubleComplex *du,
    const cuDoubleComplex *x,
    int batchSize,
    size_t *pBufferSizeInBytes);

cusparseStatus_t CUSPARSEAPI cusparseSgtsv4Batch(
    cusparseHandle_t handle,
    int n,
    float *dl,
    float  *d,
    float *du,
    float *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseDgtsv4Batch(
    cusparseHandle_t handle,
    int n,
    double *dl,
    double  *d,
    double *du,
    double *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseCgtsv4Batch(
    cusparseHandle_t handle,
    int n,
    cuComplex *dl,
    cuComplex  *d,
    cuComplex *du,
    cuComplex *x,
    int batchSize,
    void *pBuffer);

cusparseStatus_t CUSPARSEAPI cusparseZgtsv4Batch(
    cusparseHandle_t handle,
    int n,
    cuDoubleComplex *dl,
    cuDoubleComplex  *d,
    cuDoubleComplex *du,
    cuDoubleComplex *x,
    int batchSize,
    void *pBuffer);


#if defined(__cplusplus)
}
#endif /* __cplusplus */


#endif /* CUSPARSE_INTERNAL_H_ */

