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

#include <nvgraph.h>
#include <nvgraph_cusparse.hxx>
#include <cnmem_shared_ptr.hxx>

namespace nvgraph{
  void csr2coo( const int *csrSortedRowPtr,
                int nnz, int m,
                int *cooRowInd,
                cusparseIndexBase_t idxBase);
  void coo2csr( const int *cooRowInd,
                int nnz, int m,
                int *csrSortedRowPtr,
                cusparseIndexBase_t idxBase );

  void csr2csc( int m, int n, int nnz,
                const void *csrVal, const int *csrRowPtr, const int *csrColInd,
                void *cscVal, int *cscRowInd, int *cscColPtr,
                cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                cudaDataType_t *dataType);
  void csc2csr( int m, int n, int nnz,
                const void *cscVal, const int *cscRowInd, const int *cscColPtr,
                void *csrVal, int *csrRowPtr, int *csrColInd,
                cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                cudaDataType_t *dataType);

  void csr2cscP( int m, int n, int nnz,
                 const int *csrRowPtr, const int *csrColInd,
                 int *cscRowInd, int *cscColPtr, int *p, cusparseIndexBase_t idxBase);


  void cooSortBySource(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);
  void cooSortByDestination(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);

  void coos2csc(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);
  void cood2csr(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);
  void coou2csr(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);
  void coou2csc(int m, int n, int nnz,
            const void *srcVal, const int *srcRowInd, const int *srcColInd,
            void *dstVal, int *dstRowInd, int *dstColInd,
            cusparseIndexBase_t idxBase, cudaDataType_t *dataType);

  ////////////////////////// Utility functions //////////////////////////
  void createIdentityPermutation(int n, int *p);
  void gthrX(int nnz, const void *y, void *xVal, const int *xInd,
    cusparseIndexBase_t idxBase, cudaDataType_t *dataType);

  void cooSortBufferSize(int m, int n, int nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes);
  void cooGetSourcePermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer);
  void cooGetDestinationPermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer);

  void csr2csc2BufferSize(int m, int n, int nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSize);
  void csr2csc2(int m, int n, int nnz,
    const int *csrRowPtr, const int *csrColInd,
    int *cscRowInd, int *cscColPtr, int *p, void *pBuffer,
    cusparseIndexBase_t idxBase);

} //end nvgraph namespace
