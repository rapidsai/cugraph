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

#include "include/nvgraph_convert.hxx"
#include "include/nvgraph_error.hxx"



 namespace nvgraph{
    void csr2coo( const int *csrSortedRowPtr,
                  int nnz, int m, int *cooRowInd, cusparseIndexBase_t idxBase){
      CHECK_CUSPARSE( cusparseXcsr2coo(  Cusparse::get_handle(),
                                         csrSortedRowPtr, nnz, m, cooRowInd, idxBase ));
    }
    void coo2csr( const int *cooRowInd,
                  int nnz, int m, int *csrSortedRowPtr, cusparseIndexBase_t idxBase){
      CHECK_CUSPARSE( cusparseXcoo2csr( Cusparse::get_handle(),
                                        cooRowInd, nnz, m, csrSortedRowPtr, idxBase ));
    }


    void csr2csc( int m, int n, int nnz,
                  const void *csrVal, const int *csrRowPtr, const int *csrColInd,
                  void *cscVal, int *cscRowInd, int *cscColPtr,
                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                  cudaDataType_t *dataType){
      CHECK_CUSPARSE( cusparseCsr2cscEx( Cusparse::get_handle(),
                                         m, n, nnz,
                                         csrVal, *dataType, csrRowPtr, csrColInd,
                                         cscVal, *dataType, cscRowInd, cscColPtr,
                                         copyValues, idxBase, *dataType ));
    }
    void csc2csr( int m, int n, int nnz,
                  const void *cscVal, const int *cscRowInd, const int *cscColPtr,
                  void *csrVal, int *csrRowPtr, int *csrColInd,
                  cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                  cudaDataType_t *dataType){
      CHECK_CUSPARSE( cusparseCsr2cscEx( Cusparse::get_handle(),
                                         m, n, nnz,
                                         cscVal, *dataType, cscColPtr, cscRowInd,
                                         csrVal, *dataType, csrColInd, csrRowPtr,
                                         copyValues, idxBase, *dataType ));
    }


    void csr2cscP( int m, int n, int nnz,
                 const int *csrRowPtr, const int *csrColInd,
                 int *cscRowInd, int *cscColPtr, int *p,
                 cusparseIndexBase_t idxBase){

      std::shared_ptr<char> pBuffer;

      // Step 1: Allocate buffer
      size_t pBufferSizeInBytes = 0;
      csr2csc2BufferSize(m, n, nnz, csrRowPtr, csrColInd, &pBufferSizeInBytes);
      pBuffer = allocateDevice<char>(pBufferSizeInBytes, NULL);
      // Step 2: Setup permutation vector P to identity
      createIdentityPermutation(nnz, p);
      // Step 3: Convert and get perumation array
      csr2csc2(m, n, nnz, csrRowPtr, csrColInd, cscRowInd, cscColPtr, p, pBuffer.get(), idxBase);
    }


    void cooSortByDestination(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowInd, int *dstColInd,
                cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      size_t pBufferSizeInBytes = 0;
      std::shared_ptr<char> pBuffer;
      std::shared_ptr<int> P; // permutation array

      // step 0: copy src to dst
      if(dstRowInd!=srcRowInd)
        CHECK_CUDA( cudaMemcpy(dstRowInd, srcRowInd, nnz*sizeof(int), cudaMemcpyDefault) );
      if(dstColInd!=srcColInd)
        CHECK_CUDA( cudaMemcpy(dstColInd, srcColInd, nnz*sizeof(int), cudaMemcpyDefault) );
      // step 1: allocate buffer (needed for cooSortByRow)
      cooSortBufferSize(m, n, nnz, dstRowInd, dstColInd, &pBufferSizeInBytes);
      pBuffer = allocateDevice<char>(pBufferSizeInBytes, NULL);
      // step 2: setup permutation vector P to identity
      P = allocateDevice<int>(nnz, NULL);
      createIdentityPermutation(nnz, P.get());
      // step 3: sort COO format by Row
      cooGetDestinationPermutation(m, n, nnz, dstRowInd, dstColInd, P.get(), pBuffer.get());
      // step 4: gather sorted cooVals
      gthrX(nnz, srcVal, dstVal, P.get(), idxBase, dataType);
    }
    void cooSortBySource(int m, int n, int nnz,
                const void *srcVal, const int *srcRowInd, const int *srcColInd,
                void *dstVal, int *dstRowInd, int *dstColInd,
                cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      size_t pBufferSizeInBytes = 0;
      std::shared_ptr<char> pBuffer;
      std::shared_ptr<int> P; // permutation array

      // step 0: copy src to dst
      CHECK_CUDA( cudaMemcpy(dstRowInd, srcRowInd, nnz*sizeof(int), cudaMemcpyDefault) );
      CHECK_CUDA( cudaMemcpy(dstColInd, srcColInd, nnz*sizeof(int), cudaMemcpyDefault) );
      // step 1: allocate buffer (needed for cooSortByRow)
      cooSortBufferSize(m, n, nnz, dstRowInd, dstColInd, &pBufferSizeInBytes);
      pBuffer = allocateDevice<char>(pBufferSizeInBytes, NULL);
      // step 2: setup permutation vector P to identity
      P = allocateDevice<int>(nnz, NULL);
      createIdentityPermutation(nnz, P.get());
      // step 3: sort COO format by Row
      cooGetSourcePermutation(m, n, nnz, dstRowInd, dstColInd, P.get(), pBuffer.get());
      // step 4: gather sorted cooVals
      gthrX(nnz, srcVal, dstVal, P.get(), idxBase, dataType);
    }

    void coos2csc(int m, int n, int nnz,
              const void *srcVal, const int *srcRowInd, const int *srcColInd,
              void *dstVal, int *dstRowInd, int *dstColPtr,
              cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      // coos -> cood -> csc
      std::shared_ptr<int> tmp = allocateDevice<int>(nnz, NULL);
      cooSortByDestination(m, n, nnz, srcVal, srcRowInd, srcColInd, dstVal, dstRowInd, tmp.get(), idxBase, dataType);
      coo2csr(tmp.get(), nnz, m, dstColPtr, idxBase);
    }
    void cood2csr(int m, int n, int nnz,
              const void *srcVal, const int *srcRowInd, const int *srcColInd,
              void *dstVal, int *dstRowPtr, int *dstColInd,
              cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      // cood -> coos -> csr
      std::shared_ptr<int> tmp = allocateDevice<int>(nnz, NULL);
      cooSortBySource(m, n, nnz, srcVal, srcRowInd, srcColInd, dstVal, tmp.get(), dstColInd, idxBase, dataType);
      coo2csr(tmp.get(), nnz, m, dstRowPtr, idxBase);
    }
    void coou2csr(int m, int n, int nnz,
              const void *srcVal, const int *srcRowInd, const int *srcColInd,
              void *dstVal, int *dstRowPtr, int *dstColInd,
              cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      cood2csr(m, n, nnz,
              srcVal, srcRowInd, srcColInd,
              dstVal, dstRowPtr, dstColInd,
              idxBase, dataType);
    }
    void coou2csc(int m, int n, int nnz,
              const void *srcVal, const int *srcRowInd, const int *srcColInd,
              void *dstVal, int *dstRowInd, int *dstColPtr,
              cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      coos2csc(m, n, nnz,
              srcVal, srcRowInd, srcColInd,
              dstVal, dstRowInd, dstColPtr,
              idxBase, dataType);
    }

    ////////////////////////// Utility functions //////////////////////////
    void createIdentityPermutation(int n, int *p){
        CHECK_CUSPARSE( cusparseCreateIdentityPermutation(Cusparse::get_handle(), n, p) );
    }

    void gthrX( int nnz, const void *y, void *xVal, const int *xInd,
                cusparseIndexBase_t idxBase, cudaDataType_t *dataType){
      if(*dataType==CUDA_R_32F){
        CHECK_CUSPARSE( cusparseSgthr(Cusparse::get_handle(), nnz, (float*)y, (float*)xVal, xInd, idxBase ));
      } else if(*dataType==CUDA_R_64F) {
        CHECK_CUSPARSE( cusparseDgthr(Cusparse::get_handle(), nnz, (double*)y, (double*)xVal, xInd, idxBase ));
      }
    }


    void cooSortBufferSize(int m, int n, int nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes) {
        CHECK_CUSPARSE( cusparseXcoosort_bufferSizeExt( Cusparse::get_handle(),
                                                        m, n, nnz,
                                                        cooRows, cooCols, pBufferSizeInBytes ));
    }
    void cooGetSourcePermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer) {
        CHECK_CUSPARSE( cusparseXcoosortByRow( Cusparse::get_handle(),
                                                m, n, nnz,
                                                cooRows, cooCols, p, pBuffer ));
    }
    void cooGetDestinationPermutation(int m, int n, int nnz, int *cooRows, int *cooCols, int *p, void *pBuffer) {
        CHECK_CUSPARSE( cusparseXcoosortByColumn( Cusparse::get_handle(),
                                                  m, n, nnz,
                                                  cooRows, cooCols, p, pBuffer ));
    }

    void csr2csc2BufferSize(int m, int n, int nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSize){
        CHECK_CUSPARSE( cusparseXcsr2csc2_bufferSizeExt( Cusparse::get_handle(),
                                                         m, n, nnz,
                                                         csrRowPtr, csrColInd, pBufferSize ));
    }
    void csr2csc2(int m, int n, int nnz,
                  const int *csrRowPtr, const int *csrColInd,
                  int *cscRowInd, int *cscColPtr, int *p, void *pBuffer,
                  cusparseIndexBase_t idxBase){
        cusparseMatDescr_t descrA;
        CHECK_CUSPARSE( cusparseCreateMatDescr(&descrA) );
        CHECK_CUSPARSE( cusparseSetMatIndexBase(descrA, idxBase) );
        CHECK_CUSPARSE( cusparseXcsr2csc2( Cusparse::get_handle(),
                                           m, n, nnz,
                                           descrA,
                                           csrRowPtr, csrColInd,
                                           cscColPtr, cscRowInd, p,
                                           pBuffer ));
        CHECK_CUSPARSE( cusparseDestroyMatDescr(descrA) );
    }

} //end namespace nvgraph
