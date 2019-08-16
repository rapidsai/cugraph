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

#include <stdio.h>
#include <stdlib.h>  
#include <stddef.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <utility>
#include <cstdint>
extern "C" {
#include "mmio.h"
}
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <library_types.h>
#include <thrust/host_vector.h>
#include <thrust/adjacent_difference.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>


#define CUDACHECK(cudaCall)         \
  do {                \
    cudaError_t e = (cudaCall);         \
    if(e != cudaSuccess) {          \
      fprintf(stderr, "CUDA Error (%s:%d): %s\n",   \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    }               \
  } while(0)


std::string getFileName(const std::string& s) {

   char sep = '/';

#ifdef _WIN32
   sep = '\\';
#endif

   size_t i = s.rfind(sep, s.length());
   if (i != std::string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}

template <typename T> 
void verbose_diff(std::vector<T> & v1, std::vector<T> & v2) {
  for (unsigned int i = 0; i < v1.size(); ++i)
  {
    if (v1[i] != v2[i])
    {
      std::cout << "[" << i <<"] : " << v1[i] << " -- ref = "<< v2[i]<<std::endl;
    }
  }
}

template <typename T> 
int eq(std::vector<T> & v1, std::vector<T> & v2) {
    if (v1 == v2)
        return 0; 
    else  {
        verbose_diff(v1,v2);
        return 1;
    }
}

template <typename T>
void printv(size_t n, T* vec, int offset) {
    thrust::device_ptr<T> dev_ptr(vec);
    std::cout.precision(15);
    std::cout << "sample size = "<< n << ", offset = "<< offset << std::endl;
    thrust::copy(dev_ptr+offset,dev_ptr+offset+n, std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
}

template <typename T_ELEM>
void ref_csr2csc (int m, int n, int nnz, const T_ELEM *csrVals, const int *csrRowptr, const int *csrColInd, T_ELEM *cscVals, int *cscRowind, int *cscColptr, int base=0){
    int i,j, row, col, index;
    int * counters;
    T_ELEM val;

    /* early return */
    if ((m <= 0) || (n <= 0) || (nnz <= 0)){
        return;
    }

    /* build compressed column pointers */
    memset(cscColptr, 0, (n+1)*sizeof(cscColptr[0]));
    cscColptr[0]=base;
    for (i=0; i<nnz; i++){
        cscColptr[1+csrColInd[i]-base]++;
    }
    for(i=0; i<n; i++){
        cscColptr[i+1]+=cscColptr[i];
    }

    /* expand row indecis and copy them and values into csc arrays according to permutation */
    counters = (int *)malloc(n*sizeof(counters[0]));
    memset(counters, 0, n*sizeof(counters[0]));
    for (i=0; i<m; i++){
        for (j=csrRowptr[i]; j<csrRowptr[i+1]; j++){
            row = i+base;
            col = csrColInd[j-base];

            index=cscColptr[col-base]-base+counters[col-base];
            counters[col-base]++;

            cscRowind[index]=row;

            if(csrVals!=NULL || cscVals!=NULL){
                val = csrVals[j-base];
                cscVals[index]  = val;
            }
        }
    }
    free(counters);
}

template <typename T> 
int transition_matrix_cpu(int n, int e, int *csrRowPtrA, int *csrColIndA, T *weight, T* is_leaf) 
//omp_set_num_threads(4);
//#pragma omp parallel
 {
    int j,row, row_size;
    //#pragma omp for
    for (row=0; row<n; row++) 
    {  
        row_size = csrRowPtrA[row+1] - csrRowPtrA[row];
        if (row_size == 0)
            is_leaf[row]=1.0;
        else
        {
            is_leaf[row]=0.0;
            for (j=csrRowPtrA[row]; j<csrRowPtrA[row+1]; j++) 
                weight[j] = 1.0/row_size; 
        } 
    }
    return 0;
}


/// Read matrix properties from Matrix Market file
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param t (Output) MM_typecode with matrix properties.
 *  @param m (Output) Number of matrix rows.
 *  @param n (Output) Number of matrix columns.
 *  @param nnz (Output) Number of non-zero matrix entries.
 *  @return Zero if properties were read successfully. Otherwise
 *  non-zero.
 */
template <typename IndexType_>
int mm_properties(FILE * f, int tg, MM_typecode * t,
		  IndexType_ * m, IndexType_ * n,
		  IndexType_ * nnz) {

  // Read matrix properties from file
  int mint, nint, nnzint;
  if(fseek(f,0,SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if(mm_read_banner(f,t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if(!mm_is_matrix(*t) || !mm_is_coordinate(*t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if(mm_read_mtx_crd_size(f,&mint,&nint,&nnzint)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if(!mm_is_pattern(*t) && !mm_is_real(*t) &&
     !mm_is_integer(*t) && !mm_is_complex(*t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }
  *m   = mint;
  *n   = nint;
  *nnz = nnzint;

  // Find total number of non-zero entries
  if(tg && !mm_is_general(*t)) {

    // Non-diagonal entries should be counted twice
    IndexType_ nnzOld = *nnz;
    *nnz *= 2;

    // Diagonal entries should not be double-counted
    int i; int st;
    for(i=0; i<nnzOld; ++i) {

      // Read matrix entry
      IndexType_ row, col;
      double rval, ival;
      if (mm_is_pattern(*t)) 
          st = fscanf(f, "%d %d\n", &row, &col);
      else if (mm_is_real(*t) || mm_is_integer(*t))
          st = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      else // Complex matrix
          st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
      if(ferror(f) || (st == EOF)) {
          fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i+1);
          return -1;
      }

      // Check if entry is diagonal
      if(row == col)
	--(*nnz);

    }
  }

  return 0;

}

/// Read Matrix Market file and convert to COO format matrix
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param nnz Number of non-zero matrix entries.
 *  @param cooRowInd (Output) Row indices for COO matrix. Should have
 *  at least nnz entries.
 *  @param cooColInd (Output) Column indices for COO matrix. Should
 *  have at least nnz entries.
 *  @param cooRVal (Output) Real component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @param cooIVal (Output) Imaginary component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @return Zero if matrix was read successfully. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
int mm_to_coo(FILE *f, int tg, IndexType_ nnz,
	      IndexType_ * cooRowInd, IndexType_ * cooColInd, 
	      ValueType_ * cooRVal  , ValueType_ * cooIVal) {
  
  // Read matrix properties from file
  MM_typecode t;
  int m, n, nnzOld;
  if(fseek(f,0,SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if(mm_read_banner(f,&t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if(!mm_is_matrix(t) || !mm_is_coordinate(t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if(mm_read_mtx_crd_size(f,&m,&n,&nnzOld)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if(!mm_is_pattern(t) && !mm_is_real(t) &&
     !mm_is_integer(t) && !mm_is_complex(t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }

  // Add each matrix entry in file to COO format matrix
  IndexType_ i;      // Entry index in Matrix Market file
  IndexType_ j = 0;  // Entry index in COO format matrix
  for(i=0;i<nnzOld;++i) {

    // Read entry from file
    int row, col;
    double rval, ival;
    int st;
    if (mm_is_pattern(t)) {
      st = fscanf(f, "%d %d\n", &row, &col);
      rval = 1.0;
      ival = 0.0;
    }
    else if (mm_is_real(t) || mm_is_integer(t)) {
      st = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      ival = 0.0;
    }
    else // Complex matrix
      st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
    if(ferror(f) || (st == EOF)) {
        fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i+1);
      return -1;
    }

    // Switch to 0-based indexing
    --row;
    --col;

    // Record entry
    cooRowInd[j] = row;
    cooColInd[j] = col;
    if(cooRVal != NULL)
      cooRVal[j] = rval;
    if(cooIVal != NULL)
      cooIVal[j] = ival;
    ++j;

    // Add symmetric complement of non-diagonal entries
    if(tg && !mm_is_general(t) && (row!=col)) {

      // Modify entry value if matrix is skew symmetric or Hermitian
      if(mm_is_skew(t)) {
	rval = -rval;
	ival = -ival;
      }
      else if(mm_is_hermitian(t)) {
	ival = -ival;
      }

      // Record entry
      cooRowInd[j] = col;
      cooColInd[j] = row;
      if(cooRVal != NULL)
	cooRVal[j] = rval;
      if(cooIVal != NULL)
	cooIVal[j] = ival;
      ++j;
      
    }
  }
  return 0;

}

/// Compare two tuples based on the element indexed by i
class lesser_tuple {
  const int i;
public:
  lesser_tuple(int _i) : i(_i) {}
  template<typename Tuple1, typename Tuple2>
  __host__ __device__
  bool operator()(const Tuple1 t1, const Tuple2 t2) {
    switch(i) {
    case 0:  return (thrust::get<0>(t1) < thrust::get<0>(t2));
    case 1:  return (thrust::get<1>(t1) < thrust::get<1>(t2));
    default: return (thrust::get<0>(t1) < thrust::get<0>(t2));
    }
    
  }
};

/// Sort entries in COO format matrix
/** Sort is stable.
 *
 *  @param nnz Number of non-zero matrix entries.
 *  @param sort_by_row Boolean indicating whether matrix entries
 *  will be sorted by row index or by column index.
 *  @param cooRowInd Row indices for COO matrix.
 *  @param cooColInd Column indices for COO matrix.
 *  @param cooRVal Real component for COO matrix entries. Ignored if
 *  null pointer.
 *  @param cooIVal Imaginary component COO matrix entries. Ignored if
 *  null pointer.
 */
template <typename IndexType_, typename ValueType_>
void coo_sort(IndexType_ nnz, int sort_by_row,
	      IndexType_ * cooRowInd,
	      IndexType_ * cooColInd, 
	      ValueType_ * cooRVal,
	      ValueType_ * cooIVal) {

  // Determine whether to sort by row or by column
  int i;
  if(sort_by_row == 0)
    i = 1;
  else
    i = 0;

  // Apply stable sort
  using namespace thrust;
  if((cooRVal==NULL) && (cooIVal==NULL))
    stable_sort(make_zip_iterator(make_tuple(cooRowInd,cooColInd)),
		make_zip_iterator(make_tuple(cooRowInd+nnz,cooColInd+nnz)),
		lesser_tuple(i));
  else if((cooRVal==NULL) && (cooIVal!=NULL))
    stable_sort(make_zip_iterator(make_tuple(cooRowInd,cooColInd,cooIVal)),
		make_zip_iterator(make_tuple(cooRowInd+nnz,cooColInd+nnz,cooIVal+nnz)),
		lesser_tuple(i));
  else if((cooRVal!=NULL) && (cooIVal==NULL))
    stable_sort(make_zip_iterator(make_tuple(cooRowInd,cooColInd,cooRVal)),
		make_zip_iterator(make_tuple(cooRowInd+nnz,cooColInd+nnz,cooRVal+nnz)),
		lesser_tuple(i));
  else
    stable_sort(make_zip_iterator(make_tuple(cooRowInd,cooColInd,cooRVal,cooIVal)),
		make_zip_iterator(make_tuple(cooRowInd+nnz,cooColInd+nnz,
					     cooRVal+nnz,cooIVal+nnz)),
		lesser_tuple(i));
}

/// Compress sorted list of indices
/** For use in converting COO format matrix to CSR or CSC format.
 *
 *  @param n Maximum index.
 *  @param nnz Number of non-zero matrix entries.
 *  @param sortedIndices Sorted list of indices (COO format).
 *  @param compressedIndices (Output) Compressed list of indices (CSR
 *  or CSC format). Should have at least n+1 entries.
 */
template <typename IndexType_>
void coo_compress(IndexType_ m, IndexType_ n, IndexType_ nnz,
		  const IndexType_ * __restrict__ sortedIndices,
		  IndexType_ * __restrict__ compressedIndices) {
  IndexType_ i;

  // Initialize everything to zero
  memset(compressedIndices, 0, (m+1)*sizeof(IndexType_));
  
  // Count number of elements per row
  for(i=0; i<nnz; ++i)
    ++(compressedIndices[sortedIndices[i]+1]);
  
  // Compute cumulative sum to obtain row offsets/pointers
  for(i=0; i<m; ++i)
    compressedIndices[i+1] += compressedIndices[i];

}

/// Convert COO format matrix to CSR format
/** On output, matrix entries in COO format matrix will be sorted
 *  (primarily by row index, secondarily by column index).
 *
 *  @param m Number of matrix rows.
 *  @param n Number of matrix columns.
 *  @param nnz Number of non-zero matrix entries.
 *  @param cooRowInd Row indices for COO matrix.
 *  @param cooColInd Column indices for COO matrix.
 *  @param cooRVal Real component of COO matrix entries. Ignored if
 *  null pointer.
 *  @param cooIVal Imaginary component of COO matrix entries. Ignored
 *  if null pointer.
 *  @param csrRowPtr Row pointers for CSR matrix. Should have at least
 *  n+1 entries.
 *  @param csrColInd Column indices for CSR matrix (identical to
 *  output of cooColInd). Should have at least nnz entries. Ignored if
 *  null pointer.
 *  @param csrRVal Real component of CSR matrix entries (identical to
 *  output of cooRVal). Should have at least nnz entries.  Ignored if
 *  null pointer.
 *  @param csrIVal Imaginary component of CSR matrix entries
 *  (identical to output of cooIVal). Should have at least nnz
 *  entries.  Ignored if null pointer.
 *  @return Zero if matrix was converted successfully. Otherwise
 *  non-zero.
 */
template <typename IndexType_, typename ValueType_>
int coo_to_csr(IndexType_ m, IndexType_ n, IndexType_ nnz,
		IndexType_ * __restrict__ cooRowInd,
		IndexType_ * __restrict__ cooColInd, 
		ValueType_ * __restrict__ cooRVal,
		ValueType_ * __restrict__ cooIVal,
		IndexType_ * __restrict__ csrRowPtr,
		IndexType_ * __restrict__ csrColInd,
		ValueType_ * __restrict__ csrRVal,
		ValueType_ * __restrict__ csrIVal) {

  // Convert COO to CSR matrix
  coo_sort(nnz, 0, cooRowInd, cooColInd, cooRVal, cooIVal);
  coo_sort(nnz, 1, cooRowInd, cooColInd, cooRVal, cooIVal);
  coo_compress(m, n, nnz, cooRowInd, csrRowPtr);

  // Copy arrays
  if(csrColInd!=NULL)
    memcpy(csrColInd, cooColInd, nnz*sizeof(IndexType_));
  if((cooRVal!=NULL) && (csrRVal!=NULL))
    memcpy(csrRVal, cooRVal, nnz*sizeof(ValueType_));
  if((cooIVal!=NULL) && (csrIVal!=NULL))
    memcpy(csrIVal, cooIVal, nnz*sizeof(ValueType_));

  return 0;

}

