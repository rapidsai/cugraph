#pragma once

#include <stdio.h>
extern "C" {
#include "mmio.h"
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

template <typename IndexType_, typename ValueType_>
void sort(IndexType_ *col_idx, ValueType_ *a, IndexType_ start, IndexType_ end)
{
  IndexType_ i, j, it;
  ValueType_ dt;

  for (i=end-1; i>start; i--)
    for(j=start; j<i; j++)
      if (col_idx[j] > col_idx[j+1]){

    if (a){
      dt=a[j]; 
      a[j]=a[j+1]; 
      a[j+1]=dt;
        }
    it=col_idx[j]; 
    col_idx[j]=col_idx[j+1]; 
    col_idx[j+1]=it;
      
      }
}

template <typename IndexType_, typename ValueType_>
void coo2csr(IndexType_ n, IndexType_ nz, ValueType_ *a, IndexType_ *i_idx, IndexType_ *j_idx,
         ValueType_ *csr_a, IndexType_ *col_idx, IndexType_ *row_start)
{
  IndexType_ i, l;

  for (i=0; i<=n; i++) row_start[i] = 0;

  /* determine row lengths */
  for (i=0; i<nz; i++) row_start[i_idx[i]+1]++;


  for (i=0; i<n; i++) row_start[i+1] += row_start[i];


  /* go through the structure  once more. Fill in output matrix. */
  for (l=0; l<nz; l++){
    i = row_start[i_idx[l]];
    csr_a[i] = a[l];
    col_idx[i] = j_idx[l];
    row_start[i_idx[l]]++;
  }

  /* shift back row_start */
  for (i=n; i>0; i--) row_start[i] = row_start[i-1];

  row_start[0] = 0;

  for (i=0; i<n; i++){
    sort (col_idx, csr_a, row_start[i], row_start[i+1]);
  }

}
