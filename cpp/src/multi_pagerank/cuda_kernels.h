/*
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#ifndef __CUDA_HPP__
#define __CUDA_HPP__
#include <cuda_runtime.h>

#ifdef __cplusplus
extern LINKAGE size_t BinCouple2ASCIICuda(LOCINT *d_u, LOCINT *d_v, int64_t n, char **d_data, int verbose);
extern LINKAGE size_t ASCIICouple2BinCuda_entry(uint4 *d_data, size_t datalen, LOCINT **d_uout, LOCINT **d_vout, int verbose);
#endif

extern LINKAGE void init_cuda();
extern LINKAGE void copy_sort_samples(LOCINT *h_u, LOCINT *h_v, LOCINT nval, int64_t nvalmax);
extern LINKAGE void limits_cuda(int n, LOCINT *vmin, LOCINT *vmax, int *bbit, int *ebit);
extern LINKAGE void sort_cuda(LOCINT *h_u, LOCINT *h_v, int n, int bbit, int ebit);
extern LINKAGE void final_sort_cuda(LOCINT *h_u, LOCINT *h_v, int n);

extern LINKAGE void get_hist_cuda(int64_t nvals, int64_t *h_hist, LOCINT *h_probes, int np);
extern LINKAGE void getvals_cuda(LOCINT *h_u, LOCINT *h_v, int64_t n);
extern LINKAGE void setvals_cuda(LOCINT *h_u, LOCINT *h_v, int n);
extern LINKAGE void finalize_sort_cuda();

extern LINKAGE void generate_kron(int scale, int64_t ned,
				  REAL a, REAL b, REAL c,
				  LOCINT *d_i, LOCINT *d_j,
				  int64_t seed, int cnf, int perm);
extern LINKAGE int64_t keep_all_rows_cuda(LOCINT *u_h, LOCINT *v_h, int64_t ned, LOCINT **uout_d, LOCINT **vout_d);

extern LINKAGE int64_t remove_rows_cuda(LOCINT *u_h, LOCINT *v_h, int64_t ned, LOCINT **u_d, LOCINT **v_d);

extern LINKAGE void get_csr_multi_cuda(LOCINT *u_d, LOCINT *v_d, int64_t ned,
				       LOCINT *sep, int nsep,
				       LOCINT *nnz, LOCINT *nrows, LOCINT **roff_d,
				       LOCINT **rows_d, LOCINT **cols_d, REAL **vals_d);

extern LINKAGE void normalize_cols(LOCINT nnz, LOCINT *cols_d, REAL *vals_d, LOCINT N, LOCINT off, MPI_Comm COMM);
extern LINKAGE void normalize_cols_multi(LOCINT *nnz, LOCINT **cols_d, REAL **vals_d, LOCINT *last_row, int ncsr, MPI_Comm COMM);

extern LINKAGE void relabel_cuda_multi(LOCINT *lastrow_all, int ncsr, LOCINT *nrows,
				       LOCINT **rows_d, LOCINT *nnz, LOCINT **cols_d,
				       int64_t totToSend, LOCINT *rowsToSend_d, MPI_Comm COMM);

extern LINKAGE void generate_rhs(int64_t intColsNum, REAL *r);

#ifdef __cplusplus
extern LINKAGE void getSendElems(REAL *r, LOCINT *rowsToSend, int64_t totToSend, REAL *sendBuffer, cudaStream_t stream=0);
extern LINKAGE void setarray(REAL *arr, int64_t n, REAL val, cudaStream_t stream=0);
#endif

extern LINKAGE REAL reduce_cuda(REAL *v, int64_t n);
extern LINKAGE void reduce_cuda_async(REAL *v, int64_t n, REAL *sum_h, cudaStream_t stream);

extern LINKAGE void sort_csr(LOCINT nnz, LOCINT nrows, LOCINT *kthr, LOCINT **rind,
			     LOCINT **rows, LOCINT **cols, REAL **vals, LOCINT *koff);

extern LINKAGE void dump_csr(const char *fprefix, uint32_t nnz, uint32_t nrows, uint32_t *roff_d, LOCINT *rows_d, LOCINT *cols_d, REAL *vals_d);
extern LINKAGE void dump_nonlocal_rows(const char *fprefix, uint32_t nnz, uint32_t nrows, uint32_t *roff_d,
				       LOCINT *rows_d, LOCINT *cols_d, int64_t nReqCols, LOCINT *reqCols);

extern LINKAGE void cleanup_cuda();
extern LINKAGE void computeSpmvAcc(REAL c, LOCINT nrows, LOCINT *rows, LOCINT *roff,
				   LOCINT *cols, REAL *vals, REAL *rsrc, REAL *rdst,
				   LOCINT *koff, cudaStream_t stream);

extern LINKAGE void get_extdata_cuda(int ncsr, LOCINT *nnz, LOCINT **cols_d, LOCINT *lastrow_all,
				     int **recvNeigh, int *recvNum, int64_t **recvCnt, int64_t **recvOff, int64_t *totRecv,
				     int **sendNeigh, int *sendNum, int64_t **sendCnt, int64_t **sendOff, int64_t *totSend,
				     LOCINT **sendRows, MPI_Comm COMM);
extern LINKAGE void sequence(LOCINT n, LOCINT *vec, LOCINT init);
#endif
