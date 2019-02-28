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
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <getopt.h>
#include <limits.h>
#include <float.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cudf.h>
//#include <cuda_profiler_api.h>

#ifndef __APPLE__
#include <fcntl.h> // for posix_fadvise()
#endif

#ifdef __cplusplus
#define __STDC_FORMAT_MACROS 1
#endif
#include <inttypes.h>

#include "global.h"
#include "phsort.h"
#include "utils.h"
#include "cuda_kernels.h"

#include "utilities/error_utils.h"
#include "graph_utils.cuh"
#include "multi_pagerank.cuh"

#define CHUNK_SIZE	(1024*1024)
#define MAX_LINE	(256)

#define RHS_RANDOM	(0)
#define RHS_CONSTANT	(1)
#define RHS_FILE	(2)

#if LOCINT_SIZE == 8 
#define LOCINT_MAX_CHAR	(20)
#else
#define LOCINT_MAX_CHAR	(11)
#endif

// ~200 Mb buffer
// If smaller the hdd/ssd cache may have an 
// effect even with --no-rcache
#if LOCINT_SIZE == 8
#define IOINT_NUM	(1024*1024*10)
#else
#define IOINT_NUM	(1024*1024*10*2)
#endif
char IOBUF[IOINT_NUM*(LOCINT_MAX_CHAR+1)];

//#define MPR_VERBOSE 1

typedef struct {
	int	code;
	REAL	fp;
	char	*str;
} rhsv_t;

typedef struct {
	LOCINT	*u;
	LOCINT	*v;
	int64_t	ned;
} elist_t;


static spmat_t *createSpmat(int n) {

	spmat_t *m = (spmat_t *)Malloc(sizeof(*m));

	m->firstRow = -1;
	m->lastRow = -1;
	m->intColsNum = 0;
	m->extColsNum = 0;
	m->totToSend = 0;

	m->sendNum = 0;
	m->recvNum = 0;

	m->sendNeigh = NULL;
	m->recvNeigh = NULL;
	m->sendCnts = NULL;
	m->recvCnts = NULL;
	m->sendOffs = NULL;
	m->recvOffs = NULL;

	m->rowsToSend_d = NULL;
#ifdef USE_MAPPED_SENDBUF
	m->sendBuffer_m = NULL;
#else
	m->sendBuffer_d = NULL;
#endif
	m->sendBuffer = NULL;
	m->recvBuffer = NULL;

	m->ncsr = n;
	m->nnz = (LOCINT *)Malloc(n*sizeof(LOCINT));
	m->nrows = (LOCINT *)Malloc(n*sizeof(LOCINT));

	m->roff_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->rows_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->cols_d = (LOCINT **)Malloc(n*sizeof(LOCINT *));
	m->vals_d = (REAL **)Malloc(n*sizeof(REAL *));

	memset(m->roff_d, 0, n*sizeof(LOCINT *));
	memset(m->rows_d, 0, n*sizeof(LOCINT *));
	memset(m->cols_d, 0, n*sizeof(LOCINT *));
	memset(m->vals_d, 0, n*sizeof(REAL *));

	m->kthr = (LOCINT (*)[2])Malloc(n*sizeof(LOCINT[2]));
	m->koff = (LOCINT (*)[2])Malloc(n*sizeof(LOCINT[2]));

	return m;
}

static void destroySpmat(spmat_t *m) {

	if (m->sendNeigh) free(m->sendNeigh);
	if (m->recvNeigh) free(m->recvNeigh);
	if (m->sendCnts) free(m->sendCnts);
	if (m->recvCnts) free(m->recvCnts);
	if (m->sendOffs) free(m->sendOffs);
	if (m->recvOffs) free(m->recvOffs);

	if (m->rowsToSend_d) CHECK_CUDA(cudaFree(m->rowsToSend_d));
#ifndef USE_MAPPED_SENDBUF
	if (m->sendBuffer_d) CHECK_CUDA(cudaFree(m->sendBuffer_d));
#endif
	if (m->sendBuffer) {
		CHECK_CUDA(cudaHostUnregister(m->sendBuffer));
		free(m->sendBuffer);
	}
	if (m->recvBuffer) {
		CHECK_CUDA(cudaHostUnregister(m->recvBuffer));
		free(m->recvBuffer);
	}

	for(int i = 0; i < m->ncsr; i++) {
		if (m->roff_d[i]) CHECK_CUDA(cudaFree(m->roff_d[i]));
		if (m->rows_d[i]) CHECK_CUDA(cudaFree(m->rows_d[i]));
		if (m->cols_d[i]) CHECK_CUDA(cudaFree(m->cols_d[i]));
		if (m->vals_d[i]) CHECK_CUDA(cudaFree(m->vals_d[i]));
	}
	free(m->roff_d);
	free(m->rows_d);
	free(m->cols_d);
	free(m->vals_d);

	free(m->kthr);
	free(m->koff);

	return;
}

static void check_row_overlap(LOCINT first_row, LOCINT last_row, int *exchup, int *exchdown) {

	int		rank, ntask, nr;
	LOCINT 		prevrow, nextrow;
	MPI_Request	request[2];
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	nr = 0;
	prevrow = nextrow = -1;

	if (rank > 0)	    MPI_Irecv(&prevrow, 1, LOCINT_MPI, rank-1, TAG(rank-1), MPI_COMM_WORLD, &request[nr++]);
	if (rank < ntask-1) MPI_Irecv(&nextrow, 1, LOCINT_MPI, rank+1, TAG(rank+1), MPI_COMM_WORLD, &request[nr++]);

	if (rank < ntask-1) MPI_Send(&last_row,  1, LOCINT_MPI, rank+1, TAG(rank), MPI_COMM_WORLD);
	if (rank > 0)	    MPI_Send(&first_row, 1, LOCINT_MPI, rank-1, TAG(rank), MPI_COMM_WORLD);

        MPI_Waitall(nr, request, MPI_STATUS_IGNORE);

	*exchup   = (prevrow == first_row);
	*exchdown = (nextrow == last_row);

	return;
}
	
void adjust_row_range(size_t N, LOCINT *first_row, LOCINT *last_row) {

	int	rank, ntask;
	
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntask);
    
	MPI_Sendrecv(first_row, 1, LOCINT_MPI, (rank+ntask-1)%ntask, rank,
		     last_row, 1, LOCINT_MPI, (rank+1)%ntask, (rank+1)%ntask,
    		     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	*last_row = ((rank < ntask-1) ? *last_row : N-1);
	if (rank == 0) *first_row = 0;

	return;	
}

static void postIrecvs(REAL *recvBuffer, int recvNum, int *recvNeigh, int64_t *recvOffs, int64_t *recvCnts,
		       MPI_Request *request, MPI_Comm COMM) {

	int ntask;
	MPI_Comm_size(COMM, &ntask);

	for(int i = 0; i < recvNum; i++) {
		if (recvCnts[i]) {
			MPI_Irecv(recvBuffer + recvOffs[i], recvCnts[i],
				  REAL_MPI, recvNeigh[i], TAG(recvNeigh[i]),
				  COMM, request + i);
		}
	}
	return;
}

static void exchangeDataSingle(REAL *sendBuffer, int sendNeigh, int64_t sendCnts,
			       REAL *recvBuffer, int recvNeigh, int64_t recvCnts,
			       MPI_Request *request, MPI_Comm COMM) {

	int rank, ntask;
	
	MPI_Comm_rank(COMM, &rank);
	MPI_Comm_size(COMM, &ntask);

	if (sendCnts) {
		MPI_Send(sendBuffer, sendCnts,
			 REAL_MPI, sendNeigh, TAG(rank),
			 COMM);
	}
	MPI_Wait(request, MPI_STATUS_IGNORE);
	if (recvCnts) {
		MPI_Irecv(recvBuffer, recvCnts,
			  REAL_MPI, recvNeigh, TAG(recvNeigh),
			  COMM, request);
	}
	return;
}

static inline void cancelReqs(MPI_Request *request, int n) {

        int i;
        for(i = 0; i < n; i++) {
		int flag;
		MPI_Test(request+i, &flag, MPI_STATUS_IGNORE);
		if (!flag) {
			MPI_Cancel(request+i);
			MPI_Wait(request+i, MPI_STATUSES_IGNORE);
		}
	}
        return;
}

static void coo2csr(size_t N, spmat_t *m, elist_t *ein) {

	double tg=0, tdr=0;

	double	min_rt, max_rt;
	double	min_rr, max_rr;
	
	int64_t ned, rbytes=0;

	LOCINT	*u=NULL,   *v=NULL;

	int	EXCH_UP, EXCH_DOWN;
	int	rank, ntask;

	LOCINT	*lastrow_all=NULL;
	LOCINT	*rowsToSend=NULL;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);


	u = ein->u;
	v = ein->v;
	ned = ein->ned;
	
	CHECK_CUDA(cudaMemcpy(&m->firstRow, &u[0], sizeof(m->firstRow), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(&m->lastRow, &u[ned-1], sizeof(m->lastRow), cudaMemcpyDeviceToHost));


	// quick sanity check on (sorted) edges received as input
	check_row_overlap(m->firstRow, m->lastRow, &EXCH_UP, &EXCH_DOWN);
	if (EXCH_UP || EXCH_DOWN) {
		fprintf(stderr, "Processor %d shares rows with neighboring processors!!!\n", rank);
		exit(EXIT_FAILURE);
	}

	//cudaProfilerStart();
	tg = MPI_Wtime();

	// temp data using pool
	LOCINT	*u_d=NULL, *v_d=NULL; // alloc-ed in <remove/keep>_rows_cuda() and
				                // dealloc-ed in get_csr_multi_cuda()

	// remove rows smaller than 1
	//ned = remove_rows_cuda(u, v, ned, &u_d, &v_d);
	
	// simply keep the same input 
	ned = keep_all_rows_cuda(u, v, ned, &u_d, &v_d);

	// quick sanity check
	check_row_overlap(m->firstRow, m->lastRow, &EXCH_UP, &EXCH_DOWN);
	if (EXCH_UP || EXCH_DOWN) {
		fprintf(stderr, "Processor %d shares rows with neighboring processors!!!\n", rank);
		exit(EXIT_FAILURE);
	}
	
	// expand [m->firstRow, m->lastRow] ranges in order to partition [0, N-1]
	// (empty rows outside any [m->firstRow, m->lastRow] range may appear as
	// columns in other rows
	//adjust_row_range(N, &m->firstRow, &m->lastRow);
	
	m->intColsNum = m->lastRow - m->firstRow + 1;


	lastrow_all = (LOCINT *)Malloc(ntask*sizeof(*lastrow_all));
	MPI_Allgather(&m->lastRow, 1, LOCINT_MPI, lastrow_all, 1, LOCINT_MPI, MPI_COMM_WORLD);

	get_csr_multi_cuda(u_d, v_d, ned, lastrow_all, ntask,
			   m->nnz, m->nrows, m->roff_d,
			   m->rows_d, m->cols_d, m->vals_d);
	normalize_cols_multi(m->nnz, m->cols_d, m->vals_d, lastrow_all, m->ncsr, MPI_COMM_WORLD);
	for(int i = 0; i < m->ncsr; i++) {
#if 1
		m->kthr[i][0] = 1024;
		m->kthr[i][1] = 16; // good values: 7, 16
		if (m->kthr[i][1] > m->kthr[i][0]) {
			fprintf(stderr, "[%d] Error with thresholds!!!\n", rank);
			exit(EXIT_FAILURE);
		}
		// sort CSR rows in descending order of length
		sort_csr(m->nnz[i], m->nrows[i], m->kthr[i],
			 m->rows_d+i, m->roff_d+i, m->cols_d+i,
			 m->vals_d+i, m->koff[i]);
#else
		m->koff[i][0] = 0;
		m->koff[i][1] = m->nrows[i];
#endif
	}
	get_extdata_cuda(m->ncsr, m->nnz, m->cols_d, lastrow_all,
			 &m->recvNeigh, &m->recvNum, &m->recvCnts, &m->recvOffs, &m->extColsNum, 
			 &m->sendNeigh, &m->sendNum, &m->sendCnts, &m->sendOffs, &m->totToSend,
			 &rowsToSend, MPI_COMM_WORLD);
	if (m->extColsNum) {
		m->recvBuffer = (REAL *)Malloc(m->extColsNum*sizeof(*m->recvBuffer));
		CHECK_CUDA(cudaHostRegister(m->recvBuffer, m->extColsNum*sizeof(*m->recvBuffer), cudaHostRegisterMapped));
	}
	if (m->sendNum) {
		m->totToSend = m->sendOffs[m->sendNum-1] + m->sendCnts[m->sendNum-1]; // redundant
		m->sendBuffer = (REAL *)Malloc(m->totToSend*sizeof(*m->sendBuffer));
		CHECK_CUDA(cudaHostRegister(m->sendBuffer, m->totToSend*sizeof(*m->sendBuffer), cudaHostRegisterMapped));
#ifdef USE_MAPPED_SENDBUF
		CHECK_CUDA(cudaHostGetDevicePointer((void **)&(m->sendBuffer_m), m->sendBuffer, 0) );
#else
		CHECK_CUDA(cudaMalloc(&m->sendBuffer_d, m->totToSend*sizeof(*m->sendBuffer_d)));
#endif
		CHECK_CUDA(cudaMalloc(&m->rowsToSend_d, m->totToSend*sizeof(*m->rowsToSend_d)));
		CHECK_CUDA(cudaMemcpy(m->rowsToSend_d, rowsToSend, m->totToSend*sizeof(*m->rowsToSend_d), cudaMemcpyHostToDevice));
	}

	relabel_cuda_multi(lastrow_all,
			   m->ncsr,
			   (LOCINT *)m->nrows, m->rows_d,
			   (LOCINT *)m->nnz, m->cols_d,
			   m->totToSend, m->rowsToSend_d, MPI_COMM_WORLD);
	tg = MPI_Wtime()-tg;

	if (!ein) {
		MPI_Reduce(&tdr, &min_rt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&tdr, &max_rt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		double rr = rbytes/(1024.0*1024.0)/tdr;
		MPI_Reduce(&rr, &min_rr, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&rr, &max_rr, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

		MPI_Reduce(rank ? &rbytes : MPI_IN_PLACE, &rbytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	}
#if	MPR_VERBOSE
	//cudaProfilerStop();
	if (0 == rank) {
		if (!ein) {
			printf("\tread  time: %.4lf secs, %.2lf Mbytes/sec\n", tr, rbytes/(1024.0*1024.0)/tr);
			printf("\t\tmin/max disk read time: %.4lf/%.4lf secs, %.2lf/%.2lf Mbytes/sec\n", min_rt, max_rt, min_rr, max_rr);
		}
		printf("\tgen   time: %.4lf secs\n", tg);
		fflush(stdout);
	}
#endif
	// sanity check (untimed)
	if (m->totToSend) {
		CHECK_CUDA(cudaMemcpy(rowsToSend, m->rowsToSend_d, m->totToSend*sizeof(*rowsToSend), cudaMemcpyDeviceToHost));
		for(int i = 0; i < m->sendNum; i++) {
			for(int j = 0; j < m->sendCnts[i]; j++) {
				if (rowsToSend[m->sendOffs[i]+j] < 0 || rowsToSend[m->sendOffs[i]+j] > m->intColsNum) {
					fprintf(stderr, "[%d] error: rowsToSend[%" PRId64 "] (%d-th row to send to proc %d) = %" PRILOC " > %" PRId64 "\n",
						rank, m->sendOffs[i]+j, j, m->sendNeigh[i], rowsToSend[m->sendOffs[i]+j], m->intColsNum);
					exit(EXIT_FAILURE);
				}
			}
		}
	}
	if (rowsToSend) free(rowsToSend);
	if (lastrow_all) free(lastrow_all);

	return;
}


static void pagerank_solver(int numIter, REAL c, REAL a, rhsv_t rval, spmat_t *m, REAL *pr) {

	int		i, rank, ntask;
	float		evt;
	double		tg=0, tc=0, t=0;
	double		tspmv[2]={0,0}, tmpi[2]={0,0}, td2h[2]={0,0}, th2d[2]={0,0};
	REAL		*r_h=NULL, *r_d[2]={NULL,NULL}, sum;
	MPI_Request	*reqs=NULL;

	cudaStream_t	stream[2];
	cudaEvent_t	event[4];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);

	CHECK_CUDA(cudaStreamCreate(stream));
	CHECK_CUDA(cudaStreamCreate(stream+1));

	CHECK_CUDA(cudaEventCreate(event));
	CHECK_CUDA(cudaEventCreate(event+1));
	CHECK_CUDA(cudaEventCreate(event+2));
	CHECK_CUDA(cudaEventCreate(event+3));

	CHECK_CUDA(cudaMalloc(r_d+0, (m->intColsNum + MAX(1,m->extColsNum))*sizeof(*r_d[0])));
	CHECK_CUDA(cudaMalloc(r_d+1, (m->intColsNum + MAX(1,m->extColsNum))*sizeof(*r_d[1])));

	if (m->recvNum)
		reqs = (MPI_Request *)Malloc(m->recvNum*sizeof(MPI_Request));

	MPI_Barrier(MPI_COMM_WORLD);
	tg = MPI_Wtime();

	// generate RHS
	if (rval.code == RHS_RANDOM) {
		generate_rhs(m->intColsNum, r_d[0]);
	} 
	else { //constant
		setarray(r_d[0], m->intColsNum, rval.fp); 
		CHECK_CUDA(cudaDeviceSynchronize());
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tg = MPI_Wtime()-tg;
#if 1
	postIrecvs(m->recvBuffer, m->recvNum, m->recvNeigh, m->recvOffs, m->recvCnts,
		   reqs, MPI_COMM_WORLD);
#endif
	//MPI_Pcontrol(1);
	
	// dangling node experiment
	/*
		REAL *bb=0, *d_leaf_vector=0;
		ALLOC_MANAGED_TRY((void**)&d_leaf_vector, sizeof(REAL) * m->intColsNum, 0);
		ALLOC_MANAGED_TRY ((void**)&bb,    sizeof(REAL) * m->intColsNum, 0);
		REAL randomProbability =  static_cast<REAL>( 1.0/m->intColsNum);
  		cugraph::fill(m->intColsNum, bb, randomProbability);
  		int nn = m->intColsNum;
		cugraph::flag_leaves2(nn, m->nnz[rank], m->cols_d[rank], d_leaf_vector);
		cugraph::update_dangling_nodes(m->intColsNum, d_leaf_vector, c);
	*/
	tc = MPI_Wtime();
	START_RANGE("SPMV_ALL_ITER", 1);
	for(i = 0; i < numIter; i++) {

		int s = i&1;
		int d = s^1;

		START_RANGE("BEFORE_SPMV_LOOP", 0);
#ifdef ASYNC_RED
		reduce_cuda_async(r_d[s], m->intColsNum, &sum, stream[0]);
#else
		sum = reduce_cuda(r_d[s], m->intColsNum);
#endif		
		if (m->totToSend) {
			CHECK_CUDA(cudaEventRecord(event[2], stream[1]));
#ifdef USE_MAPPED_SENDBUF
			getSendElems(r_d[s], m->rowsToSend_d, m->totToSend, m->sendBuffer_m, stream[1]);
#else
			getSendElems(r_d[s], m->rowsToSend_d, m->totToSend, m->sendBuffer_d, stream[1]);
			CHECK_CUDA(cudaMemcpyAsync(m->sendBuffer,
						   m->sendBuffer_d,
						   m->totToSend*sizeof(*m->sendBuffer),
						   cudaMemcpyDeviceToHost, stream[1]));
#endif
			CHECK_CUDA(cudaEventRecord(event[3], stream[1]));
		}

#ifdef ASYNC_RED
		CHECK_CUDA(cudaStreamSynchronize(stream[0]));
#endif
		t = MPI_Wtime();
		MPI_Allreduce(MPI_IN_PLACE, &sum, 1, REAL_MPI, MPI_SUM, MPI_COMM_WORLD);
		tmpi[0] += MPI_Wtime()-t;

		setarray(r_d[d], m->intColsNum, a*sum, stream[0]);
		END_RANGE;


		START_RANGE("SPMV_LOOP", 1);
		for(int k = 0; k < m->ncsr; k++) {

			int curr = (rank+k) % ntask;

			START_RANGE("SPMV_str0", 2);
			CHECK_CUDA(cudaEventRecord(event[0], stream[0]));
			computeSpmvAcc(/* NO ADD TERM */
				       c, m->nrows[curr], m->rows_d[curr],
                            	       m->roff_d[curr], m->cols_d[curr], m->vals_d[curr],
                            	       r_d[s], r_d[d], m->koff[curr], stream[0]);
			CHECK_CUDA(cudaEventRecord(event[1], stream[0]));
			// dangling node experiment
			/*
    		float dot_res = cugraph::dot( m->intColsNum, d_leaf_vector, r_d[s]);
    		cugraph::axpy(m->intColsNum, dot_res,  bb,  r_d[d]);
   			cugraph::scal(m->intColsNum, static_cast<REAL>(1.0/cugraph::nrm2(m->intColsNum, r_d[d])) , r_d[d]);
			*/
			START_RANGE("MPI+H2D_str1", 3)
			if (k < m->ncsr-1) {

				START_RANGE("SYNC_str1", 4)
				if (k == 0) {
					// wait for the D2H copy of m->sendBuffer initiated before SPMV loop
					if (m->totToSend) {
						CHECK_CUDA(cudaStreamSynchronize(stream[1]));
						CHECK_CUDA(cudaEventElapsedTime(&evt, event[2], event[3]));
						td2h[0] += evt / 1000.0; 
					}
				}
				END_RANGE;
			
	 			START_RANGE("MPI", 5)
				//MPI_Barrier(MPI_COMM_WORLD);
				t = MPI_Wtime();
				exchangeDataSingle(m->sendBuffer + m->sendOffs[k], m->sendNeigh[k], m->sendCnts[k],
						   m->recvBuffer + m->recvOffs[k], m->recvNeigh[k], m->recvCnts[k],
						   reqs+k, MPI_COMM_WORLD);
				tmpi[0] += MPI_Wtime()-t;
				END_RANGE;

				START_RANGE("H2D_str1", 6)
				if (m->recvCnts[k]) {
					CHECK_CUDA(cudaEventRecord(event[2], stream[1]));
					CHECK_CUDA(cudaMemcpyAsync(r_d[s] + m->intColsNum + m->recvOffs[k],
								   m->recvBuffer + m->recvOffs[k],
								   m->recvCnts[k]*sizeof(*r_d[s]),
								   cudaMemcpyHostToDevice, stream[1]));
					CHECK_CUDA(cudaEventRecord(event[3], stream[1]));

					CHECK_CUDA(cudaStreamSynchronize(stream[1]));
					CHECK_CUDA(cudaEventElapsedTime(&evt, event[2], event[3]));
					th2d[0] += evt / 1000.0; 
				}
				END_RANGE;
			}
			END_RANGE;

			CHECK_CUDA(cudaStreamSynchronize(stream[0]));
			CHECK_CUDA(cudaEventElapsedTime(&evt, event[0], event[1]));
			tspmv[0] += evt / 1000.0; 
			END_RANGE;
		}
		END_RANGE;
	}
	//MPI_Pcontrol(0);
	END_RANGE;
	MPI_Barrier(MPI_COMM_WORLD);
	tc = MPI_Wtime()-tc;
	sum = reduce_cuda(r_d[numIter&1], m->intColsNum);
	MPI_Reduce(rank?&sum:MPI_IN_PLACE, &sum, 1, REAL_MPI, MPI_SUM, 0, MPI_COMM_WORLD);

	float loc_nrm_1, glob_nrm1;
	loc_nrm_1= cugraph::nrm1(m->intColsNum,r_d[numIter&1]);
	MPI_Allreduce(&loc_nrm_1, &glob_nrm1, 1, REAL_MPI, MPI_SUM, MPI_COMM_WORLD);

	cugraph::scal(m->intColsNum, (float)1.0/glob_nrm1, r_d[numIter&1]);

#if	MPR_VERBOSE
	{
		char fname[256];
		snprintf(fname, 256, "myresult_%d.txt", rank);
		REAL *r = new REAL[m->intColsNum];
		CHECK_CUDA(cudaMemcpy(r, r_d[numIter&1], m->intColsNum*sizeof(*r), cudaMemcpyDeviceToHost));
		FILE *fp = fopen(fname, "w");
		for(i = 0; i < m->intColsNum; i++)
			fprintf(fp, "%d %E\n",m->firstRow+i, r[i]);
		fclose(fp);
		delete [] r;
	}
#endif

	MPI_Reduce(td2h, td2h+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?td2h:MPI_IN_PLACE, td2h, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(tmpi, tmpi+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?tmpi:MPI_IN_PLACE, tmpi, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(th2d, th2d+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?th2d:MPI_IN_PLACE, th2d, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

	MPI_Reduce(tspmv, tspmv+1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(rank?tspmv:MPI_IN_PLACE, tspmv, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

#if	MPR_VERBOSE
	if (0 == rank) {
		printf("\tgen   time: %.4lf secs\n", tg);
		printf("\tcomp  time: %.4lf secs\n", tc);
		printf("\t\tmin/max  d2h: %.4lf/%.4lf secs\n", td2h[0], td2h[1]);
		printf("\t\tmin/max  mpi: %.4lf/%.4lf secs\n", tmpi[0], tmpi[1]);
		printf("\t\tmin/max  h2d: %.4lf/%.4lf secs\n", th2d[0], th2d[1]);
		printf("\t\tmin/max spmv: %.4lf/%.4lf secs\n", tspmv[0], tspmv[1]);
		printf("PageRank sum: %E\n", sum);
	}
#endif

	cancelReqs(reqs, m->recvNum);

	CHECK_CUDA(cudaStreamDestroy(stream[0]));
	CHECK_CUDA(cudaStreamDestroy(stream[1]));
	CHECK_CUDA(cudaEventDestroy(event[0]));
	CHECK_CUDA(cudaEventDestroy(event[1]));

	if (r_h) free(r_h);
	if (reqs) free(reqs);

	cudaMemcpy(pr,   r_d[numIter&1],   sizeof(float) * m->intColsNum, cudaMemcpyDeviceToDevice);

	if (r_d[0]) CHECK_CUDA(cudaFree(r_d[0]));
	if (r_d[1]) CHECK_CUDA(cudaFree(r_d[1]));

	cudaCheckError();
	return;
}

// Perform gdf input check
// Make local elist_t point to local GDF data 
// No copy
gdf_error load_gdf_input (const gdf_column *src_indices, 
       					  const gdf_column *dest_indices,
       					  elist_t *el) {

  GDF_REQUIRE( src_indices->size == dest_indices->size, GDF_COLUMN_SIZE_MISMATCH );
  GDF_REQUIRE( src_indices->dtype == dest_indices->dtype, GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( ((src_indices->dtype == GDF_INT32) || (src_indices->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
  GDF_REQUIRE( src_indices->size > 0, GDF_DATASET_EMPTY ); 

  el->u = (LOCINT*)src_indices->data; 
  el->v = (LOCINT*)dest_indices->data; 
  el->ned = src_indices->size; 
  return GDF_SUCCESS;
}

// Setup local gdf output
// column gdf_v_idx contains global vertex IDs
// column gdf_pr contains corresponding pr
// No copy
gdf_error fill_gdf_output (spmat_t *m, 
						   REAL *pr,
       					   gdf_column *gdf_v_idx, 
       					   gdf_column *gdf_pr) {

if (gdf_v_idx->dtype == GDF_INT64)
 cugraph::sequence<int64_t>(m->intColsNum,(int64_t*)gdf_v_idx->data,(int64_t)m->firstRow);
else
 cugraph::sequence<int>(m->intColsNum,(int*)gdf_v_idx->data,(int)m->firstRow);
    int	rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

std::cout<< rank<<" "<<m->intColsNum<<std::endl;
  CHECK_CUDA(cudaMemcpy(gdf_pr->data, pr, m->intColsNum*sizeof(float), cudaMemcpyDeviceToDevice));

  return GDF_SUCCESS;
}

// coo to csr
// spmat_t is a custom structure for distributed csr matriices in PRBench
void gdf_multi_coo2csr_t(size_t N, const gdf_column *src_indices, const gdf_column *dest_indices, spmat_t *m) {
	elist_t * el = (elist_t *)Malloc(sizeof(*el));
	load_gdf_input(dest_indices, src_indices, el);
	coo2csr(N, m, el);
	if (el) free(el); //just free the structure
}



//Build a CSR matrix and solve Pagerank
gdf_error gdf_multi_pagerank_impl (const size_t global_v, const gdf_column *src_indices, const gdf_column *dest_indices, 
	                         gdf_column *v_idx, gdf_column *pagerank, const float damping_factor, const int max_iter) {
	GDF_REQUIRE( ((v_idx->dtype == GDF_INT32) || (v_idx->dtype == GDF_INT64)), GDF_UNSUPPORTED_DTYPE );
	GDF_REQUIRE((pagerank->dtype == GDF_FLOAT32), GDF_UNSUPPORTED_DTYPE );

    int	rank, ntask;
	rhsv_t	rval = {RHS_CONSTANT, REAL(1.0)/REAL(global_v), NULL};
	REAL a = (REALV(1.0)-REAL(damping_factor))/((REAL)global_v);
	
	//setup 
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ntask);
	MPI_Barrier(MPI_COMM_WORLD);
	init_cuda();
	spmat_t *m = createSpmat(ntask);
    REAL* pr = nullptr;

    //coo2csr transposed
	gdf_multi_coo2csr_t(global_v, src_indices, dest_indices, m);
	cudaCheckError();
	//allocate local result
	cudaMalloc(&pr,m->intColsNum*sizeof(float));
	//solve
	pagerank_solver(max_iter, damping_factor, a, rval, m, pr);

	//store the local result in gdf_columns
	fill_gdf_output(m, pr, v_idx, pagerank);

	//cleanup
	if (rval.str) free(rval.str);
	cudaFree(pr);
	destroySpmat(m);
	cleanup_cuda();

	return GDF_SUCCESS;
}