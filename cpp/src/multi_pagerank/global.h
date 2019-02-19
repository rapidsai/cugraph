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
#ifndef __PRBENCH_H

#define MIN(x,y) (((x)<(y))?(x):(y))
#define MAX(x,y) (((x)>(y))?(x):(y))

#define LOCINT_SIZE (4)

#if LOCINT_SIZE == 8
#define LOCINT		int64_t
#define LOCINT2		longlong2
#define MAKE_LINT2(x,y)	make_longlong2((x),(y))
#define PRILOC		PRId64
#define LOCINT_MPI	MPI_LONG_LONG
#define LOCINT_MAX	LLONG_MAX
#define LOCINT_MIN	LLONG_MIN
#define ABS(i)		llabs(i)
#define CPUCTZ(x)	(__builtin_ctzll(x))
#define CPUCLZ(x)	(__builtin_clzll(x))
#else
#define LOCINT		int
#define LOCINT2		int2
#define MAKE_LINT2(x,y)	make_int2((x),(y))
#define PRILOC		PRId32
#define LOCINT_MPI	MPI_INT
#define LOCINT_MAX	INT_MAX
#define LOCINT_MIN	INT_MIN
#define ABS(i)		abs(i)
#define CPUCTZ(x)	(__builtin_ctz(x))
#define CPUCLZ(x)	(__builtin_clz(x))
#endif

#define REAL_SIZE (4)

#if REAL_SIZE == 8
#define REAL		double
#define REAL_MPI	MPI_DOUBLE
#define	REAL_MAX	DBL_MAX
#define REALV(a)	(a)
#define REAL_SPEC	"%lf"
#else
#define REAL		float
#define REAL_MPI	MPI_FLOAT
#define	REAL_MAX	FLT_MAX
#define REALV(a)	(a##f)
#define REAL_SPEC	"%f"
#endif

#define TAG(t)  (100*ntask+(t))

#define USE_MAPPED_SENDBUF

#define USE_DEV_IOCONV_READ
#define USE_DEV_IOCONV_WRITE

//#define USE_NVTX

#ifdef __cplusplus
#define LINKAGE "C"
#else
#define LINKAGE
#endif

#define CHECK_CUDA(call) {                                                   \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define CHECK_ERROR(errorMessage) {                                          \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
    }

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors4[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors4 = sizeof(colors4)/sizeof(colors4[0]);

#define START_RANGE(name,cid) { \
        int color_id = cid; \
        color_id = color_id%num_colors4;\
        nvtxEventAttributes_t eventAttrib = {0}; \
        eventAttrib.version = NVTX_VERSION; \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
        eventAttrib.colorType = NVTX_COLOR_ARGB; \
        eventAttrib.color = colors4[color_id]; \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name; \
        nvtxRangePushEx(&eventAttrib); \
}
#define END_RANGE { \
        nvtxRangePop(); \
}
#else
#define START_RANGE(name,cid)
#define END_RANGE
#endif

typedef struct {
	LOCINT	firstRow, lastRow;
	int64_t	intColsNum, extColsNum, totToSend;

	int	*sendNeigh, sendNum;
	int	*recvNeigh, recvNum;
	int64_t	*sendCnts, *recvCnts;
	int64_t	*sendOffs, *recvOffs;

	LOCINT	*rowsToSend_d;
#ifdef USE_MAPPED_SENDBUF
	REAL	*sendBuffer_m;
#else
	REAL	*sendBuffer_d;
#endif
	REAL	*sendBuffer, *recvBuffer;

	// CSRs
	int	ncsr;
	LOCINT	*nnz;
	LOCINT	*nrows;
	LOCINT	**roff_d;
	LOCINT	**rows_d;
	LOCINT	**cols_d;
	REAL	**vals_d;
	LOCINT	(*kthr)[2];
	LOCINT	(*koff)[2];
} spmat_t;

#endif
