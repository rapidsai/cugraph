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
#include <mpi.h>
#ifdef __cplusplus
#ifdef __STDC_LIMIT_MACROS
	#undef __STDC_LIMIT_MACROS
#endif
#define __STDC_LIMIT_MACROS 1
#define __STDC_FORMAT_MACROS 1
#endif
#include <inttypes.h>
#include <cuda.h>
#include <curand_kernel.h>

#ifdef NDEBUG
	#undef NDEBUG
#endif
#define NDEBUG
#include <assert.h>

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "global.h"
#include "phsort.cuh"
#include "utils.h"
#include "cuda_kernels.h"

#include "cub/cub.cuh"
using namespace cub;

#include "tmp_pool.h"

#define THREADS	(128)

#define	NTH_SPMV_BL	(512)
#define	NTH_SPMV_WP	(128)
#define	NTH_SPMV_TH	(128)

#define RXCTA_BL	(1)
#define RXCTA_WP	(1)
#define RXCTA_TH	(1)

#if __CUDA_ARCH__ >= 350
#define LDG(x)		(__ldg(&(x)))
#else
#define LDG(x)		(x)
#endif

#define	DIV_UP(a,b)	(((a)+((b)-1))/(b))

static tmp_pool_t *bufpool=NULL;

extern "C" int assignDeviceToProcess();

// cub utility wrappers ////////////////////////////////////////////////////////
template<typename InputIteratorT,
	 typename OutputIteratorT,
	 typename ReductionOpT,
	 typename T>
static inline void cubReduce(InputIteratorT d_in, OutputIteratorT d_out,
			     int num_items, ReductionOpT reduction_op,
			     T init, cudaStream_t stream=0,
			     bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
					     d_in, d_out, num_items, reduction_op,
					     init, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes,
					     d_in, d_out, num_items, reduction_op,
					     init, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template<typename InputIteratorT , typename OutputIteratorT >
static inline void cubSum(InputIteratorT d_in, OutputIteratorT d_out,
			  int num_items, cudaStream_t stream=0,
			  bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename KeyT>
static inline void cubSortKeys(KeyT *d_keys_in, KeyT *d_keys_out, int num_items,
			       int begin_bit=0, int end_bit=sizeof(KeyT)*8,
			       cudaStream_t stream=0, bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
						  d_keys_in, d_keys_out, num_items,
						  begin_bit, end_bit, stream,
						  debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
						  d_keys_in, d_keys_out, num_items,
						  begin_bit, end_bit, stream,
						  debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template<typename KeyT , typename ValueT >
static inline void cubSortPairs(KeyT *d_keys_in, KeyT *d_keys_out,
				ValueT *d_values_in, ValueT *d_values_out,
				int num_items, int begin_bit=0, int end_bit=sizeof(KeyT)*8,
				cudaStream_t stream=0, bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
						   d_keys_in, d_keys_out, d_values_in,
						   d_values_out, num_items, begin_bit,
						   end_bit, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
						   d_keys_in, d_keys_out, d_values_in,
						   d_values_out, num_items, begin_bit,
						   end_bit, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template<typename KeyT , typename ValueT >
static inline void cubSortPairsDescending(KeyT *d_keys_in, KeyT *d_keys_out,
					  ValueT *d_values_in, ValueT *d_values_out,
					  int num_items, int begin_bit=0, int end_bit=sizeof(KeyT)*8,
					  cudaStream_t stream=0, bool debug_synchronous=false) {
	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
							     d_keys_in, d_keys_out, d_values_in,
							     d_values_out, num_items, begin_bit,
							     end_bit, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes,
							     d_keys_in, d_keys_out, d_values_in,
							     d_values_out, num_items, begin_bit,
							     end_bit, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template<typename InputIteratorT,
	 typename OutputIteratorT ,
	 typename NumSelectedIteratorT>
static inline void cubUnique(InputIteratorT d_in, OutputIteratorT d_out,
			     NumSelectedIteratorT d_num_selected_out, int num_items,
			     cudaStream_t stream=0, bool debug_synchronous=false) {
  
	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
					     d_in, d_out, d_num_selected_out,
					     num_items, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceSelect::Unique(d_temp_storage, temp_storage_bytes,
					     d_in, d_out, d_num_selected_out,
					     num_items, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
	  typename UniqueOutputIteratorT,
          typename LengthsOutputIteratorT,
          typename NumRunsOutputIteratorT>
static inline void cubEncode(InputIteratorT d_in, UniqueOutputIteratorT d_unique_out,
			     LengthsOutputIteratorT d_counts_out, NumRunsOutputIteratorT d_num_runs_out,
			     int num_items, cudaStream_t stream=0, bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes,
						      d_in, d_unique_out, d_counts_out,
						      d_num_runs_out, num_items, stream,
						      debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes,
						      d_in, d_unique_out, d_counts_out,
						      d_num_runs_out, num_items, stream,
						      debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
          typename OutputIteratorT>
static inline void cubMin(InputIteratorT d_in, OutputIteratorT d_out,
			  int num_items, cudaStream_t stream=0,
			  bool debug_synchronous=false) {

	
	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
          typename OutputIteratorT>
static inline void cubMax(InputIteratorT d_in, OutputIteratorT d_out,
			  int num_items, cudaStream_t stream=0,
			  bool debug_synchronous=false) {

	
	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,
					  d_in, d_out, num_items, stream,
					  debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
	  typename OutputIteratorT,
	  typename NumSelectedIteratorT,
	  typename SelectOp>
static inline void cubIf(InputIteratorT d_in, OutputIteratorT d_out,
			 NumSelectedIteratorT d_num_selected_out,
			 int num_items, SelectOp select_op,
			 cudaStream_t stream=0, bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;
	
	CHECK_CUDA(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
					 d_in, d_out, d_num_selected_out,
					 num_items, select_op, stream,
					 debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes,
					 d_in, d_out, d_num_selected_out,
					 num_items, select_op, stream,
					 debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
	  typename FlagIterator,
	  typename OutputIteratorT,
	  typename NumSelectedIteratorT>
static inline void cubFlagged(InputIteratorT d_in, FlagIterator d_flags,
			      OutputIteratorT d_out, NumSelectedIteratorT d_num_selected_out,
			      int num_items, cudaStream_t stream=0,
			      bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
					      d_in, d_flags, d_out, d_num_selected_out,
					      num_items, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes,
					      d_in, d_flags, d_out, d_num_selected_out,
					      num_items, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
	  typename OutputIteratorT>
static inline void cubExclusiveSum(InputIteratorT d_in, OutputIteratorT d_out,
				   int num_items, cudaStream_t stream=0,
				   bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
						 d_in, d_out, num_items, stream,
						 debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
						 d_in, d_out, num_items, stream,
						 debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename InputIteratorT,
	  typename OutputIteratorT>
static inline void cubInclusiveSum(InputIteratorT d_in, OutputIteratorT d_out,
				   int num_items, cudaStream_t stream=0,
				   bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
						 d_in, d_out, num_items, stream,
						 debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
						 d_in, d_out, num_items, stream,
						 debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}

template <typename KeysInputIteratorT,
	  typename UniqueOutputIteratorT,
	  typename ValuesInputIteratorT,
	  typename AggregatesOutputIteratorT,
	  typename NumRunsOutputIteratorT,
	  typename ReductionOpT>
static inline void cubReduceByKey(KeysInputIteratorT d_keys_in, UniqueOutputIteratorT d_unique_out,
				  ValuesInputIteratorT d_values_in, AggregatesOutputIteratorT d_aggregates_out,
				  NumRunsOutputIteratorT d_num_runs_out, ReductionOpT reduction_op,
				  int num_items, cudaStream_t stream=0, bool debug_synchronous=false) {

	void	*d_temp_storage = NULL;
	size_t	temp_storage_bytes = 0;

	CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
						  d_keys_in, d_unique_out,
						  d_values_in, d_aggregates_out,
						  d_num_runs_out, reduction_op,
						  num_items, stream, debug_synchronous));
	d_temp_storage = tmp_get(bufpool, temp_storage_bytes);
	CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp_storage, temp_storage_bytes,
						  d_keys_in, d_unique_out,
						  d_values_in, d_aggregates_out,
						  d_num_runs_out, reduction_op,
						  num_items, stream, debug_synchronous));
	tmp_release(bufpool, d_temp_storage);

	return;
}
 
void init_cuda() {

	int dev = assignDeviceToProcess();
	CHECK_CUDA(cudaSetDevice(dev));
	//CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));

	bufpool = tmp_create();
	return;
}

// Graph ASCII 2 BIN convert kernels ///////////////////////////////////////////

void *CudaMalloc(size_t sz) {

        void *ptr;

        CHECK_CUDA(cudaMalloc(&ptr, sz));
	return ptr;
}

__global__ void	fix_last_nl(char *lastch) {

	lastch[1+threadIdx.x] = (0 == threadIdx.x) ? ((lastch[0] != '\n') ? '\n': -1) : -1;
	return;
}	
/*
template<typename OFFVTYPE>
__global__ void find_nl(const uint4 * __restrict__ v, int64_t n, OFFVTYPE *__restrict__ bl_off) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;

	uint4 data = v[tid];
	OFFVTYPE offs;

	data.x ^= 0x0A0A0A0A;
	data.x = (data.x - 0x01010101) & ~data.x & 0x80808080;
	data.x = __ffs(data.x);

	// assigning -1 is the most portable, correct way to
	// set all bits of offs.x
	offs.x = data.x ? sizeof(data)*tid + ((data.x-1)>>3) : -1;
	data.y ^= 0x0A0A0A0A;
	data.y = (data.y - 0x01010101) & ~data.y & 0x80808080;
	data.y = __ffs(data.y);
	offs.y = data.y ? tid*sizeof(data) + 4 + ((data.y-1)>>3) : -1;
	data.z ^= 0x0A0A0A0A;
	data.z = (data.z - 0x01010101) & ~data.z & 0x80808080;
	data.z = __ffs(data.z);
	offs.z = data.z ? tid*sizeof(data) + 8 + ((data.z-1)>>3) : -1;
	data.w ^= 0x0A0A0A0A;
	data.w = (data.w - 0x01010101) & ~data.w & 0x80808080;
	data.w = __ffs(data.w);
	offs.w = data.w ? tid*sizeof(data) + 12 + ((data.w-1)>>3) : -1;
	bl_off[tid] = offs;
	return;
}
*/

template<typename INT_T>
__device__ const char *str2bin(const char *s, INT_T *v) {

	INT_T pw=1;
	
	*v = 0;
	while(*s >= '0' && *s <= '9') {
		*v += pw*(*s-- - '0');
		pw *= 10;
	}
	return s;
}

template<typename INT_T>
__device__ const char *str2bin(const char *s, INT_T *v, const char *const base) {

	INT_T pw=1;
	
	*v = 0;
	while(s >= base && *s >= '0' && *s <= '9') {
		*v += pw*(*s-- - '0');
		pw *= 10;
	}
	return s;
}

template<typename VALINT_T,
	 typename OFFINT_T,
	 typename LENINT_T>
__global__ void convert_a2b(const char *__restrict__ data,
			    const OFFINT_T *__restrict__ str_off, const LENINT_T n,
			    VALINT_T *__restrict__ u, VALINT_T *__restrict__ v) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;

	VALINT_T _u, _v;
	const char *ptr = str2bin(data + str_off[tid]-1, &_v);
	
	if (tid) str2bin(ptr-1, &_u);
	else	 str2bin(ptr-1, &_u, data); // th. 0 does not have a preceding sep

	u[tid] = _u;
	v[tid] = _v;

	return;
}

template<typename CMPTYPE>
struct NotEqualTo {
	CMPTYPE compare;
	__device__ CUB_RUNTIME_FUNCTION __forceinline__ NotEqualTo(CMPTYPE compare) : compare(compare) {}
	__device__ CUB_RUNTIME_FUNCTION __forceinline__ bool operator()(const CMPTYPE &a) const {
		return (a != compare);
	}
};

// ascii data is always read as uint4 while to store char
// offsets we use either uint or ulonglong based on file size
template<typename OFFSTYPE, typename OFFVTYPE>
size_t ASCIICouple2BinCuda(uint4 *d_data, size_t datalen, LOCINT **d_uout, LOCINT **d_vout, int verbose) {

	cudaEvent_t start, stop;
        float et;

	int nchunks = DIV_UP(datalen+1, sizeof(uint4));

	if (verbose) {
		CHECK_CUDA( cudaEventCreate(&start) );
		CHECK_CUDA( cudaEventCreate(&stop) );
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	// if last char is not a newline, add one
	// after it; all subsequent bytes are set to 0xFF
	// (this is called with a number of threads between 1 and 16)
	fix_last_nl<<<1, sizeof(uint4)*nchunks-datalen>>>(((char *)d_data)+datalen-1); 
	CHECK_ERROR("fix_last_nl");
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stdout, "Kernel fix_last_nl execution time: %E ms\n", et);
	}

	dim3 block(THREADS, 1, 1);
	dim3 grid(DIV_UP(nchunks,THREADS), 1, 1);

	//uint4 *d_str_off = (uint4 *)CudaMalloc(nchunks*sizeof(*d_str_off));
	OFFVTYPE *d_str_off = (OFFVTYPE *)tmp_get(bufpool, nchunks*sizeof(*d_str_off));

	if (verbose) {
		fprintf(stdout, "File size: %zu, nblocks: %d, nthreads: %d\n", datalen, grid.x, block.x);
		fprintf(stdout, "Using %zu bytes for d_data[]\n", nchunks*sizeof(*d_data));
		fprintf(stdout, "Using %zu bytes for d_str_off[]\n", nchunks*sizeof(*d_str_off));
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	find_nl<<<grid, block>>>(d_data, nchunks, d_str_off);
	CHECK_ERROR("find_nl");
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel find_nl execution time: %E ms\n", et);
	}

	NotEqualTo<OFFSTYPE> neq(-1);

	size_t  nstr;
	size_t *d_nstr = (size_t *)tmp_get(bufpool, sizeof(*d_nstr));

	if (verbose) {
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	cubIf((OFFSTYPE *)d_str_off, (OFFSTYPE *)d_str_off, d_nstr, nchunks*4, neq);
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel cub::DeviceSelect::If execution time: %E ms\n", et);
	}

	// nstr contains the number of '\n' found in the file data
	CHECK_CUDA(cudaMemcpy(&nstr, d_nstr, sizeof(nstr), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, d_nstr);

	if (verbose) {
		OFFSTYPE *str_off_h = (OFFSTYPE *)Malloc(nstr*sizeof(*str_off_h));
		CHECK_CUDA( cudaMemcpy(str_off_h, d_str_off, nstr*sizeof(*str_off_h), cudaMemcpyDeviceToHost) );

		int i, j;
		printf("Edge offsets (%zu):", nstr);
		for(i = j = 0; i < nstr; i++) {
			if (str_off_h[i] == (OFFSTYPE)-1) {
				fprintf(stderr, "Error, found invalid offset value after compact!\n");
				exit(EXIT_FAILURE);
			}
			if (!(j % 10)) printf("\n\t");

			printf("%llu ", (unsigned long long)str_off_h[i]);

			j++; 
			if (j >= 1000) {
				printf("\n\t...");
				break;
			}
		}
		printf("\n");
		free(str_off_h);
	}

	LOCINT *d_u, *d_v;
	CHECK_CUDA(cudaMalloc(&d_u, nstr*sizeof(*d_u)));
	CHECK_CUDA(cudaMalloc(&d_v, nstr*sizeof(*d_v)));

	if (verbose) {
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	convert_a2b<<<DIV_UP(nstr,THREADS), THREADS>>>((char *)d_data, (OFFSTYPE *)d_str_off, nstr, d_u, d_v);
	CHECK_ERROR("convert_a2b");
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel convert_a2b execution time: %E ms\n", et);
	}

	CHECK_CUDA(cudaDeviceSynchronize());
	tmp_release(bufpool, d_str_off);

	*d_uout = d_u;
	*d_vout = d_v;

	return nstr;
}

/*
size_t ASCIICouple2BinCuda_entry(uint4 *d_data, size_t datalen, LOCINT **d_uout, LOCINT **d_vout, int verbose) {

	size_t rv;
	if (datalen < 0xFFFFFFFFllu) rv = ASCIICouple2BinCuda<      unsigned int,      uint4>(d_data, datalen, d_uout, d_vout, verbose);
	else			     rv = ASCIICouple2BinCuda<unsigned long long, ulonglong4>(d_data, datalen, d_uout, d_vout, verbose);

	return rv;
}*/

template<typename INT_T>
__device__ INT_T ilog10_ceil(INT_T x) {

	INT_T l=0;
	do {
		x /= 10;
		l++;
	} while(x);
	return l;
}

template<typename VALINT_T,
	 typename LENINT_T,
	 typename OFFINT_T>
__global__ void get_str_len(const VALINT_T *__restrict__ u,
			    const VALINT_T *__restrict__ v,
			    LENINT_T n,
			    OFFINT_T *__restrict__ slenv) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	OFFINT_T len;

	if (tid >= n) return;

	len  = ilog10_ceil(u[tid]);
	len += ilog10_ceil(v[tid]);

	slenv[tid] = len+2; // add '\t' and '\n'
	return;
}

template<typename INT_T>
__device__  char *bin2str(const INT_T v, char *s) {

	INT_T pw=1;
	do {
		*s-- = '0' + ((v/pw)%10);
		pw *= 10;
	} while (v/pw);
	return s;
}

template<char SEP,
	 typename VALINT_T,
	 typename OFFINT_T,
	 typename LENINT_T>
__global__ void convert_b2a(const VALINT_T *__restrict__ const u,
			    const VALINT_T *__restrict__ const v,
			    const OFFINT_T *__restrict__ const str_off,
			    const LENINT_T n,
			    char *__restrict__ data) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid >= n) return;
	
	VALINT_T valu = u[tid];
	VALINT_T valv = v[tid];

	data += str_off[tid]-1;

	*data-- = '\n';
	data = bin2str(valv, data);

	*data-- = SEP;
	bin2str(valu, data);
	
	return;
}

size_t BinCouple2ASCIICuda(LOCINT *d_u, LOCINT *d_v, int64_t n, char **d_data, int verbose) {

	cudaEvent_t start, stop;
        float et;
	size_t dlen;

	if (!n) return 0;
	size_t *d_str_off = (size_t *)tmp_get(bufpool, n*sizeof(*d_str_off));

	if (verbose) {
		CHECK_CUDA( cudaEventCreate(&start) );
		CHECK_CUDA( cudaEventCreate(&stop) );
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	get_str_len<<<DIV_UP(n,THREADS), THREADS>>>(d_u, d_v, n, d_str_off);
	CHECK_ERROR("get_str_len");
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel get_str_len time: %E ms\n", et);
	}

	if (verbose) {
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	cubInclusiveSum(d_str_off, d_str_off, n);
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel cub::DeviceScan::ExclusiveSum execution time: %E ms\n", et);
	}

	CHECK_CUDA(cudaMemcpy(&dlen, d_str_off+n-1, sizeof(dlen), cudaMemcpyDeviceToHost));
	//printf("Required %zu bytes\n", dlen);

	d_data[0] = (char *)CudaMalloc(dlen);
	
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(start, 0) );
	}
	convert_b2a<'\t'><<<DIV_UP(n,THREADS), THREADS>>>(d_u, d_v, d_str_off, n, d_data[0]);
	CHECK_ERROR("convert_b2a");
	if (verbose) {
		CHECK_CUDA( cudaEventRecord(stop, 0) );
		CHECK_CUDA( cudaEventSynchronize(stop) );
		CHECK_CUDA( cudaEventElapsedTime(&et, start, stop) );
		fprintf(stderr, "Kernel convert_b2a execution time: %E ms\n", et);
	}
	CHECK_CUDA(cudaDeviceSynchronize());
	tmp_release(bufpool, d_str_off);

	return dlen;
}

// Parallel Sort ///////////////////////////////////////////////////////////////

static int64_t	nmax;
static int	currSamp, nextSamp;
static LOCINT	*d_usamp[2]={NULL,NULL};
static LOCINT	*d_vsamp[2]={NULL,NULL};

void copy_sort_samples(LOCINT *h_u, LOCINT *h_v, LOCINT nsample, int64_t nsamplemax) {

	if (nsample > nsamplemax) {
		fprintf(stderr, "%s:%d: error nsample=%" PRILOC " > nsamplemax=%" PRId64 "!\n", __FILE__, __LINE__, nsample, nsamplemax);
		exit(EXIT_FAILURE);
	}

	d_usamp[0] = (LOCINT *)tmp_get(bufpool, nsamplemax*sizeof(LOCINT));
	d_vsamp[0] = (LOCINT *)tmp_get(bufpool, nsamplemax*sizeof(LOCINT));
	d_usamp[1] = (LOCINT *)tmp_get(bufpool, nsamplemax*sizeof(LOCINT));
	d_vsamp[1] = (LOCINT *)tmp_get(bufpool, nsamplemax*sizeof(LOCINT));

	currSamp = 0;
	nextSamp = 1; 

	CHECK_CUDA(cudaMemcpy(d_usamp[currSamp], h_u, nsample*sizeof(LOCINT), cudaMemcpyDeviceToDevice));
	CHECK_CUDA(cudaMemcpy(d_vsamp[currSamp], h_v, nsample*sizeof(LOCINT), cudaMemcpyDeviceToDevice));

	nmax = nsamplemax;
	return;
}

struct OrOper {
	template <typename T>
	__device__ CUB_RUNTIME_FUNCTION __forceinline__
	T operator()(const T &a, const T &b) const {
		return a|b;
	}
};

void limits_cuda(int n, LOCINT *smin, LOCINT *smax, int *bbit, int *ebit) {

	LOCINT	h_limits[3];
	LOCINT	*d_limits=NULL;

	OrOper	orOp;

	if (!n) return;
	if (n > nmax) {
		fprintf(stderr, "%s:%s:%d: n(=%d) > nmax(=%" PRId64 ")\n", __FILE__, __func__, __LINE__, n, nmax);
		exit(EXIT_FAILURE);
	}

	d_limits = (LOCINT *)tmp_get(bufpool, 3*sizeof(*d_limits));

	cubMin(d_usamp[currSamp], d_limits, n);
	cubMax(d_usamp[currSamp], d_limits+1, n);
	cubReduce(d_usamp[currSamp], d_limits+2, n, orOp, 0);

	CHECK_CUDA(cudaMemcpy(h_limits, d_limits, 3*sizeof(*h_limits), cudaMemcpyDeviceToHost));

	*smin = h_limits[0];
	*smax = h_limits[1];

	*bbit = CPUCTZ(h_limits[2]);
	*ebit = 8*sizeof(LOCINT) - CPUCLZ(h_limits[2]);

	//printf("smin=%lld, smax=%lld, bbit=%d, ebit=%d\n", *smin, *smax, *bbit, *ebit);
	tmp_release(bufpool, d_limits);
	return;	
}

__device__ int bsectl(const LOCINT *v, const int num, const LOCINT val) {

	if (0 == num) return -1;
#if 1
	int  min = 0;
	int  max = num-1;
	int  mid = max >> 1;

	while(min <= max) {

		if (v[mid] == val)      break;
		if (v[mid]  < val)      min = mid+1;
		else			max = mid-1;
		mid = (max>>1)+(min>>1)+((min&max)&1);
	}
	if (mid >= 0 && v[mid] == val) {
		while(mid) {
			if (v[mid-1] == val) mid--;
			else		     break;
		}
	} else mid++;

	return mid;
#else
	int i;
	for(i = 0; i < num; i++)
		if (v[i] >= val) break;

	return i;
#endif
}

__global__ void hist(LOCINT *vals, int nv, LOCINT *prb, int np, int64_t *hist) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < np) hist[tid] = bsectl(vals, nv, prb[tid]);
	return;
}

void get_hist_cuda(int64_t nsample, int64_t *h_hist, LOCINT *h_probes, int np) {

	LOCINT	*d_probes=NULL;
	int64_t	*d_hist=NULL;

	d_probes = (LOCINT *)tmp_get(bufpool, np*sizeof(*d_probes));
	d_hist = (int64_t *)tmp_get(bufpool, np*sizeof(*d_hist));

	CHECK_CUDA(cudaMemcpy(d_probes, h_probes, np*sizeof(*d_probes), cudaMemcpyHostToDevice));
	hist<<<DIV_UP(np,THREADS), THREADS>>>(d_usamp[currSamp], nsample, d_probes, np, d_hist);
	CHECK_ERROR("hist");

	CHECK_CUDA(cudaMemcpy(h_hist, d_hist, np*sizeof(*h_hist), cudaMemcpyDeviceToHost));

	tmp_release(bufpool, d_probes);
	tmp_release(bufpool, d_hist);

	return;
}

void sort_cuda(LOCINT *h_u, LOCINT *h_v, int n, int bbit, int ebit) {

	if (!n) return;
	if (n > nmax) {
		fprintf(stderr, "%s:%s:%d: n(=%d) > nmax(=%" PRId64 ")\n", __FILE__, __func__, __LINE__, n, nmax);
		exit(EXIT_FAILURE);
	}
	//printf("bbit=%d, ebit=%d\n", bbit, ebit);

	if (h_u) {
		CHECK_CUDA(cudaMemcpy(d_usamp[currSamp], h_u, n*sizeof(LOCINT), cudaMemcpyHostToDevice));
		CHECK_CUDA(cudaMemcpy(d_vsamp[currSamp], h_v, n*sizeof(LOCINT), cudaMemcpyHostToDevice));
	}

	// does not work in-place with int64_t (returns a sorted array with duplicates not in the original array)
	cubSortPairs(d_usamp[currSamp], d_usamp[nextSamp], d_vsamp[currSamp], d_vsamp[nextSamp], n, bbit, ebit);

	currSamp ^= 1;
	nextSamp ^= 1;

	if (h_u) {
		CHECK_CUDA(cudaMemcpy(h_u, d_usamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(h_v, d_vsamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToHost));
	}

	return;
}

void final_sort_cuda(LOCINT *h_u, LOCINT *h_v, int n) {

	if (!n) return;
	if (n > nmax) {
		fprintf(stderr, "%s:%s:%d: n(=%d) > nmax(=%" PRId64 ")\n", __FILE__, __func__, __LINE__, n, nmax);
		exit(EXIT_FAILURE);
	}
	//printf("bbit=%d, ebit=%d\n", bbit, ebit);

	CHECK_CUDA(cudaMemcpy(d_usamp[currSamp], h_u, n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));
	CHECK_CUDA(cudaMemcpy(d_vsamp[currSamp], h_v, n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));

	cubSortPairs(d_vsamp[currSamp], d_vsamp[nextSamp], d_usamp[currSamp], d_usamp[nextSamp], n);
	cubSortPairs(d_usamp[nextSamp], d_usamp[currSamp], d_vsamp[nextSamp], d_vsamp[currSamp], n);

	CHECK_CUDA(cudaMemcpy(h_u, d_usamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));
	CHECK_CUDA(cudaMemcpy(h_v, d_vsamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));

	return;
}

void getvals_cuda(LOCINT *h_u, LOCINT *h_v, int64_t n) {
	
	if (!n) return;
	if (n > nmax) {
		fprintf(stderr, "%s:%s:%d: n(=%" PRId64 ") > nmax(=%" PRId64 ")\n", __FILE__, __func__, __LINE__, n, nmax);
		exit(EXIT_FAILURE);
	}
	CHECK_CUDA(cudaMemcpy(h_u, d_usamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));
	CHECK_CUDA(cudaMemcpy(h_v, d_vsamp[currSamp], n*sizeof(LOCINT), cudaMemcpyDeviceToDevice));
	return;
}

void setvals_cuda(LOCINT *h_u, LOCINT *h_v, int n) {
	
	if (!n) return;
	if (n > nmax) {
		fprintf(stderr, "%s:%s:%d: n(=%d) > nmax(=%" PRId64 ")\n", __FILE__, __func__, __LINE__, n, nmax);
		exit(EXIT_FAILURE);
	}
	CHECK_CUDA(cudaMemcpy(d_usamp[currSamp], h_u, n*sizeof(LOCINT), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(d_vsamp[currSamp], h_v, n*sizeof(LOCINT), cudaMemcpyHostToDevice));
	return;
}

void finalize_sort_cuda() {

	tmp_release(bufpool, d_usamp[0]);
	tmp_release(bufpool, d_usamp[1]);
	tmp_release(bufpool, d_vsamp[0]);
	tmp_release(bufpool, d_vsamp[1]);

	d_usamp[0] = NULL;
	d_usamp[1] = NULL;
	d_vsamp[0] = NULL;
	d_vsamp[1] = NULL;

	return;
}

// Kronecker Graph Generator ///////////////////////////////////////////////////

template<typename LenghtTypeT,
	 typename ElemIteratorT,
	 typename CoupIteratorT>
__global__ void zip(LenghtTypeT n, ElemIteratorT *v1, ElemIteratorT *v2, CoupIteratorT *v12) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) {
		CoupIteratorT __tmp;
		__tmp.x = v1[tid];
		__tmp.y = v2[tid];
		v12[tid] = __tmp;
	}
	return;
}

template<typename LenghtTypeT,
	 typename ElemIteratorT,
	 typename CoupIteratorT>
__global__ void unzip(LenghtTypeT n, CoupIteratorT *v12, ElemIteratorT *v1, ElemIteratorT *v2) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) {
		CoupIteratorT __tmp = v12[tid];
		if (v1) v1[tid] = __tmp.x;
		if (v2) v2[tid] = __tmp.y;
	}
	return;
}

template<typename T>
__global__ void setval(T *arr, int64_t n, T val) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) arr[tid] = val;
	return;
}

template<typename T>
__device__ __host__ int bsearch(const T *v, const int num, const T val) {

	if (0 == num) return -1;

	int  min = 0;
	int  max = num-1;
	int  mid = max >> 1;

	while(min <= max) {

		if (v[mid] == val)	return mid;
		if (v[mid]  < val)	min = mid+1;
		else			max = mid-1;

		mid = (max>>1)+(min>>1)+((min&max)&1);
	}
	return -1;
}

__device__ __forceinline__ uint32_t bitreverse(uint32_t w) {

	w = ((w >>  1) & 0x55555555) | ((w & 0x55555555) <<  1);
	w = ((w >>  2) & 0x33333333) | ((w & 0x33333333) <<  2);
	w = ((w >>  4) & 0x0F0F0F0F) | ((w & 0x0F0F0F0F) <<  4);
	w = ((w >>  8) & 0x00FF00FF) | ((w & 0x00FF00FF) <<  8);
	w = ((w >> 16)) | (w << 16);

	return w;
}

__device__ __forceinline__ uint64_t bitreverse(uint64_t w) {

	w = ((w >>  1) & 0x5555555555555555) | ((w & 0x5555555555555555) <<  1);
	w = ((w >>  2) & 0x3333333333333333) | ((w & 0x3333333333333333) <<  2);
	w = ((w >>  4) & 0x0F0F0F0F0F0F0F0F) | ((w & 0x0F0F0F0F0F0F0F0F) <<  4);
	w = ((w >>  8) & 0x00FF00FF00FF00FF) | ((w & 0x00FF00FF00FF00FF) <<  8);
	w = ((w >> 16) & 0x0000FFFF0000FFFF) | ((w & 0x0000FFFF0000FFFF) << 16);
	w = ((w >> 32)) | (w << 32);

	return w;
}

/* Taken from graph_generator.c in Graph500 reference code */
template<typename T>
__device__ __forceinline__ T scramble(T v0, int lgN, uint64_t val0, uint64_t val1) {

  uint64_t v = (uint64_t)v0;

  v += val0 + val1;
  v *= (val0 | 0x4519840211493211ull);
  v = (bitreverse(v) >> (64 - lgN));
  //assert ((v >> lgN) == 0);
  v *= (val1 | 0x3050852102C843A5ull);
  v = (bitreverse(v) >> (64 - lgN));
  //assert ((v >> lgN) == 0);

  return (T)v;
}

template<typename T, int CLIPNFLIP>
__global__
__launch_bounds__(THREADS, 16)
void rmatgen(T *__restrict vi, T *__restrict vj, int ned, int scale,
	     REAL a, REAL ab, REAL abc, uint64_t seed, int64_t permV) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	curandStatePhilox4_32_10_t st;

	if (tid < ned) {
		// this may be moved in a separate kernel at the cost
		// of the memory required to store the state for each thread
		curand_init(seed, tid, 0, &st);

		T x=0, y=0;

		for(T s = (1<<(scale-1)); s > 0; s >>= 1) {
#if REAL_SIZE == 8
			REAL r = curand_uniform_double(&st);
#else
			REAL r = curand_uniform(&st);
#endif
			int ib = r > ab;
			int jb = (r > a  && r < ab ) || (r > abc);
			if (CLIPNFLIP) {
				if (x == y) {
					if (ib > jb) {
						int tmp = ib;
						ib = jb;
						jb = tmp;
					}
				}
			}
			x |= ib*s;
			y |= jb*s;
		}
		// Simple modulo-based permutation based on the fact that
		// if MCD(N,P)=1 => [((P*i) mod N) for i in [1,N]] is a
		// permutation of [0,N-1].
		// Since we want to permute numbers between 0 and 2^k-1 (N=2^k)
		// any odd (big) number will do as P (920419813).
		if (permV) {
#if 0
			x = (permV*(x+1))&((1<<scale)-1);
			y = (permV*(y+1))&((1<<scale)-1);
#else
			x = scramble(x, scale, 34247921037847, 7386743432);
			y = scramble(y, scale, 34247921037847, 7386743432);
#endif
		}
		vi[tid] = x;
		vj[tid] = y;
	}
	return;
}

void generate_kron(int scale, int64_t ned,
		   REAL a, REAL b, REAL c,
		   LOCINT *d_i, LOCINT *d_j,
		   int64_t seed, int cnf, int perm) {

	if (scale > 31 && sizeof(LOCINT) == 4) {
		fprintf(stderr,
			"%s:%s:%d: cannot generate edges for a graph with 2^%d vertices using 4 bytes type!",
			__FILE__, __func__, __LINE__, scale);
		exit(EXIT_FAILURE);
	}

	int	nblocks = DIV_UP(ned,THREADS);
	int64_t	permv =  perm ? ((9204198134365434+seed)|1) : 0;

	if (cnf) rmatgen<LOCINT, 1><<<nblocks, THREADS>>>(d_i, d_j, ned, scale, a, a+b, a+b+c, seed, permv);
	else	 rmatgen<LOCINT, 0><<<nblocks, THREADS>>>(d_i, d_j, ned, scale, a, a+b, a+b+c, seed, permv);

	CHECK_ERROR("rmatgen");
	CHECK_CUDA(cudaDeviceSynchronize());

	return;
}

// Kernel2 kenrels (CSR gen and opt) ///////////////////////////////////////////

template<typename T>
__global__ void mark_outside(T *v, int64_t n, T min, T max) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid < n) {
		T val = v[tid];
		v[tid] = (val <= min) || (val >= max);
	}
	return;
}

template<typename TIN, typename TMARK>
__global__ void mark_notinset(TIN *vin, TMARK *vout, const int64_t n, const TIN *__restrict toMark, const int nmark) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid < n) {
		const TIN val = vin[tid];
		vout[tid] = (bsearch<TIN>(toMark, nmark, val) == -1) ? (TMARK)1 : (TMARK)0;
	}
	return;
}

__global__ void mark_subm(LOCINT *v, int64_t n, LOCINT *sep, int nsep, int *subm) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n)
		subm[tid] = bsectl(sep, nsep, v[tid]);
	return;
} 

int64_t keep_all_rows_cuda(LOCINT *u_h, LOCINT *v_h, int64_t ned, LOCINT **uout_d, LOCINT **vout_d) {
	uout_d[0] = (LOCINT *)tmp_get(bufpool, ned*sizeof(*uout_d[0]));
	vout_d[0] = (LOCINT *)tmp_get(bufpool, ned*sizeof(*vout_d[0]));
	CHECK_CUDA(cudaMemcpy(uout_d[0], u_h, ned*sizeof(*uout_d[0]), cudaMemcpyDeviceToDevice));
	CHECK_CUDA(cudaMemcpy(vout_d[0], v_h, ned*sizeof(*vout_d[0]), cudaMemcpyDeviceToDevice));
	return ned;
}

int64_t remove_rows_cuda(LOCINT *u_h, LOCINT *v_h, int64_t ned, LOCINT **uout_d, LOCINT **vout_d) {

	// 1. REMOVE ROWS WITH LENGTH EQUAL TO MAXIMUM OR 1 

	LOCINT *u_d = (LOCINT *)tmp_get(bufpool, ned*sizeof(*u_d));
	CHECK_CUDA(cudaMemcpy(u_d, u_h, ned*sizeof(*u_d), cudaMemcpyDeviceToDevice));

	// compute length of rows
	LOCINT *uu_d = (LOCINT *)tmp_get(bufpool, ned*sizeof(*uu_d));
	LOCINT *uc_d = (LOCINT *)tmp_get(bufpool, ned*sizeof(*uc_d));
	int64_t *n_d = (int64_t *)tmp_get(bufpool, sizeof(*n_d));
	cubEncode(u_d, uu_d, uc_d, n_d, ned);

	// d2h non-zero rows
	int64_t nuniq;
	CHECK_CUDA(cudaMemcpy(&nuniq, n_d, sizeof(nuniq), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, n_d);

	// compute local max length
	LOCINT maxc;
	LOCINT *maxc_d = (LOCINT *)tmp_get(bufpool, sizeof(*maxc_d));
	cubMax(uc_d, maxc_d, nuniq);
	CHECK_CUDA(cudaMemcpy(&maxc, maxc_d, sizeof(maxc), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, maxc_d);

	// find global max length
	MPI_Allreduce(MPI_IN_PLACE, &maxc, 1, LOCINT_MPI, MPI_MAX, MPI_COMM_WORLD);

	// compute rows-to-del array
	mark_outside<LOCINT><<<DIV_UP(nuniq,THREADS), THREADS>>>(uc_d, nuniq, (LOCINT)1, maxc);
	CHECK_ERROR("mark_outside");

	LOCINT *ubad_d = (LOCINT *)tmp_get(bufpool, nuniq*sizeof(*ubad_d));
	LOCINT *nbad_d = (LOCINT *)tmp_get(bufpool,       sizeof(*nbad_d));
	cubFlagged(uu_d, uc_d, ubad_d, nbad_d, nuniq);
	LOCINT nbad;
	CHECK_CUDA(cudaMemcpy(&nbad, nbad_d, sizeof(nbad), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, nbad_d);

	// release buffers no longer needed
	tmp_release(bufpool, uu_d);
	tmp_release(bufpool, uc_d);

	//remove edges leaving a bad row
	unsigned char *eflag = (unsigned char *)tmp_get(bufpool, ned*sizeof(*eflag));
	mark_notinset<LOCINT, unsigned char><<<DIV_UP(ned,THREADS), THREADS>>>(u_d, eflag, ned, ubad_d, nbad);
	CHECK_ERROR("mark_notinset");

	int64_t nrem;
	int64_t *nrem_d = (int64_t *)tmp_get(bufpool, sizeof(*nrem_d));

	LOCINT *tmp_d = (LOCINT *)tmp_get(bufpool, ned*sizeof(*tmp_d));
	LOCINT *v_d = (LOCINT *)tmp_get(bufpool, ned*sizeof(*v_d));
	CHECK_CUDA(cudaMemcpy(v_d, v_h, ned*sizeof(*v_d), cudaMemcpyDeviceToDevice));

	cubFlagged(u_d, eflag, tmp_d, nrem_d, ned);
	CHECK_CUDA(cudaMemcpy(&nrem, nrem_d, sizeof(nrem), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(u_d, tmp_d, nrem*sizeof(*u_d), cudaMemcpyDeviceToDevice));

	cubFlagged(v_d, eflag, tmp_d, nrem_d, ned);
	// nrem doesn't change
	CHECK_CUDA(cudaMemcpy(v_d, tmp_d, nrem*sizeof(*v_d), cudaMemcpyDeviceToDevice));

	// let's record new number of edges now...
	ned = nrem; 

	uout_d[0] = u_d;
	vout_d[0] = v_d;
	
	tmp_release(bufpool, ubad_d);
	tmp_release(bufpool, nrem_d);
	tmp_release(bufpool, eflag);
	tmp_release(bufpool, tmp_d);

	return ned;
}

void get_csr_multi_cuda(LOCINT *u_d, LOCINT *v_d, int64_t ned,
			LOCINT *sep, int ntask,
			LOCINT *nnz, LOCINT *nrows, LOCINT **roff_d,
			LOCINT **rows_d, LOCINT **cols_d, REAL **vals_d) {

	LOCINT *sep_d = (LOCINT *)tmp_get(bufpool, ntask*sizeof(*sep_d));
	CHECK_CUDA(cudaMemcpy(sep_d, sep, ntask*sizeof(*sep_d), cudaMemcpyHostToDevice));
	
	int *subm_d = (int *)tmp_get(bufpool, ned*sizeof(*subm_d));
	mark_subm<<<DIV_UP(ned,THREADS), THREADS>>>(v_d, ned, sep_d, ntask, subm_d);
	CHECK_ERROR("mark_subm");

	LOCINT2 *uv_d = (LOCINT2 *)tmp_get(bufpool, ned*sizeof(*uv_d));
	zip<<<DIV_UP(ned,THREADS), THREADS>>>(ned, u_d, v_d, uv_d);
	CHECK_ERROR("zip");

	tmp_release(bufpool, u_d);
	tmp_release(bufpool, v_d);
	
	int *subm_sort_d = (int *)tmp_get(bufpool, ned*sizeof(*subm_sort_d));
	LOCINT2 *uv_sort_d = (LOCINT2 *)tmp_get(bufpool, ned*sizeof(*uv_sort_d));
	cubSortPairs(subm_d, subm_sort_d, uv_d, uv_sort_d, ned);

	tmp_release(bufpool, subm_d);
	tmp_release(bufpool, uv_d);
	tmp_release(bufpool, sep_d);
	
	int *subm_id_d = (int *)tmp_get(bufpool, ned*sizeof(*subm_id_d));
	int *subm_ned_d = (int *)tmp_get(bufpool, ned*sizeof(*subm_ned_d));
	int *necsr_d = (int *)tmp_get(bufpool, sizeof(*necsr_d));

	int necsr;
	int *subm_id = (int *)Malloc(ntask*sizeof(*subm_id));
	
	cubEncode(subm_sort_d, subm_id_d, subm_ned_d, necsr_d, ned);
	CHECK_CUDA(cudaMemcpy(&necsr, necsr_d, sizeof(necsr), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(subm_id, subm_id_d, necsr*sizeof(*subm_id), cudaMemcpyDeviceToHost));

	tmp_release(bufpool, subm_id_d);
	tmp_release(bufpool, subm_sort_d);
	tmp_release(bufpool, necsr_d);
	
	assert(necsr <= ntask);
	//fprintf(stderr, "ncsr=%d\n", ncsr[0]);

	int *subm_ned = (int *)Malloc(ntask*sizeof(*subm_ned));
	memset(subm_ned, 0, ntask*sizeof(*subm_ned));
	CHECK_CUDA(cudaMemcpy(subm_ned, subm_ned_d, necsr*sizeof(*subm_ned), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, subm_ned_d);

	// from subm_id[i], subm_ned[i] -> i, subm_ned[i]
	for(int i = necsr-1; i >= 0; i--) {
		if (subm_id[i] != i) {
			assert(subm_id[i] > i);
			subm_ned[subm_id[i]] = subm_ned[i];
			subm_ned[i] = 0;
		}
	}
	free(subm_id);
			
	int offs = 0;
	for(int i = 0; i < ntask; i++) {

		if (subm_ned[i] == 0) {
			nnz[i] = 0;
			nrows[i] = 0;
			roff_d[i] = NULL;
			rows_d[i] = NULL;
			cols_d[i] = NULL; 
			vals_d[i] = NULL;
			continue;
		}

		LOCINT2 *uvu_d = (LOCINT2 *)tmp_get(bufpool, subm_ned[i]*sizeof(*uvu_d));
		REAL *values_d = (REAL *)tmp_get(bufpool, subm_ned[i]*sizeof(*values_d));
		int64_t *n_d = (int64_t *)tmp_get(bufpool, sizeof(*n_d));

		cubEncode(uv_sort_d + offs, uvu_d, values_d, n_d, subm_ned[i]);

		int64_t nuvuniq;
		CHECK_CUDA(cudaMemcpy(&nuvuniq, n_d, sizeof(nuvuniq), cudaMemcpyDeviceToHost));
		if (nuvuniq > LOCINT_MAX) {
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			fprintf(stderr, "Too many nonzeroes for processor %d: %" PRId64 "!\n", rank, nuvuniq);
			exit(EXIT_FAILURE);
		}
		nnz[i] = nuvuniq;
	
		u_d = (LOCINT *)tmp_get(bufpool, nnz[i]*sizeof(*u_d));
		v_d = (LOCINT *)tmp_get(bufpool, nnz[i]*sizeof(*v_d));

		// build csr's row offset array
		unzip<<<DIV_UP(nnz[i],THREADS), THREADS>>>(nnz[i], uvu_d, u_d, v_d);
		CHECK_ERROR("unzip");

		LOCINT *uu_d = (LOCINT *)tmp_get(bufpool, nnz[i]*sizeof(*uu_d));
		LOCINT *uc_d = (LOCINT *)tmp_get(bufpool, nnz[i]*sizeof(*uc_d));

		cubEncode(u_d, uu_d, uc_d, n_d, nnz[i]);
		tmp_release(bufpool, uvu_d);
		tmp_release(bufpool, u_d);

		int64_t nrows64;
		CHECK_CUDA(cudaMemcpy(&nrows64, n_d, sizeof(nrows64), cudaMemcpyDeviceToHost));
		nrows[i] = nrows64;
		
		LOCINT *offs_d = (LOCINT *)tmp_get(bufpool, (nrows[i]+1)*sizeof(*offs_d));
		tmp_release(bufpool, n_d);
		
		cubExclusiveSum(uc_d, offs_d, nrows[i]);
		CHECK_CUDA(cudaMemcpy(offs_d + nrows[i], nnz+i, sizeof(*offs_d), cudaMemcpyHostToDevice));
		tmp_release(bufpool, uc_d);

		tmp_detach(bufpool, offs_d);
		roff_d[i] = offs_d;
		
		tmp_detach(bufpool, uu_d);
		rows_d[i] = uu_d;

		tmp_detach(bufpool, v_d);
		cols_d[i] = v_d; 

		tmp_detach(bufpool, values_d);
		vals_d[i] = values_d;

		offs += subm_ned[i];
	}

	tmp_release(bufpool, uv_sort_d);
	free(subm_ned);
	return;
}

template<typename SRCT, typename MAPT, typename MAPLT, typename DSTT>
__global__ void scatter_asum(SRCT *src, MAPT *map, MAPLT maplen, MAPT mapoff, DSTT *dst) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < maplen) atomicAdd(dst + map[tid] + mapoff, (DSTT)src[tid]);
	return;
}

__global__ void update_vals(REAL *vals, LOCINT *cols, uint32_t nnz, LOCINT off, int *sums) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < nnz) vals[tid] /= (REAL)sums[cols[tid] + off];
	return;
}

void normalize_cols(LOCINT nnz, LOCINT *cols_d, REAL *vals_d, LOCINT N, LOCINT off, MPI_Comm COMM) {

	int *sum_d = (int *)tmp_get(bufpool, N*sizeof(*sum_d));
	
	setval<int><<<DIV_UP(N,THREADS), THREADS>>>(sum_d, N, 0);
	CHECK_ERROR("setval");

	scatter_asum<<<DIV_UP(nnz,THREADS), THREADS>>>(vals_d, cols_d, nnz, off, sum_d);
	CHECK_ERROR("scatter_asum");

	int *sum_h = (int *)Malloc(N*sizeof(*sum_d));
	CHECK_CUDA(cudaMemcpy(sum_h, sum_d, N*sizeof(*sum_h), cudaMemcpyDeviceToHost));
	MPI_Allreduce(MPI_IN_PLACE, sum_h, N, MPI_INT, MPI_SUM, COMM);
	CHECK_CUDA(cudaMemcpy(sum_d, sum_h, N*sizeof(*sum_h), cudaMemcpyHostToDevice));

	update_vals<<<DIV_UP(nnz,THREADS), THREADS>>>(vals_d, cols_d, nnz, off, sum_d);
	CHECK_ERROR("update_vals");

	CHECK_CUDA(cudaDeviceSynchronize());

	tmp_release(bufpool, sum_d);
	free(sum_h);
	return;
}

void normalize_cols_multi(LOCINT *nnz, LOCINT **cols_d, REAL **vals_d, LOCINT *last_row, int ncsr, MPI_Comm COMM) {

	for(int i = 0; i < ncsr; i++) {
		LOCINT firstcol = (i > 0) ? last_row[i-1]+1 : 0;
		LOCINT lastcol  = last_row[i];
		normalize_cols(nnz[i], cols_d[i], vals_d[i], lastcol-firstcol+1, -firstcol, COMM);
	}
	return;
}

__device__ __host__ inline bool operator==(const LOCINT2 &lhs, const LOCINT2 &rhs) { 
	return (lhs.x == rhs.x && lhs.y == rhs.y);
}

struct ValidColSelect {
	int64_t	n;
	LOCINT	*d_ptr;

	__device__ CUB_RUNTIME_FUNCTION __forceinline__ ValidColSelect(LOCINT *d_ptr, int64_t n) : d_ptr(d_ptr), n(n) {}
	__device__ CUB_RUNTIME_FUNCTION __forceinline__ bool operator()(const LOCINT2 v) const {
		return (bsearch<LOCINT>(d_ptr, n, v.y) < 0);
	}
};


template<typename T>
__global__ void setseq(T a, T b, T *v, T n) {
	
	const T   rng = b-a+1;  
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (tid < n) v[tid] = a + tid-(tid/rng)*rng; /* a + tid%rng */
	return;
}

template<typename MAPT, typename VALT>
__global__ void perm_v(MAPT n, VALT *v_in, MAPT *rmap, VALT *v_out) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (tid < n) 
		v_out[tid] = v_in[rmap[tid]];

	return;
}
	
template<typename T>
__global__ void getrlen(T *roff, T nr, T *rlen) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	
	if (tid < nr)
		rlen[tid] = roff[tid+1]-roff[tid];
	return;
}

// simple, warp-based, rows permutating kernel
template<int NWARP,
	 int RXW,
	 typename OFFT,
	 typename MAPT,
	 typename VALT>
__global__ void perm_m(MAPT *rmap, OFFT nr, OFFT *offs_in, VALT *v_in, OFFT *offs_out, VALT *v_out) {

	assert(RXW <= 32);

	const int rid_out = (blockIdx.x*NWARP + threadIdx.y)*RXW;

	OFFT  soff_out=0;
	OFFT  soff_in=0;
#ifndef NDEBUG
	OFFT eoff_in=0;
#endif
	if (threadIdx.x < RXW && rid_out+threadIdx.x <= nr) {
		if (rid_out+threadIdx.x < nr) {
			const MAPT rid_in = rmap[rid_out+threadIdx.x];
			soff_in  = offs_in[rid_in];
#ifndef NDEBUG
			eoff_in  = offs_in[rid_in+1];
#endif
		}
		soff_out = offs_out[rid_out+threadIdx.x];
	}

	OFFT start_out = __shfl(soff_out, 0);
	//OFFT start_in =  __shfl(soff_in, 0);

	#pragma unroll
	for(int i = 0; i < RXW; i++) {

		if (rid_out+i >= nr) break;

		const OFFT end_out = (i < RXW-1) ? __shfl(soff_out, i+1) : offs_out[rid_out+RXW];
		const OFFT start_in  = __shfl(soff_in, i);
#ifndef NDEBUG
		const OFFT end_in  = __shfl(eoff_in, i);
		assert(end_out-start_out == end_in-start_in);
#endif
		for(OFFT j = threadIdx.x; j < end_out-start_out; j += 32)
			v_out[start_out + j] = v_in[start_in + j];

		start_out = end_out;
	}
	return;
}

template<int NTH, typename T>
__global__ void get_kthres_offs(T n, T *val, T thr0, T thr1, T *offs) {

	__shared__ T sh[NTH];
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;

	sh[threadIdx.x] = (tid < n) ? val[tid] : 0;
	__syncthreads();
	
	if (tid < n) {
		T vcurr = sh[threadIdx.x];
		T vprev = (threadIdx.x > 0) ? sh[threadIdx.x-1] : ((blockIdx.x > 0) ? val[tid-1] : vcurr);

		if (vprev >= thr0 && vcurr < thr0) offs[0] = tid;
		if (vprev >= thr1 && vcurr < thr1) offs[1] = tid;
		if (tid == 0) {
			if (vcurr < thr0) offs[0] = 0;
			if (vcurr < thr1) offs[1] = 0;
		}
		if (tid == n-1) {
			if (vcurr >= thr0) offs[0] = n;
			if (vcurr >= thr1) offs[1] = n;
		}
	}
	return;
}

void sort_csr(LOCINT nnz, LOCINT nrows, LOCINT *kthr, LOCINT **rows,
	      LOCINT **roff, LOCINT **cols, REAL **vals, LOCINT *koff) {

	LOCINT *rlens = (LOCINT *)tmp_get(bufpool, nrows*sizeof(*rlens));
	getrlen<LOCINT><<<DIV_UP(nrows,THREADS), THREADS>>>(roff[0], nrows, rlens);
	CHECK_ERROR("getrlen");

	LOCINT *seq = (LOCINT *)tmp_get(bufpool, nrows*sizeof(*seq));
	setseq<LOCINT><<<DIV_UP(nrows,THREADS), THREADS>>>(0, nrows-1, seq, nrows);
	CHECK_ERROR("setseq");
	
	LOCINT *lenSortDef = (LOCINT *)tmp_get(bufpool, (nrows+1)*sizeof(*lenSortDef));
	LOCINT *srcPermDef = (LOCINT *)tmp_get(bufpool, nrows*sizeof(*srcPermDef));

	cubSortPairsDescending(rlens, lenSortDef, seq, srcPermDef, nrows);
	tmp_release(bufpool, rlens);

	LOCINT *lenSortOpt = (LOCINT *)tmp_get(bufpool, (nrows+1)*sizeof(*lenSortOpt));
	LOCINT *srcPermOpt = (LOCINT *)tmp_get(bufpool, nrows*sizeof(*srcPermOpt));

	LOCINT *koff_d = (LOCINT *)tmp_get(bufpool, 2*sizeof(*koff_d));
	get_kthres_offs<THREADS, LOCINT><<<DIV_UP(nrows,THREADS), THREADS>>>(nrows, lenSortDef, kthr[0], kthr[1], koff_d);
	CHECK_ERROR("get_kthres_offs");

	CHECK_CUDA(cudaMemcpy(koff, koff_d, 2*sizeof(*koff), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, koff_d);
	// second-sort rows for block-based SPMV kernel [0,koff[0])
	if (koff[0] && RXCTA_BL > 1) {
		LOCINT nr0 = koff[0];
		setseq<LOCINT><<<DIV_UP(nr0,THREADS), THREADS>>>(0, DIV_UP(nr0, RXCTA_BL)-1, seq, nr0);
		CHECK_ERROR("setseq");

		LOCINT *dummy = (LOCINT *)tmp_get(bufpool, nr0*sizeof(*dummy));
		cubSortPairs(seq, dummy, lenSortDef, lenSortOpt, nr0);
		cubSortPairs(seq, dummy, srcPermDef, srcPermOpt, nr0);
		tmp_release(bufpool, dummy);
	} else {
		LOCINT nr0 = koff[0];
		CHECK_CUDA(cudaMemcpy(lenSortOpt, lenSortDef, nr0*sizeof(*lenSortOpt), cudaMemcpyDeviceToDevice));
		CHECK_CUDA(cudaMemcpy(srcPermOpt, srcPermDef, nr0*sizeof(*srcPermOpt), cudaMemcpyDeviceToDevice));
	}
	// second-sort rows for warp-based SPMV kernel [koff[0], koff[1])
	if (koff[0] < koff[1] && RXCTA_WP > 1) {
		LOCINT nr1 = koff[1]-koff[0];
		setseq<LOCINT><<<DIV_UP(nr1,THREADS), THREADS>>>(0, DIV_UP(nr1, RXCTA_WP)-1, seq, nr1);
		CHECK_ERROR("setseq");

		LOCINT *dummy = (LOCINT *)tmp_get(bufpool, nr1*sizeof(*dummy));
		cubSortPairs(seq, dummy, lenSortDef+koff[0], lenSortOpt+koff[0], nr1);
		cubSortPairs(seq, dummy, srcPermDef+koff[0], srcPermOpt+koff[0], nr1);
		tmp_release(bufpool, dummy);
	} else {
		uint32_t nr1 = koff[1]-koff[0];
		CHECK_CUDA(cudaMemcpy(lenSortOpt+koff[0], lenSortDef+koff[0], nr1*sizeof(*lenSortOpt), cudaMemcpyDeviceToDevice));
		CHECK_CUDA(cudaMemcpy(srcPermOpt+koff[0], srcPermDef+koff[0], nr1*sizeof(*srcPermOpt), cudaMemcpyDeviceToDevice));
	}
	// second-sort rows for thread-based SPMV kernel [koff[1], nrows)
	if (koff[1] < nrows && RXCTA_TH > 1) {
		LOCINT nr2 = nrows-koff[1];
		setseq<LOCINT><<<DIV_UP(nr2,THREADS), THREADS>>>(0, DIV_UP(nr2,RXCTA_TH)-1, seq, nr2);
		CHECK_ERROR("setseq");

		LOCINT *dummy = (LOCINT *)tmp_get(bufpool, nr2*sizeof(*dummy));
		cubSortPairs(seq, dummy, lenSortDef+koff[1], lenSortOpt+koff[1], nr2);
		cubSortPairs(seq, dummy, srcPermDef+koff[1], srcPermOpt+koff[1], nr2);
		tmp_release(bufpool, dummy);
	} else {
		LOCINT nr2 = nrows-koff[1];
		CHECK_CUDA(cudaMemcpy(lenSortOpt+koff[1], lenSortDef+koff[1], nr2*sizeof(*lenSortOpt), cudaMemcpyDeviceToDevice));
		CHECK_CUDA(cudaMemcpy(srcPermOpt+koff[1], srcPermDef+koff[1], nr2*sizeof(*srcPermOpt), cudaMemcpyDeviceToDevice));
	}

	tmp_release(bufpool, seq);
	tmp_release(bufpool, srcPermDef);
	tmp_release(bufpool, lenSortDef);

	// compute sorted row offsets
	LOCINT *roffSort;
	CHECK_CUDA(cudaMalloc(&roffSort, (nrows+1)*sizeof(*roffSort)));
	cubExclusiveSum(lenSortOpt, roffSort, nrows+1);
	tmp_release(bufpool, lenSortOpt);
	
	// permute row indices
	LOCINT   *rowsSort;
	CHECK_CUDA(cudaMalloc(&rowsSort, nrows*sizeof(*rowsSort)));
	perm_v<LOCINT, LOCINT><<<DIV_UP(nrows,THREADS), THREADS>>>(nrows, rows[0], srcPermOpt, rowsSort);
	CHECK_ERROR("perm_v");
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(rows[0]));
	rows[0] = rowsSort;

	// permute cols
	LOCINT	 *colsSort;
	CHECK_CUDA(cudaMalloc(&colsSort, nnz*sizeof(*colsSort)));
	perm_m<THREADS/32,32><<<DIV_UP(DIV_UP(nrows,32), THREADS/32), dim3(32,THREADS/32)>>>(srcPermOpt, nrows, roff[0], cols[0], roffSort, colsSort);
	CHECK_ERROR("perm_m");
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(cols[0]));
	cols[0] = colsSort;

	// permute vals
	REAL	 *valsSort;
	CHECK_CUDA(cudaMalloc(&valsSort, nnz*sizeof(*valsSort)));
	perm_m<THREADS/32,32><<<DIV_UP(DIV_UP(nrows,32), THREADS/32), dim3(32,THREADS/32)>>>(srcPermOpt, nrows, roff[0], vals[0], roffSort, valsSort);
	CHECK_ERROR("perm_m");
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaFree(vals[0]));
	vals[0] = valsSort;

	//perm[0] = srcPermOpt;
	tmp_release(bufpool, srcPermOpt);
	
	// rows no longer needed...	
	CHECK_CUDA(cudaFree(roff[0]));
	roff[0] = roffSort;
	CHECK_CUDA(cudaMemcpy(roff[0]+nrows, &nnz, sizeof(**roff), cudaMemcpyHostToDevice));

	return;
}

template<int NT>
__global__ void find_recvoffs(LOCINT *cols, int64_t ncol, LOCINT *maxrow_all, int ntask, int64_t*roffs) {
	
	const int64_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	int mytask;
	__shared__ int task[NT];

	if (tid < ncol) {
		mytask = bsectl(maxrow_all, ntask, cols[tid]);
		task[threadIdx.x] = mytask;
	}
	__syncthreads();
	if (tid < ncol) {
		int ptask;
		if (threadIdx.x > 0) ptask = task[threadIdx.x-1];
		else		     ptask = (blockIdx.x == 0) ? -1 : bsectl(maxrow_all, ntask, cols[tid-1]);

		if (ptask != mytask) roffs[mytask] = tid;
	}
	return;
}

#define NEXT_PROC(i) (((rank)+(i))%(ntask))
#define PREV_PROC(i) (((rank)-(i)+(ntask))%(ntask))
void get_extdata_cuda(int ncsr, LOCINT *nnz, LOCINT **cols_d, LOCINT *lastrow_all,
		      int **recvNeigh, int *recvNum, int64_t **recvCnt, int64_t **recvOff, int64_t *totRecv,
		      int **sendNeigh, int *sendNum, int64_t **sendCnt, int64_t **sendOff, int64_t *totSend,
		      LOCINT **sendRows, MPI_Comm COMM) {

	int rank, ntask;
	MPI_Comm_size(COMM, &ntask);
	MPI_Comm_rank(COMM, &rank);

	if (ntask == 1) {
		recvNum[0] = 0;
		totRecv[0] = 0;
		recvNeigh[0] = NULL;
		recvCnt[0] = NULL;
		recvOff[0] = NULL;
		return;
	}

	LOCINT maxcol = lastrow_all[0]+1;
	for(int i = 1; i < ncsr; i++) {
		maxcol = max(maxcol, lastrow_all[i]-lastrow_all[i-1]);
	}

	LOCINT  *recvRows, *recvRows_m;
	recvRows = (LOCINT *)Malloc(ntask*maxcol*sizeof(*recvRows));
	CHECK_CUDA(cudaHostRegister(recvRows, ntask*maxcol*sizeof(*recvRows), cudaHostRegisterMapped));
	CHECK_CUDA(cudaHostGetDevicePointer((void **)&(recvRows_m), recvRows, 0) );

	// initial guess
	size_t sRows_len = maxcol;
	LOCINT *sRows = (LOCINT *)Malloc(sRows_len*sizeof(*sRows));

	int	*rneigh = (int     *)Malloc((ntask-1)*sizeof(*rneigh));
	int	*sneigh = (int     *)Malloc((ntask-1)*sizeof(*sneigh));

	int64_t	*rcount = (int64_t *)Malloc((ntask-1)*sizeof(*rcount));
	int64_t	*scount = (int64_t *)Malloc((ntask-1)*sizeof(*scount));

	int64_t	*roffs  = (int64_t *)Malloc((ntask-1)*sizeof(*roffs));
	int64_t	*soffs  = (int64_t *)Malloc((ntask-1)*sizeof(*soffs));

	LOCINT maxnnz = 0;
	for(int i = 0; i < ncsr; i++) {
		maxnnz = max(maxnnz, nnz[i]);
	}
	LOCINT *cols_sort_d = (LOCINT *)tmp_get(bufpool, maxnnz*sizeof(*cols_sort_d));

	int	nrecv=0;
	int	nsend=0;
	// int64_t groff=0;
	// int64_t gsoff=0;
	size_t groff=0;
	size_t gsoff=0;

	int64_t *n_d=(int64_t *)tmp_get(bufpool, sizeof(*n_d));

	roffs[0] = 0;
	soffs[0] = 0;

	for(int i = 1; i < ntask; i++) {

		int nextp = NEXT_PROC(i);
		int prevp = PREV_PROC(i);

		cubSortKeys(cols_d[nextp], cols_sort_d, nnz[nextp]);
		cubUnique(cols_sort_d, recvRows_m, n_d, nnz[nextp]);
		CHECK_CUDA(cudaMemcpy(rcount + nrecv,
				      n_d,
				      sizeof(*rcount),
				      cudaMemcpyDeviceToHost));

		MPI_Sendrecv(rcount + nrecv, 1, MPI_LONG_LONG, nextp, TAG(rank),
			     scount + nsend, 1, MPI_LONG_LONG, prevp, TAG(prevp),
            		     COMM, MPI_STATUS_IGNORE);

		if (gsoff+scount[nsend] > sRows_len) {
			sRows = (LOCINT *)Realloc(sRows, (sRows_len + maxcol)*sizeof(*sRows));
			sRows_len += maxcol;
		}
		MPI_Sendrecv(recvRows,      rcount[nrecv], LOCINT_MPI, nextp, TAG(rank),
			     sRows + gsoff, scount[nsend], LOCINT_MPI, prevp, TAG(prevp),
            		     COMM, MPI_STATUS_IGNORE);

		roffs[nrecv] = groff;
		soffs[nsend] = gsoff;

		rneigh[nrecv] = nextp;
		sneigh[nsend] = prevp;

		groff += rcount[nrecv];
		gsoff += scount[nsend];

		nrecv++;
		nsend++;
	}

	recvNum[0] = nrecv;
	totRecv[0] = groff;

	recvNeigh[0] = rneigh;
	recvCnt[0] = rcount;
	recvOff[0] = roffs;
	
	sendRows[0] = sRows;
	sendNum[0] = nsend;
	totSend[0] = gsoff;

	sendNeigh[0] = sneigh;
	sendCnt[0] = scount;
	sendOff[0] = soffs;

	tmp_release(bufpool, n_d);
	tmp_release(bufpool, cols_sort_d);
	
	CHECK_CUDA(cudaHostUnregister(recvRows));
	free(recvRows);
	return;
}

template<typename VALT, typename LENT>
__global__ void	vsub(VALT *v, LENT n, VALT tr) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) v[tid] -= tr;
	return;
}

template<typename VALT, typename LENT>
__global__ void	column_map(VALT *cols, LENT ncol, VALT frow, VALT *map) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < ncol) {
		map[cols[tid] - frow] = 1;
	}
	return;
}

template<typename VALT, typename LENT>
__global__ void	relabel_multi(VALT *cols, LENT ncol, VALT frow, VALT off, VALT *csum) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < ncol) {
		LOCINT col = cols[tid];
		cols[tid] = csum[col - frow] + off;
	}
	return;
}

void relabel_cuda_multi(LOCINT *lastrow_all, int ncsr, LOCINT *nrows,
			LOCINT **rows_d, LOCINT *nnz, LOCINT **cols_d,
			int64_t totToSend, LOCINT *rowsToSend_d, MPI_Comm COMM) {

	int rank;
	MPI_Comm_rank(COMM, &rank);

	LOCINT maxcol = lastrow_all[0]+1;
	for(int i = 1; i < ncsr; i++) {
		maxcol = max(maxcol, lastrow_all[i]-lastrow_all[i-1]);
	}

	LOCINT *cmap_d = (LOCINT *)tmp_get(bufpool, (maxcol+1)*sizeof(*cmap_d));
	LOCINT *csum_d = (LOCINT *)tmp_get(bufpool, (maxcol+1)*sizeof(*csum_d));

	LOCINT myfirst_row = rank ? lastrow_all[rank-1]+1 : 0;
	LOCINT excol_off = lastrow_all[rank] - myfirst_row + 1;

	for(int j = 0; j < ncsr; j++) {
		int i = (rank+j) % ncsr;

		LOCINT first_row = i ? lastrow_all[i-1]+1 : 0;
		LOCINT curr_cols = lastrow_all[i]-first_row+1;

		if (i == rank) {
			vsub<<<DIV_UP(nnz[i],THREADS), THREADS>>>(cols_d[i], nnz[i], first_row);
			if (totToSend)
				vsub<<<DIV_UP(totToSend,THREADS), THREADS>>>(rowsToSend_d, totToSend, first_row);
			CHECK_ERROR("vsub");
		} else {
			CHECK_CUDA(cudaMemset(cmap_d, 0, (curr_cols+1)*sizeof(*cmap_d)));

			column_map<<<DIV_UP(nnz[i],THREADS), THREADS>>>(cols_d[i], nnz[i], first_row, cmap_d);
			CHECK_ERROR("column_map");
			cubExclusiveSum(cmap_d, csum_d, curr_cols+1);

			relabel_multi<<<DIV_UP(nnz[i],THREADS), THREADS>>>(cols_d[i], nnz[i], first_row, excol_off, csum_d);
			CHECK_ERROR("relabel_multi");

			LOCINT tmp;
			CHECK_CUDA(cudaMemcpy(&tmp, csum_d+curr_cols, sizeof(LOCINT), cudaMemcpyDeviceToHost));
			excol_off += tmp;
		}
		vsub<<<DIV_UP(nrows[i], THREADS), THREADS>>>(rows_d[i], nrows[i], myfirst_row);
		CHECK_ERROR("vsub");
	}
	CHECK_CUDA(cudaDeviceSynchronize());

	tmp_release(bufpool, cmap_d);
	tmp_release(bufpool, csum_d);

	return;
}

__global__ void	setrnd(REAL *v, LOCINT n, uint64_t seed) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) {
		curandStatePhilox4_32_10_t state;
		curand_init(seed, tid, 0, &state);
#if REAL_SIZE == 8
		v[tid] = curand_uniform_double(&state);
#else
		v[tid] = curand_uniform(&state);
#endif
	}
	return;
}

template<typename T>
__global__ void	vscale(T *v, LOCINT n, T f) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) v[tid] *= f;
	return;
}

template<typename T>
__global__ void gather(T *src, LOCINT *map, uint32_t maplen, T *dst) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < maplen) dst[tid] = src[map[tid]];
	return;
}

template<typename T>
__global__ void scatter(T *src, LOCINT *map, uint32_t maplen, T *dst) {
	
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < maplen) dst[map[tid]] = src[tid];
	return;
}

void generate_rhs(int64_t intColsNum, REAL *r) {

	REAL lsum, gsum;

	setrnd<<<(intColsNum + THREADS-1)/THREADS, THREADS>>>(r, intColsNum, 0);
	CHECK_ERROR("setrnd");

	cubSum(r, r+intColsNum, intColsNum);
	CHECK_CUDA(cudaMemcpy(&lsum, r+intColsNum, sizeof(lsum), cudaMemcpyDeviceToHost));

	MPI_Allreduce(&lsum, &gsum, 1, REAL_MPI, MPI_SUM, MPI_COMM_WORLD);

	vscale<REAL><<<DIV_UP(intColsNum,THREADS), THREADS>>>(r, intColsNum, REALV(1.0)/gsum);
	CHECK_ERROR("vscale");

	CHECK_CUDA(cudaDeviceSynchronize());
	return;
}

void getSendElems(REAL *r, LOCINT *rowsToSend, int64_t totToSend, REAL *sendBuffer, cudaStream_t stream) {

	if (totToSend) {
		gather<REAL><<<DIV_UP(totToSend,THREADS), THREADS, 0, stream>>>(r, rowsToSend, totToSend, sendBuffer);
		CHECK_ERROR("gather");
		//CHECK_CUDA(cudaDeviceSynchronize());
	}
	return;
}

REAL reduce_cuda(REAL *v, int64_t n) {
	
	REAL 	sum_h, *sum_d=NULL;

	sum_d = (REAL *)tmp_get(bufpool, sizeof(*sum_d));
	cubSum(v, sum_d, n);
	CHECK_CUDA(cudaMemcpy(&sum_h, sum_d, sizeof(sum_h), cudaMemcpyDeviceToHost));
	tmp_release(bufpool, sum_d);

	return sum_h;
}

void reduce_cuda_async(REAL *v, int64_t n, REAL *sum_h, cudaStream_t stream) {

	static int	ftime = 1;	
	static REAL	*sum_d=NULL;

	if (ftime) {
		sum_d = (REAL *)tmp_get(bufpool, sizeof(*sum_d));
		ftime = 0;
	}
	cubSum(v, sum_d, n, stream);
	CHECK_CUDA(cudaMemcpyAsync(sum_h, sum_d, sizeof(sum_h), cudaMemcpyDeviceToHost, stream));

	return;
}

void setarray(REAL *arr, int64_t n, REAL val, cudaStream_t stream) {

	if (n) {
		setval<REAL><<<DIV_UP(n,THREADS), THREADS, 0, stream>>>(arr, n, val);
		CHECK_ERROR("setval");
		//CHECK_CUDA(cudaDeviceSynchronize());
	}
	return;
}

template<typename T>
__global__ void setval_rng(T *arr, int64_t n, int64_t ioff, int64_t inum, T vint, T vext) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n)
		arr[tid] = (tid < ioff+inum) ? ((tid < ioff) ? vext : vint) : (vext);
	return;
}

void cleanup_cuda() {

	tmp_destroy(bufpool);
	return;
}

__global__ void axpb_k(REAL *v, int64_t n, REAL x, REAL b) {

	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	if (tid < n) v[tid] = v[tid]*x + b;
	return;

}

// Kernel3 kenrels (SPMV) //////////////////////////////////////////////////////

__device__ __inline__ double __shfl_xor64(double x, int mask) {
	// Split the double number into 2 32b registers.
	int lo, hi;
	asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

	// Shuffle the two 32b registers.
	lo = __shfl_xor(lo, mask);
	hi = __shfl_xor(hi, mask);

	// Recreate the 64b number.
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
	return x;
}

__device__ __inline__ double __shfl_down64(double x, int delta) {
	// Split the double number into 2 32b registers.
	int lo, hi;
	asm volatile( "mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

	// Shuffle the two 32b registers.
	lo = __shfl_down(lo, delta);
	hi = __shfl_down(hi, delta);

	// Recreate the 64b number.
	asm volatile( "mov.b64 %0, {%1,%2};" : "=d"(x) : "r"(lo), "r"(hi) );
	return x;
}

template<typename T>
__device__ __forceinline__ T block_bcast(const T val, const int tid) {

	__shared__ T sh;

	__syncthreads();
	if (threadIdx.x == tid)	sh = val;
	__syncthreads();

	return sh;
}

template<int WSIZE, int NWARP, typename T>
__device__ __forceinline__ T block_reduce(T val) {

	const int wid = threadIdx.x / WSIZE;
	const int lid = threadIdx.x % WSIZE;
	__shared__ T sh[NWARP];

	#pragma unroll
	for(int l = (WSIZE>>1); l > 0; l>>=1) {
#if REAL_SIZE == 8
		val += __shfl_xor64(val, l);
#else
		val += __shfl_xor(val, l);
#endif
	}
	__syncthreads();

	if (lid == 0) sh[wid] = val;
	__syncthreads();

	if (wid == 0) {
		val = (lid < NWARP) ? sh[lid] : 0;

		#pragma unroll
		for(int l = (NWARP>>1); l > 0; l>>=1) {
#if REAL_SIZE == 8
			val += __shfl_down64(val, l);
#else
			val += __shfl_down(val, l);
#endif
		}
		if (lid == 0) sh[0] = val;
	}
	__syncthreads();

	return sh[0];
}

template<int WSIZE, int NWARP, int NR, int ILP, typename T>
__global__ void spmv_warp_ACC(REAL c, const T *m, LOCINT nrow, const LOCINT *row,
			  const LOCINT *roff, const LOCINT *col, const T *__restrict r,
			  T *vout) {

	LOCINT wid = (blockIdx.x*NWARP + threadIdx.y)*NR;
	LOCINT soff = ((wid+threadIdx.x) <= nrow) ? roff[wid+threadIdx.x] : 0;
	LOCINT start = __shfl(soff, 0);
	LOCINT end;
	T      outv;
	int i;

	T sum[ILP+1];

	#pragma unroll
	for(i = 0; i < NR; i++) {

		if ((wid+i) >= nrow) break;
		end = (i < (NR-1)) ? __shfl(soff, i+1) : roff[wid + NR];

		sum[ILP] = 0;
		for(int j = start+threadIdx.x; j < end; j += ILP*WSIZE) {

			#pragma unroll
			for(int k = 0; k < ILP; k++) sum[k]=0;

			#pragma unroll
			for(int k = 0; k < ILP; k++) {
				if ((j + k*WSIZE) >= end) break;
				sum[k] = m[j + k*WSIZE] * LDG(r[col[j + k*WSIZE]]);
			}
			
			#pragma unroll
			for(int k = 0; k < ILP; k++) sum[ILP] += sum[k];
		}

		#pragma unroll
		for(int l = (WSIZE>>1); l > 0; l>>=1)
#if REAL_SIZE == 8
			sum[ILP] += __shfl_xor64(sum[ILP], l);
#else
			sum[ILP] += __shfl_xor(sum[ILP], l);
#endif

		if (threadIdx.x == i) outv = sum[ILP];
		start = end;
	}
	if (threadIdx.x < i) vout[row[wid + threadIdx.x]] += outv*c;
	return;
}

template<int WSIZE, int NWARP, int NR, typename T>
__global__ void spmv_block_ACC(T c, const T *m, LOCINT nrow, const LOCINT *row,
			       const LOCINT *roff, const LOCINT *col, const T *__restrict r,
			       T *vout) {

	assert(NR < WSIZE*NWARP);

	LOCINT const rid = blockIdx.x*NR;
	LOCINT const soff = ((threadIdx.x) < NR && (rid+threadIdx.x) <= nrow) ? roff[rid+threadIdx.x] : 0;

	LOCINT start = block_bcast<LOCINT>(soff, 0);
	LOCINT end;

	T	outv = 0;
	int	i;

	#pragma unroll
	for(i = 0; i < NR; i++) {

		if ((rid+i) >= nrow) break;
		end = (i < (NR-1)) ? block_bcast<LOCINT>(soff, i+1) : roff[rid + NR];
		
		T sum = 0;
		for(int j = start+threadIdx.x; j < end; j += WSIZE*NWARP) {
			sum += m[j] * LDG(r[col[j]]);
		}
		sum = block_reduce<WSIZE, NWARP, T>(sum);

		if (threadIdx.x == i) outv = sum;
		start = end;
	}
	if (threadIdx.x < i)
		vout[row[rid + threadIdx.x]] += outv*c;
	return;
}

template<int NT, int NR, typename T>
__global__ void spmv_thread_shm_ACC(T c, const T *m, LOCINT nrow, const LOCINT *row,
				const LOCINT *roff, const LOCINT *col, const T *__restrict r,
				T *vout) {
	assert(NR == 1);

	__shared__ REAL shr[NT]; 
	int *shi = (int *)shr; //sizeof(int) <= sizeof(REAL)

	const LOCINT rid = (blockIdx.x*blockDim.x + threadIdx.x)*NR;

	LOCINT	soff, eoff;
	int	start, end, rlen;

	start = (rid < nrow) ? roff[rid] : 0;
	shi[threadIdx.x] = start;
	__syncthreads();

	rlen = 0;
	soff = shi[0];
	if (rid < nrow) {
		if (blockIdx.x < gridDim.x-1) {
			rlen = ((threadIdx.x == NT-1) ? roff[rid+1] : shi[threadIdx.x+1]) - start;
		} else {
			rlen = ((rid == nrow-1) ? roff[rid+1] : shi[threadIdx.x+1]) - start;
		}
	}
	__syncthreads();

	shi[threadIdx.x] = rlen;
	__syncthreads();

	// psum
	int t = rlen;
	#pragma unroll
	for(int i = 1; i < NT; i <<= 1) {

		if (threadIdx.x >= i) t += shi[threadIdx.x-i];
		__syncthreads();

		if (threadIdx.x >= i) shi[threadIdx.x] = t;
		__syncthreads();
	}

	end = shi[threadIdx.x];
	start = end-rlen;
	eoff = soff + shi[NT-1];

	T sum = 0;
	for(; soff < eoff; soff += NT) {

		__syncthreads();
		if (soff+threadIdx.x < eoff)
			shr[threadIdx.x] = m[soff+threadIdx.x] * LDG(r[col[soff+threadIdx.x]]);

		__syncthreads();

		for(int i = start; i < min(NT,end); i++) sum += shr[i];

		start -= (start < NT) ? start : NT;
		end   -= (end   < NT) ?   end : NT;
	}
	if (rid < nrow) vout[row[rid]] += sum*c;
	return;
}

void computeSpmvAcc(REAL c, LOCINT nrows, LOCINT *rows, LOCINT *roff,
		    LOCINT *cols, REAL *vals, REAL *rsrc, REAL *rdst,
		    LOCINT *koff, cudaStream_t stream) {

	if (koff[0]) {
		LOCINT nr0 = koff[0];

		int nblocks = DIV_UP(nr0, RXCTA_BL);
		spmv_block_ACC<32, NTH_SPMV_BL/32, RXCTA_BL, REAL><<<nblocks, NTH_SPMV_BL, 0, stream>>>(c, vals, nr0, rows, roff, cols, rsrc, rdst);
		CHECK_ERROR("spmv_block_ACC");
	}
	if (koff[0] < koff[1]) {
		LOCINT nr1 = koff[1]-koff[0];

		dim3 block(32, NTH_SPMV_WP/32);
		dim3 grid(DIV_UP(DIV_UP(nr1, RXCTA_WP), NTH_SPMV_WP/32), 1, 1);
		spmv_warp_ACC<32, NTH_SPMV_WP/32, RXCTA_WP, 4, REAL><<<grid, block, 0, stream>>>(c, vals, nr1, rows+koff[0], roff+koff[0], cols, rsrc, rdst);
		CHECK_ERROR("spmv_warp_ACC");
	}
	if (koff[1] < nrows) {
		LOCINT nr2 = nrows-koff[1];
		int nblocks = DIV_UP(DIV_UP(nr2, RXCTA_TH), NTH_SPMV_TH);
		spmv_thread_shm_ACC<NTH_SPMV_TH, 1, REAL><<<nblocks, NTH_SPMV_TH, 0, stream>>>(c, vals, nr2, rows+koff[1], roff+koff[1], cols, rsrc, rdst);
		CHECK_ERROR("spmv_thread_shm_ACC");
	}

	return;
}

void sequence(LOCINT n, LOCINT *vec, LOCINT init = 0)
{
  thrust::sequence(thrust::device,thrust::device_pointer_cast(vec), thrust::device_pointer_cast(vec+n), init);
}

