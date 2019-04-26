/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 */

// Interanl helper functions 
// Author: Alex Fender afender@nvidia.com
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
//#include <library_types.h>
//#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <rmm_utils.h>

#define USE_CG 1
//#define DEBUG 1

namespace cugraph
{

#define CUDA_MAX_BLOCKS 65535
#define CUDA_MAX_KERNEL_THREADS 256  //kernel will launch at most 256 threads per block
#define DEFAULT_MASK 0xffffffff
#define US

//error check
#ifdef DEBUG
#define WHERE " at: " << __FILE__ << ':' << __LINE__
#define cudaCheckError() {                                              \
    cudaError_t e=cudaGetLastError();                                     \
    if(e!=cudaSuccess) {                                                  \
      std::cerr << "Cuda failure: "  << cudaGetErrorString(e) << WHERE << std::endl;        \
    }                                                                     \
  }
#else 
#define cudaCheckError()
#define WHERE ""
#endif 

	template<typename T>
	static __device__  __forceinline__ T shfl_up(T r, int offset, int bound = 32, int mask =
																									DEFAULT_MASK)
																							{
#if __CUDA_ARCH__ >= 300
#if USE_CG
		return __shfl_up_sync( mask, r, offset, bound );
#else
		return __shfl_up( r, offset, bound );
#endif
#else
		return 0.0f;
#endif
	}

	template<typename T>
	static __device__  __forceinline__ T shfl(T r, int lane, int bound = 32, int mask = DEFAULT_MASK)
																						{
#if __CUDA_ARCH__ >= 300
#if USE_CG
		return __shfl_sync(mask, r, lane, bound );
#else
		return __shfl(r, lane, bound );
#endif
#else
		return 0.0f;
#endif
	}

	template<typename IdxType, typename ValType>
	__inline__   __device__
	ValType parallel_prefix_sum(IdxType n, IdxType *ind, ValType *w) {
		IdxType i, j, mn;
		ValType v, last;
		ValType sum = 0.0;
		bool valid;

		//Parallel prefix sum (using __shfl)
		mn = (((n + blockDim.x - 1) / blockDim.x) * blockDim.x); //n in multiple of blockDim.x
		for (i = threadIdx.x; i < mn; i += blockDim.x) {
			//All threads (especially the last one) must always participate
			//in the shfl instruction, otherwise their sum will be undefined.
			//So, the loop stopping condition is based on multiple of n in loop increments,
			//so that all threads enter into the loop and inside we make sure we do not
			//read out of bounds memory checking for the actual size n.

			//check if the thread is valid
			valid = i < n;

			//Notice that the last thread is used to propagate the prefix sum.
			//For all the threads, in the first iteration the last is 0, in the following
			//iterations it is the value at the last thread of the previous iterations.

			//get the value of the last thread
			last = shfl(sum, blockDim.x - 1, blockDim.x);

			//if you are valid read the value from memory, otherwise set your value to 0
			sum = (valid) ? w[ind[i]] : 0.0;

			//do prefix sum (of size warpSize=blockDim.x =< 32)
			for (j = 1; j < blockDim.x; j *= 2) {
				v = shfl_up(sum, j, blockDim.x);
				if (threadIdx.x >= j)
					sum += v;
			}
			//shift by last
			sum += last;
			//notice that no __threadfence or __syncthreads are needed in this implementation
		}
		//get the value of the last thread (to all threads)
		last = shfl(sum, blockDim.x - 1, blockDim.x);

		return last;
	}

//dot
	template<typename T>
	T dot(size_t n, T* x, T* y) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);
		T result = thrust::inner_product(thrust::cuda::par(allocator).on(stream),
																			thrust::device_pointer_cast(x),
																			thrust::device_pointer_cast(x + n),
																			thrust::device_pointer_cast(y),
																			0.0f);
		cudaCheckError();
		return result;
	}

//axpy
	template<typename T>
	struct axpy_functor: public thrust::binary_function<T, T, T> {
		const T a;
		axpy_functor(T _a) :
				a(_a) {
		}
		__host__  __device__
		T operator()(const T& x, const T& y) const {
			return a * x + y;
		}
	};

	template<typename T>
	void axpy(size_t n, T a, T* x, T* y) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);
		thrust::transform(thrust::cuda::par(allocator).on(stream),
											thrust::device_pointer_cast(x),
											thrust::device_pointer_cast(x + n),
											thrust::device_pointer_cast(y),
											thrust::device_pointer_cast(y),
											axpy_functor<T>(a));
		cudaCheckError();
	}

//norm
	template<typename T>
	struct square {
		__host__  __device__
		T operator()(const T& x) const {
			return x * x;
		}
	};

	template<typename T>
	T nrm2(size_t n, T* x) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);
		T init = 0;
		T result = std::sqrt(thrust::transform_reduce(thrust::cuda::par(allocator).on(stream),
																									thrust::device_pointer_cast(x),
																									thrust::device_pointer_cast(x + n),
																									square<T>(),
																									init,
																									thrust::plus<T>()));
		cudaCheckError();
		return result;
	}

	template<typename T>
	T nrm1(size_t n, T* x) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);

		T result = thrust::reduce(thrust::cuda::par(allocator).on(stream),
															thrust::device_pointer_cast(x),
															thrust::device_pointer_cast(x + n));
		cudaCheckError();
		return result;
	}

	template<typename T>
	void scal(size_t n, T val, T* x) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);

		thrust::transform(thrust::cuda::par(allocator).on(stream),
											thrust::device_pointer_cast(x),
											thrust::device_pointer_cast(x + n),
											thrust::make_constant_iterator(val),
											thrust::device_pointer_cast(x),
											thrust::multiplies<T>());
		cudaCheckError();
	}

	template<typename T>
	void fill(size_t n, T* x, T value) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);

		thrust::fill(thrust::cuda::par(allocator).on(stream),
									thrust::device_pointer_cast(x),
									thrust::device_pointer_cast(x + n), value);
		cudaCheckError();
	}

	template<typename T>
	void printv(size_t n, T* vec, int offset) {
		thrust::device_ptr<T> dev_ptr(vec);
		std::cout.precision(15);
		std::cout << "sample size = " << n << ", offset = " << offset << std::endl;
		thrust::copy(dev_ptr + offset, dev_ptr + offset + n, std::ostream_iterator<T>(std::cout, " ")); //Assume no RMM dependency; TODO: check / test (potential BUG !!!!!)
		cudaCheckError();
		std::cout << std::endl;
	}

	template<typename T>
	void copy(size_t n, T *x, T *res)
						{
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);

		thrust::device_ptr<T> dev_ptr(x);
		thrust::device_ptr<T> res_ptr(res);
		thrust::copy_n(thrust::cuda::par(allocator).on(stream), dev_ptr, n, res_ptr);
		cudaCheckError();
	}

	template<typename T>
	struct is_zero {
		__host__ __device__
		bool operator()(const T x) {
			return x == 0;
		}
	};

	template<typename T>
	struct dangling_functor: public thrust::unary_function<T, T> {
		const T val;
		dangling_functor(T _val) :
				val(_val) {
		}
		__host__  __device__
		T operator()(const T& x) const {
			return val + x;
		}
	};

	template<typename T>
	void update_dangling_nodes(size_t n, T* dangling_nodes, T damping_factor) {
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);

		thrust::transform_if(thrust::cuda::par(allocator).on(stream),
													thrust::device_pointer_cast(dangling_nodes),
													thrust::device_pointer_cast(dangling_nodes + n),
													thrust::device_pointer_cast(dangling_nodes),
													dangling_functor<T>(1.0 - damping_factor),
													is_zero<T>());
		cudaCheckError();
	}

//google matrix kernels
	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	degree_coo(const IndexType n, const IndexType e, const IndexType *ind, IndexType *degree) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
			atomicAdd(&degree[ind[i]], 1.0);
	}
	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	equi_prob(const IndexType n,
						const IndexType e,
						const IndexType *ind,
						ValueType *val,
						IndexType *degree) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
			val[i] = 1.0 / degree[ind[i]];
	}

	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	flag_leafs(const IndexType n, IndexType *degree, ValueType *bookmark) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
			if (degree[i] == 0)
				bookmark[i] = 1.0;
	}

        template<typename IndexType, typename ValueType>
        __global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
        degree_offsets(const IndexType n, const IndexType e, const IndexType *ind, IndexType *degree) {
                for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x)
                        degree[i] += ind[i+1]-ind[i];
        }


//notice that in the transposed matrix/csc a dangling node is a node without incomming edges
//just swap coo src and dest arrays after that to interpret it as HT
	template<typename IndexType, typename ValueType>
	void HT_matrix_coo(	const IndexType n,
											const IndexType e,
											const IndexType *src,
											ValueType *cooVal,
											ValueType *bookmark) {
		IndexType *degree { nullptr };
		cudaStream_t stream { nullptr };
		ALLOC_MANAGED_TRY((void** )&degree, sizeof(IndexType) * n, stream);

		cudaMemset(degree, 0, sizeof(IndexType) * n);

		dim3 nthreads, nblocks;
		nthreads.x = min(e, CUDA_MAX_KERNEL_THREADS);
		nthreads.y = 1;
		nthreads.z = 1;
		nblocks.x = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
		nblocks.y = 1;
		nblocks.z = 1;
		degree_coo<IndexType, ValueType> <<<nblocks, nthreads>>>(n, e, src, degree);
		equi_prob<IndexType, ValueType> <<<nblocks, nthreads>>>(n, e, src, cooVal, degree);
		ValueType val = 0.0;
		fill(n, bookmark, val);
		nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
		nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
		flag_leafs<IndexType, ValueType> <<<nblocks, nthreads>>>(n, degree, bookmark);

		//printv(n, degree , 0);
		//printv(n, bookmark , 0);
		//printv(e, cooVal , 0);

		//this was missing: TODO: check if okay
		ALLOC_FREE_TRY(degree, stream);
	}

	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	equi_prob3(	const IndexType n,
							const IndexType e,
							const IndexType *csrPtr,
							const IndexType *csrInd,
							ValueType *val,
							IndexType *degree) {
		int j, row, col;
		for (row = threadIdx.z + blockIdx.z * blockDim.z; row < n; row += gridDim.z * blockDim.z) {
			for (j = csrPtr[row] + threadIdx.y + blockIdx.y * blockDim.y; j < csrPtr[row + 1];
					j += gridDim.y * blockDim.y) {
				col = csrInd[j];
				val[j] = 1.0 / degree[col];
				//val[j] = 999;
			}
		}
	}

	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	equi_prob2(	const IndexType n,
							const IndexType e,
							const IndexType *csrPtr,
							const IndexType *csrInd,
							ValueType *val,
							IndexType *degree) {
		int row = blockIdx.x * blockDim.x + threadIdx.x;
		if (row < n)
				{
			int row_begin = csrPtr[row];
			int row_end = csrPtr[row + 1];
			int col;
			for (int i = row_begin; i < row_end; i++) {
				col = csrInd[i];
				val[i] = 1.0 / degree[col];
			}
		}
	}

// compute the H^T values for an already transposed adjacency matrix, leveraging coo info
	template<typename IndexType, typename ValueType>
	void HT_matrix_csc_coo(	const IndexType n,
													const IndexType e,
													const IndexType *csrPtr,
													const IndexType *csrInd,
													ValueType *val,
													ValueType *bookmark) {
		IndexType *degree;
		cudaStream_t stream { nullptr };
		ALLOC_MANAGED_TRY((void** )&degree, sizeof(IndexType) * n, stream);
		cudaMemset(degree, 0, sizeof(IndexType) * n);

		dim3 nthreads, nblocks;
		nthreads.x = min(e, CUDA_MAX_KERNEL_THREADS);
		nthreads.y = 1;
		nthreads.z = 1;
		nblocks.x = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
		nblocks.y = 1;
		nblocks.z = 1;
		degree_coo<IndexType, ValueType> <<<nblocks, nthreads>>>(n, e, csrInd, degree);
		cudaCheckError();

		int y = 4;
		nthreads.x = 32 / y;
		nthreads.y = y;
		nthreads.z = 8;
		nblocks.x = 1;
		nblocks.y = 1;
		nblocks.z = min((n + nthreads.z - 1) / nthreads.z, CUDA_MAX_BLOCKS); //1;
		equi_prob3<IndexType, ValueType> <<<nblocks, nthreads>>>(n, e, csrPtr, csrInd, val, degree);
		//printv(e, val , 0);
		cudaCheckError();

		ValueType a = 0.0;
		fill(n, bookmark, a);
		cudaCheckError();

		nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
		nthreads.y = 1;
		nthreads.z = 1;
		nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
		nblocks.y = 1;
		nblocks.z = 1;
		flag_leafs<IndexType, ValueType> <<<nblocks, nthreads>>>(n, degree, bookmark);
		cudaCheckError();

		//this was missing! TODO: check if okay.
		ALLOC_FREE_TRY(degree, stream);
	}

	template<typename IndexType, typename ValueType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
	permute_vals_kernel(const IndexType e, IndexType *perm, ValueType *in, ValueType *out) {
		for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x)
			out[i] = in[perm[i]];
	}

	template<typename IndexType, typename ValueType>
	void permute_vals(const IndexType e, IndexType *perm, ValueType *in, ValueType *out) {
		int nthreads = min(e, CUDA_MAX_KERNEL_THREADS);
		int nblocks = min((e + nthreads - 1) / nthreads, CUDA_MAX_BLOCKS);
		permute_vals_kernel<<<nblocks, nthreads>>>(e, perm, in, out);
		//printv(e, in , 0);
		//printv(e, perm , 0);
		//printv(e, out , 0);
	}

// This will remove duplicate along with sorting
// This will sort the COO Matrix, row will be sorted and each column of same row will be sorted. 
	template<typename IndexType, typename ValueType, typename SizeT>
	void remove_duplicate(IndexType* src, IndexType* dest, ValueType* val, SizeT &nnz)
												{
		//RMM:
		//
		cudaStream_t stream { nullptr };
		rmm_temp_allocator allocator(stream);
		if (val != NULL)
				{
			thrust::stable_sort_by_key(	thrust::cuda::par(allocator).on(stream),
																	thrust::raw_pointer_cast(val),
																	thrust::raw_pointer_cast(val) + nnz,
																	thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src),
																																								thrust::raw_pointer_cast(dest))));
			thrust::stable_sort_by_key(	thrust::cuda::par(allocator).on(stream),
																	thrust::raw_pointer_cast(dest),
																	thrust::raw_pointer_cast(dest + nnz),
																	thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src),
																																								thrust::raw_pointer_cast(val))));
			thrust::stable_sort_by_key(	thrust::cuda::par(allocator).on(stream),
																	thrust::raw_pointer_cast(src),
																	thrust::raw_pointer_cast(src + nnz),
																	thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(dest),
																																								thrust::raw_pointer_cast(val))));

			typedef thrust::tuple<IndexType*, ValueType*> IteratorTuple;
			typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
			typedef thrust::tuple<IndexType*, ZipIterator> ZipIteratorTuple;
			typedef thrust::zip_iterator<ZipIteratorTuple> ZipZipIterator;

			ZipZipIterator newEnd =
					thrust::unique(	thrust::cuda::par(allocator).on(stream),
													thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src),
																																				thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(dest),
																																																											thrust::raw_pointer_cast(val))))),
													thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src
																																						+ nnz),
																																				thrust::make_zip_iterator(thrust::make_tuple(	dest
																																																													+ nnz,
																																																											val
																																																													+ nnz)))));

			ZipIteratorTuple endTuple = newEnd.get_iterator_tuple();
			IndexType* row_end = thrust::get<0>(endTuple);

			nnz = ((size_t) row_end - (size_t) src) / sizeof(IndexType);
		}
		else
		{
			thrust::stable_sort_by_key(	thrust::cuda::par(allocator).on(stream),
																	thrust::raw_pointer_cast(dest),
																	thrust::raw_pointer_cast(dest + nnz),
																	thrust::raw_pointer_cast(src));
			thrust::stable_sort_by_key(	thrust::cuda::par(allocator).on(stream),
																	thrust::raw_pointer_cast(src),
																	thrust::raw_pointer_cast(src + nnz),
																	thrust::raw_pointer_cast(dest));

			typedef thrust::tuple<IndexType*, IndexType*> IteratorTuple;
			typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

			ZipIterator newEnd =
					thrust::unique(	thrust::cuda::par(allocator).on(stream),
													thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src),
																																				thrust::raw_pointer_cast(dest))),
													thrust::make_zip_iterator(thrust::make_tuple(	thrust::raw_pointer_cast(src
																																						+ nnz),
																																				thrust::raw_pointer_cast(dest
																																						+ nnz))));

			IteratorTuple endTuple = newEnd.get_iterator_tuple();
			IndexType* row_end = thrust::get<0>(endTuple);

			nnz = ((size_t) row_end - (size_t) src) / sizeof(IndexType);
		}
	}

	template<typename IndexType>
	__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS) offsets_to_indices_kernel(	const IndexType *offsets,
																																												IndexType v,
																																												IndexType *indices) {

		int tid, ctaStart;
		tid = threadIdx.x;
		ctaStart = blockIdx.x;

		for (int j = ctaStart; j < v; j += gridDim.x) {
			IndexType colStart = offsets[j];
			IndexType colEnd = offsets[j + 1];
			IndexType rowNnz = colEnd - colStart;

			for (int i = 0; i < rowNnz; i += blockDim.x) {
				if ((colStart + tid + i) < colEnd) {
					indices[colStart + tid + i] = j;
				}
			}
		}
	}

	template<typename IndexType>
	void offsets_to_indices(const IndexType *offsets, IndexType v, IndexType *indices)
													{
		int nthreads = min(v, CUDA_MAX_KERNEL_THREADS);
		int nblocks = min((v + nthreads - 1) / nthreads, CUDA_MAX_BLOCKS);
		offsets_to_indices_kernel<<<nblocks, nthreads>>>(offsets, v, indices);
		cudaCheckError();
	}

	template<typename IndexType>
	void sequence(IndexType n, IndexType *vec, IndexType init = 0)
								{
		thrust::sequence(	thrust::device,
											thrust::device_pointer_cast(vec),
											thrust::device_pointer_cast(vec + n),
											init);
		cudaCheckError();
	}

} //namespace cugraph
